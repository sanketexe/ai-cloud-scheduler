import boto3
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from botocore.exceptions import ClientError, PaginationError

logger = logging.getLogger(__name__)

class AWSCollector:
    """
    Collector for AWS resources (EC2, EBS) and CloudWatch metrics.
    Uses IAM roles and handles pagination/throttling.
    """

    def __init__(self, region: str = "us-east-1", role_arn: Optional[str] = None):
        """
        Initialize AWS Collector.
        
        :param region: AWS region to collect from
        :param role_arn: Valid IAM Role ARN to assume (optional)
        """
        self.region = region
        self.role_arn = role_arn
        self._session = self._create_session()
        self.ec2_client = self._session.client("ec2", region_name=self.region)
        self.cloudwatch = self._session.client("cloudwatch", region_name=self.region)

    def _create_session(self) -> boto3.Session:
        """Create boto3 session, assuming role if ARN is provided."""
        if not self.role_arn:
            return boto3.Session()

        try:
            sts_client = boto3.client("sts")
            assumed_role_object = sts_client.assume_role(
                RoleArn=self.role_arn,
                RoleSessionName="CloudSchedulerCollectorSession"
            )
            credentials = assumed_role_object["Credentials"]
            return boto3.Session(
                aws_access_key_id=credentials["AccessKeyId"],
                aws_secret_access_key=credentials["SecretAccessKey"],
                aws_session_token=credentials["SessionToken"],
            )
        except ClientError as e:
            logger.error(f"Failed to assume role {self.role_arn}: {e}")
            raise

    def collect_resources(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Main entry point to collect all supported resources with metrics.
        
        :return: comprehensive dictionary of instances and volumes
        """
        logger.info(f"Starting resource collection in {self.region}")
        instances = self.get_ec2_instances()
        volumes = self.get_ebs_volumes()

        # Collect and attach EC2 metrics
        if instances:
            instance_ids = [i["instance_id"] for i in instances]
            ec2_metrics = self.collect_ec2_metrics(instance_ids)
            for instance in instances:
                instance["metrics"] = ec2_metrics.get(instance["instance_id"], {})

        # Collect and attach EBS metrics
        if volumes:
            volume_ids = [v["volume_id"] for v in volumes]
            ebs_metrics = self.collect_ebs_metrics(volume_ids)
            for volume in volumes:
                volume["metrics"] = ebs_metrics.get(volume["volume_id"], {})
        
        return {
            "instances": instances,
            "volumes": volumes
        }

    def get_ec2_instances(self) -> List[Dict[str, Any]]:
        """Fetch all EC2 instances with pagination."""
        instances = []
        paginator = self.ec2_client.get_paginator("describe_instances")
        
        try:
            for page in paginator.paginate():
                for reservation in page["Reservations"]:
                    for instance in reservation["Instances"]:
                        processed_instance = self._process_instance(instance)
                        instances.append(processed_instance)
        except (ClientError, PaginationError) as e:
            logger.error(f"Error collecting EC2 instances: {e}")
            
        logger.info(f"Collected {len(instances)} EC2 instances")
        return instances

    def _process_instance(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant fields from raw EC2 response."""
        return {
            "instance_id": instance["InstanceId"],
            "instance_type": instance["InstanceType"],
            "state": instance["State"]["Name"],
            "launch_time": instance["LaunchTime"].isoformat(),
            "region": self.region,
            "cpu_cores": instance.get("CpuOptions", {}).get("CoreCount", 1), # Default fallback
            # Note: Memory isn't directly in describe_instances, usually needs lookup table or simplified
            # For this example, we'll leave it or fetch from a type map if available
            "tags": instance.get("Tags", [])
        }

    def get_ebs_volumes(self) -> List[Dict[str, Any]]:
        """Fetch all EBS volumes with pagination."""
        volumes = []
        paginator = self.ec2_client.get_paginator("describe_volumes")

        try:
            for page in paginator.paginate():
                for volume in page["Volumes"]:
                    processed_volume = self._process_volume(volume)
                    volumes.append(processed_volume)
        except (ClientError, PaginationError) as e:
            logger.error(f"Error collecting EBS volumes: {e}")

        logger.info(f"Collected {len(volumes)} EBS volumes")
        return volumes

    def _process_volume(self, volume: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant fields from raw EBS response."""
        attached_instance_id = None
        if volume["Attachments"]:
            attached_instance_id = volume["Attachments"][0]["InstanceId"]

        return {
            "volume_id": volume["VolumeId"],
            "size_gb": volume["Size"],
            "volume_type": volume["VolumeType"],
            "state": volume["State"],
            "iops": volume.get("Iops"),
            "region": self.region,
            "attached_instance_id": attached_instance_id,
            "create_time": volume["CreateTime"].isoformat()
        }

    def collect_ec2_metrics(self, instance_ids: List[str], days: int = 7) -> Dict[str, Dict[str, Any]]:
        """
        Batch collect EC2 metrics for multiple instances.
        
        :param instance_ids: List of instance IDs
        :param days: Number of past days to fetch (default 7)
        :return: Nested dict {instance_id: {metric_name: [datapoints]}}
        """
        metrics_config = [
            ("CPUUtilization", "AWS/EC2", "Average"),
            ("NetworkIn", "AWS/EC2", "Average"),
            ("NetworkOut", "AWS/EC2", "Average")
        ]
        return self._batch_get_metrics(
            resource_ids=instance_ids,
            id_dimension="InstanceId",
            metrics_config=metrics_config,
            days=days
        )

    def collect_ebs_metrics(self, volume_ids: List[str], days: int = 7) -> Dict[str, Dict[str, Any]]:
        """
        Batch collect EBS metrics for multiple volumes.
        
        :param volume_ids: List of volume IDs
        :param days: Number of past days to fetch (default 7)
        :return: Nested dict {volume_id: {metric_name: [datapoints]}}
        """
        metrics_config = [
            ("VolumeReadOps", "AWS/EBS", "Sum"),
            ("VolumeWriteOps", "AWS/EBS", "Sum")
        ]
        return self._batch_get_metrics(
            resource_ids=volume_ids,
            id_dimension="VolumeId",
            metrics_config=metrics_config,
            days=days
        )

    def _batch_get_metrics(
        self,
        resource_ids: List[str],
        id_dimension: str,
        metrics_config: List[tuple],
        days: int
    ) -> Dict[str, Dict[str, Any]]:
        """
        Helper to fetch metrics in batches using CloudWatch GetMetricData.
        Handles query construction, batching, and result parsing.
        """
        results = {rid: {} for rid in resource_ids}
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        # Max 500 queries per GetMetricData call
        # Each resource has N metrics, so batch_size = floor(500 / N)
        metrics_per_resource = len(metrics_config)
        if metrics_per_resource == 0:
            return results
            
        batch_size = 500 // metrics_per_resource
        
        for i in range(0, len(resource_ids), batch_size):
            batch_ids = resource_ids[i:i + batch_size]
            queries = []
            query_map = {} # query_id -> (resource_id, metric_name)
            
            for idx, rid in enumerate(batch_ids):
                # Construct unique query IDs for this batch call
                # Format: q_{resource_index}_{metric_index}
                for m_idx, (metric_name, namespace, stat) in enumerate(metrics_config):
                    query_id = f"q_{idx}_{m_idx}" 
                    query_map[query_id] = (rid, metric_name)
                    
                    queries.append({
                        "Id": query_id,
                        "MetricStat": {
                            "Metric": {
                                "Namespace": namespace,
                                "MetricName": metric_name,
                                "Dimensions": [{"Name": id_dimension, "Value": rid}]
                            },
                            "Period": 3600, # 1 hour aggregation
                            "Stat": stat,
                        },
                        "ReturnData": True
                    })
            
            if not queries:
                continue

            try:
                paginator = self.cloudwatch.get_paginator('get_metric_data')
                # get_metric_data paginator handles "NextToken" automatically
                for page in paginator.paginate(
                    MetricDataQueries=queries,
                    StartTime=start_time,
                    EndTime=end_time
                ):
                    for metric_data in page["MetricDataResults"]:
                        qid = metric_data["Id"]
                        if qid in query_map:
                            rid, m_name = query_map[qid]
                            
                            datapoints = []
                            if "Timestamps" in metric_data and "Values" in metric_data:
                                for t, v in zip(metric_data["Timestamps"], metric_data["Values"]):
                                    datapoints.append({
                                        "Timestamp": t.isoformat(),
                                        "Value": v
                                    })
                            
                            # Sort by timestamp to ensure time-series order
                            datapoints.sort(key=lambda x: x["Timestamp"])
                            results[rid][m_name] = datapoints

            except ClientError as e:
                logger.error(f"Failed to fetch batch metrics for batch starting at index {i}: {e}")
        
        return results
