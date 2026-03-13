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

    def get_cloudwatch_metrics(
        self, 
        namespace: str, 
        metric_name: str, 
        dimensions: List[Dict[str, str]], 
        start_time: datetime, 
        end_time: datetime,
        period: int = 300,
        stat: str = "Average"
    ) -> List[Dict[str, Any]]:
        """
        Fetch metrics from CloudWatch.
        
        :param namespace: AWS/EC2, AWS/EBS, etc.
        :param metric_name: CPUUtilization, etc.
        :param dimensions: List of dimensions like [{'Name': 'InstanceId', 'Value': 'i-123'}]
        :param start_time: datetime object
        :param end_time: datetime object
        :param period: Granularity in seconds (default 5 mins)
        :param stat: Statistic to fetch (Average, Sum, Maximum)
        """
        try:
            response = self.cloudwatch.get_metric_statistics(
                Namespace=namespace,
                MetricName=metric_name,
                Dimensions=dimensions,
                StartTime=start_time,
                EndTime=end_time,
                Period=period,
                Statistics=[stat]
            )
            # Sort datapoints by timestamp
            datapoints = sorted(response["Datapoints"], key=lambda x: x["Timestamp"])
            return datapoints
        except ClientError as e:
            logger.warning(f"Failed to fetch metric {metric_name}: {e}")
            return []

    def collect_ec2_metrics(
        self, 
        instance_ids: List[str], 
        days: int = 7
    ) -> Dict[str, Dict[str, List[Dict]]]:
        """
        Batch collect metrics for multiple instances (CPU, Network).
        
        :param instance_ids: List of instance IDs
        :param days: Number of past days to fetch
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        metrics_data = {}

        for instance_id in instance_ids:
            metrics_data[instance_id] = {}
            for metric in ["CPUUtilization", "NetworkIn", "NetworkOut"]:
                data = self.get_cloudwatch_metrics(
                    namespace="AWS/EC2",
                    metric_name=metric,
                    dimensions=[{"Name": "InstanceId", "Value": instance_id}],
                    start_time=start_time,
                    end_time=end_time
                )
                metrics_data[instance_id][metric] = data
        
        return metrics_data

    def collect_ebs_metrics(
        self, 
        volume_ids: List[str], 
        days: int = 7
    ) -> Dict[str, Dict[str, List[Dict]]]:
        """
        Batch collect metrics for multiple volumes (Read/Write Ops).
        
        :param volume_ids: List of volume IDs
        :param days: Number of past days to fetch
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        metrics_data = {}

        for volume_id in volume_ids:
            metrics_data[volume_id] = {}
            # EBS metrics: VolumeReadOps, VolumeWriteOps
            for metric in ["VolumeReadOps", "VolumeWriteOps"]:
                data = self.get_cloudwatch_metrics(
                    namespace="AWS/EBS",
                    metric_name=metric,
                    dimensions=[{"Name": "VolumeId", "Value": volume_id}],
                    start_time=start_time,
                    end_time=end_time,
                    stat="Sum" # Ops are counts, so Sum makes sense usually
                )
                metrics_data[volume_id][metric] = data
                
        return metrics_data
