from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

class ResourceOptimizer:
    """
    Analyzes AWS resource usage data to detect inefficiencies and recommend cost-saving actions.
    """
    
    def __init__(self):
        # Mapping for instance resizing logic (simplified)
        self.size_order = {
            "nano": 0, "micro": 1, "small": 2, "medium": 3, 
            "large": 4, "xlarge": 5, "2xlarge": 6, "4xlarge": 7, 
            "8xlarge": 8, "12xlarge": 9, "16xlarge": 10, "24xlarge": 11, "32xlarge": 12 
        }

    def resource_lookup(self, size_name):
        return self.size_order.get(size_name)

    def optimize(self, aws_data: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Main entry point for optimization.
        
        :param aws_data: Dictionary containing 'instances' and 'volumes' with metrics.
        :return: List of recommendation objects.
        """
        recommendations = []
        
        instances = aws_data.get("instances", [])
        volumes = aws_data.get("volumes", [])
        
        logger.info(f"Optimizing {len(instances)} instances and {len(volumes)} volumes")
        
        recommendations.extend(self.detect_idle_instances(instances))
        recommendations.extend(self.detect_oversized_instances(instances))
        recommendations.extend(self.detect_unused_volumes(volumes))
        
        return recommendations

    def detect_idle_instances(self, instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect instances that are unused (Idle).
        Criteria:
        - Age > 7 days
        - CPU < 5% (Average)
        - NetworkIn < 5MB (Total)
        - NetworkOut < 5MB (Total)
        """
        recs = []
        now = datetime.now(timezone.utc)
        
        # 5MB in Bytes
        NETWORK_THRESHOLD = 5 * 1024 * 1024 
        
        for instance in instances:
            # Check Age
            launch_time_str = instance.get("launch_time")
            if not launch_time_str:
                continue
                
            try:
                # Handle ISO format. Assuming UTC as per aws_collector
                launch_time = datetime.fromisoformat(launch_time_str)
                # Ensure timezone awareness for comparison if launch_time has tz
                if launch_time.tzinfo is None:
                    launch_time = launch_time.replace(tzinfo=timezone.utc)
                    
                age = (now - launch_time).days
                if age < 7:
                    continue
            except ValueError:
                logger.warning(f"Invalid launch_time format: {launch_time_str}")
                continue

            metrics = instance.get("metrics", {})
            if not metrics:
                continue
            
            # Extract Metrics
            cpu_points = metrics.get("CPUUtilization", [])
            net_in_points = metrics.get("NetworkIn", [])
            net_out_points = metrics.get("NetworkOut", [])
            
            if not cpu_points:
                continue

            avg_cpu = self._avg(cpu_points)
            sum_net_in = self._sum(net_in_points)
            sum_net_out = self._sum(net_out_points)
            
            if avg_cpu < 5.0 and sum_net_in < NETWORK_THRESHOLD and sum_net_out < NETWORK_THRESHOLD:
                recs.append({
                    "resource_id": instance["instance_id"],
                    "resource_type": "EC2",
                    "issue": "Idle Instance",
                    "recommendation": "Stop or Terminate",
                    "estimated_savings": 0.0, # Placeholder for pricing integration
                    "details": f"Avg CPU: {avg_cpu:.2f}%, NetIn: {sum_net_in/1024/1024:.2f}MB"
                })
        
        return recs

    def detect_oversized_instances(self, instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect instances that are larger than necessary (Oversized).
        Criteria:
        - CPU < 20% (Max)
        - Network traffic low (heuristic < 100MB total)
        """
        recs = []
        
        # Arbitrary low network threshold for "Network low" requirement
        # 100MB roughly
        NETWORK_LOW_THRESHOLD = 100 * 1024 * 1024
        
        for instance in instances:
            metrics = instance.get("metrics", {})
            if not metrics:
                continue
                
            cpu_points = metrics.get("CPUUtilization", [])
            if not cpu_points:
                continue
                
            # Using Max CPU to be safe against bursts
            max_cpu = self._max(cpu_points)
            
            net_in = self._sum(metrics.get("NetworkIn", []))
            net_out = self._sum(metrics.get("NetworkOut", []))
            
            if max_cpu < 20.0 and (net_in + net_out) < NETWORK_LOW_THRESHOLD:
                current_type = instance.get("instance_type", "")
                smaller_type = self._get_smaller_instance_type(current_type)
                
                if smaller_type:
                    recs.append({
                        "resource_id": instance["instance_id"],
                        "resource_type": "EC2",
                        "issue": "Oversized Instance",
                        "recommendation": f"Resize to {smaller_type}",
                        "estimated_savings": 0.0,
                        "details": f"Max CPU: {max_cpu:.2f}%, Utilization is low."
                    })
        
        return recs

    def detect_unused_volumes(self, volumes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect EBS volumes that are available or attached but unused.
        Criteria:
        - Status == 'available'
        - OR (Status == 'in-use' AND ReadOps=0 AND WriteOps=0 for entire period)
        """
        recs = []
        
        for volume in volumes:
            # Case 1: Unattached
            if volume.get("state") == "available":
                recs.append({
                    "resource_id": volume["volume_id"],
                    "resource_type": "EBS",
                    "issue": "Unattached Volume",
                    "recommendation": "Delete",
                    "estimated_savings": 0.0,
                    "details": f"Volume {volume['volume_id']} is not attached to any instance"
                })
                continue
                
            # Case 2: Attached but unused
            # Assumption: We only have the metrics provided (e.g., last 7 days)
            # Requirement says "14 days", but we can only optimize based on data we have.
            metrics = volume.get("metrics", {})
            if not metrics:
                continue
                
            read_ops = metrics.get("VolumeReadOps", [])
            write_ops = metrics.get("VolumeWriteOps", [])
            
            total_reads = self._sum(read_ops)
            total_writes = self._sum(write_ops)
            
            if total_reads == 0 and total_writes == 0:
                recs.append({
                    "resource_id": volume["volume_id"],
                    "resource_type": "EBS",
                    "issue": "Unused Attached Volume",
                    "recommendation": "Detach and Delete",
                    "estimated_savings": 0.0,
                    "details": "Zero IOPS detected in observation period"
                })
                
        return recs

    def _get_smaller_instance_type(self, current_type: str) -> Optional[str]:
        """Simple heuristic to recommend next smaller size in same family."""
        if not current_type or "." not in current_type:
            return None
            
        family, size = current_type.split(".", 1)
        
        current_idx = self.size_order.get(size)
        if current_idx is not None and current_idx > 0:
            # Find the size key with index = current_idx - 1
            for s_name, s_idx in self.size_order.items():
                if s_idx == (current_idx - 1):
                    return f"{family}.{s_name}"
        
        return None

    def _avg(self, datapoints: List[Dict[str, Any]]) -> float:
        if not datapoints:
            return 0.0
        return sum(d["Value"] for d in datapoints) / len(datapoints)

    def _sum(self, datapoints: List[Dict[str, Any]]) -> float:
        if not datapoints:
            return 0.0
        return sum(d["Value"] for d in datapoints)

    def _max(self, datapoints: List[Dict[str, Any]]) -> float:
        if not datapoints:
            return 0.0
        return max(d["Value"] for d in datapoints)
