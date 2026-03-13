"""
Scaling Rules Engine — Evaluates user-defined scaling rules against
CloudWatch metrics and executes pre-authorized scaling actions via boto3.

Supported actions:
  - EBS:  increase/decrease volume size
  - EC2:  resize instance type (stop → modify → start)
  - RDS:  modify DB instance class
"""

import uuid
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


# Ordered instance families for limit comparison
EC2_INSTANCE_SIZE_ORDER = [
    "nano", "micro", "small", "medium", "large",
    "xlarge", "2xlarge", "4xlarge", "8xlarge", "12xlarge", "16xlarge", "24xlarge",
]

RDS_INSTANCE_SIZE_ORDER = EC2_INSTANCE_SIZE_ORDER  # same sizing


def _instance_size_index(instance_type: str) -> int:
    """Get numeric index for an instance size for ordering comparison."""
    # e.g. "m5.xlarge" → "xlarge" or "db.m5.large" → "large"
    size_part = instance_type.rsplit(".", 1)[-1] if "." in instance_type else instance_type
    try:
        return EC2_INSTANCE_SIZE_ORDER.index(size_part)
    except ValueError:
        return -1


class ScalingRulesEngine:
    """
    Core engine that evaluates scaling rules and executes actions.
    Uses a boto3 session from AWSDataService for real AWS API calls.
    """

    def __init__(self, boto3_session=None, region: str = "us-east-1"):
        self.session = boto3_session
        self.region = region
        # In-memory store for rules (MVP — production would use DB)
        self._rules: Dict[str, Dict[str, Any]] = {}
        self._executions: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Rule CRUD
    # ------------------------------------------------------------------

    def create_rule(self, rule_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new scaling rule."""
        rule_id = str(uuid.uuid4())[:8]
        rule = {
            "id": rule_id,
            "name": rule_data.get("name", "Untitled Rule"),
            "description": rule_data.get("description", ""),
            "service_type": rule_data.get("service_type", "ebs"),
            "resource_filter": rule_data.get("resource_filter", {}),
            "metric_namespace": rule_data.get("metric_namespace", "AWS/EBS"),
            "metric_name": rule_data.get("metric_name", "VolumeQueueLength"),
            "metric_dimension_name": rule_data.get("metric_dimension_name", "VolumeId"),
            "metric_statistic": rule_data.get("metric_statistic", "Average"),
            "threshold_operator": rule_data.get("threshold_operator", "gt"),
            "threshold_value": float(rule_data.get("threshold_value", 80)),
            "evaluation_periods": int(rule_data.get("evaluation_periods", 3)),
            "evaluation_interval_seconds": int(rule_data.get("evaluation_interval_seconds", 300)),
            "scaling_direction": rule_data.get("scaling_direction", "scale_up"),
            "scaling_action": rule_data.get("scaling_action", {}),
            "max_scaling_limit": rule_data.get("max_scaling_limit", {}),
            "cooldown_seconds": int(rule_data.get("cooldown_seconds", 1800)),
            "is_enabled": rule_data.get("is_enabled", True),
            "last_triggered_at": None,
            "trigger_count": 0,
            "total_cost_impact": 0.0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._rules[rule_id] = rule
        logger.info(f"Created scaling rule '{rule['name']}' (id={rule_id})")
        return rule

    def get_rules(self) -> List[Dict[str, Any]]:
        """Get all scaling rules."""
        return list(self._rules.values())

    def get_rule(self, rule_id: str) -> Optional[Dict[str, Any]]:
        """Get a single rule by ID."""
        return self._rules.get(rule_id)

    def update_rule(self, rule_id: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update a scaling rule."""
        rule = self._rules.get(rule_id)
        if not rule:
            return None

        updatable = [
            "name", "description", "service_type", "resource_filter",
            "metric_namespace", "metric_name", "metric_dimension_name",
            "metric_statistic", "threshold_operator", "threshold_value",
            "evaluation_periods", "evaluation_interval_seconds",
            "scaling_direction", "scaling_action", "max_scaling_limit",
            "cooldown_seconds", "is_enabled",
        ]
        for field in updatable:
            if field in data:
                rule[field] = data[field]

        rule["updated_at"] = datetime.now(timezone.utc).isoformat()
        return rule

    def delete_rule(self, rule_id: str) -> bool:
        """Delete a scaling rule."""
        if rule_id in self._rules:
            del self._rules[rule_id]
            logger.info(f"Deleted scaling rule {rule_id}")
            return True
        return False

    def toggle_rule(self, rule_id: str) -> Optional[Dict[str, Any]]:
        """Toggle a rule's enabled state."""
        rule = self._rules.get(rule_id)
        if not rule:
            return None
        rule["is_enabled"] = not rule["is_enabled"]
        rule["updated_at"] = datetime.now(timezone.utc).isoformat()
        return rule

    # ------------------------------------------------------------------
    # Rule Evaluation
    # ------------------------------------------------------------------

    def evaluate_all_rules(self) -> List[Dict[str, Any]]:
        """
        Evaluate all enabled rules against current CloudWatch metrics.
        Returns a list of evaluation results.
        """
        results = []
        for rule_id, rule in self._rules.items():
            if not rule["is_enabled"]:
                continue
            try:
                result = self._evaluate_rule(rule)
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating rule {rule_id}: {e}")
                results.append({
                    "rule_id": rule_id,
                    "rule_name": rule["name"],
                    "triggered": False,
                    "error": str(e),
                })
        return results

    def _evaluate_rule(self, rule: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single rule against CloudWatch metrics."""
        rule_id = rule["id"]

        # Check cooldown
        if rule["last_triggered_at"]:
            last = datetime.fromisoformat(rule["last_triggered_at"])
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            elapsed = (datetime.now(timezone.utc) - last).total_seconds()
            if elapsed < rule["cooldown_seconds"]:
                return {
                    "rule_id": rule_id,
                    "rule_name": rule["name"],
                    "triggered": False,
                    "reason": f"In cooldown ({int(rule['cooldown_seconds'] - elapsed)}s remaining)",
                }

        # Discover target resources
        resource_ids = self._discover_resources(rule)
        if not resource_ids:
            return {
                "rule_id": rule_id,
                "rule_name": rule["name"],
                "triggered": False,
                "reason": "No matching resources found",
            }

        # Check each resource
        triggered_resources = []
        for resource_id in resource_ids:
            metric_value = self._get_metric_value(rule, resource_id)
            if metric_value is None:
                continue

            breached = self._check_threshold(metric_value, rule["threshold_operator"], rule["threshold_value"])
            if breached:
                triggered_resources.append({
                    "resource_id": resource_id,
                    "metric_value": metric_value,
                })

        if not triggered_resources:
            return {
                "rule_id": rule_id,
                "rule_name": rule["name"],
                "triggered": False,
                "reason": "Threshold not breached on any resource",
                "resources_checked": len(resource_ids),
            }

        # Execute scaling on all breached resources
        execution_results = []
        for res in triggered_resources:
            exec_result = self._execute_scaling(rule, res["resource_id"], res["metric_value"])
            execution_results.append(exec_result)

        # Update rule stats
        rule["last_triggered_at"] = datetime.now(timezone.utc).isoformat()
        rule["trigger_count"] += len(triggered_resources)

        return {
            "rule_id": rule_id,
            "rule_name": rule["name"],
            "triggered": True,
            "triggered_count": len(triggered_resources),
            "executions": execution_results,
        }

    def _discover_resources(self, rule: Dict[str, Any]) -> List[str]:
        """Discover resources matching the rule's filter."""
        resource_filter = rule.get("resource_filter", {})

        # If specific resource IDs are given, use them
        if "resource_ids" in resource_filter and resource_filter["resource_ids"]:
            return resource_filter["resource_ids"]

        # Otherwise, discover resources from AWS
        if not self.session:
            return []

        service = rule["service_type"]
        try:
            if service == "ebs":
                ec2 = self.session.client("ec2", region_name=self.region)
                filters = [{"Name": "status", "Values": ["in-use"]}]
                if "tags" in resource_filter:
                    for key, value in resource_filter["tags"].items():
                        filters.append({"Name": f"tag:{key}", "Values": [value]})
                response = ec2.describe_volumes(Filters=filters)
                return [v["VolumeId"] for v in response.get("Volumes", [])]

            elif service == "ec2":
                ec2 = self.session.client("ec2", region_name=self.region)
                filters = [{"Name": "instance-state-name", "Values": ["running"]}]
                if "tags" in resource_filter:
                    for key, value in resource_filter["tags"].items():
                        filters.append({"Name": f"tag:{key}", "Values": [value]})
                reservations = ec2.describe_instances(Filters=filters).get("Reservations", [])
                ids = []
                for res in reservations:
                    for inst in res.get("Instances", []):
                        ids.append(inst["InstanceId"])
                return ids

            elif service == "rds":
                rds = self.session.client("rds", region_name=self.region)
                dbs = rds.describe_db_instances().get("DBInstances", [])
                ids = [db["DBInstanceIdentifier"] for db in dbs if db.get("DBInstanceStatus") == "available"]
                return ids

        except Exception as e:
            logger.error(f"Error discovering resources for rule {rule['id']}: {e}")
        return []

    def _get_metric_value(self, rule: Dict[str, Any], resource_id: str) -> Optional[float]:
        """Get the current metric value from CloudWatch."""
        if not self.session:
            return None

        try:
            cw = self.session.client("cloudwatch", region_name=self.region)
            end = datetime.now(timezone.utc)
            start = end - timedelta(seconds=rule["evaluation_interval_seconds"] * rule["evaluation_periods"])

            response = cw.get_metric_statistics(
                Namespace=rule["metric_namespace"],
                MetricName=rule["metric_name"],
                Dimensions=[{"Name": rule["metric_dimension_name"], "Value": resource_id}],
                StartTime=start,
                EndTime=end,
                Period=rule["evaluation_interval_seconds"],
                Statistics=[rule["metric_statistic"]],
            )

            datapoints = response.get("Datapoints", [])
            if not datapoints:
                return None

            # Check if threshold was breached for all evaluation periods
            stat = rule["metric_statistic"]
            values = sorted(datapoints, key=lambda d: d["Timestamp"])
            recent_values = [d[stat] for d in values[-rule["evaluation_periods"]:]]

            if len(recent_values) < rule["evaluation_periods"]:
                # Not enough datapoints yet
                return None

            # All periods must breach for trigger
            all_breached = all(
                self._check_threshold(v, rule["threshold_operator"], rule["threshold_value"])
                for v in recent_values
            )

            if all_breached:
                return recent_values[-1]  # Return the most recent value
            return None

        except Exception as e:
            logger.error(f"Error getting metric for {resource_id}: {e}")
            return None

    def _check_threshold(self, value: float, operator: str, threshold: float) -> bool:
        """Check if a value breaches a threshold."""
        if operator == "gt":
            return value > threshold
        elif operator == "lt":
            return value < threshold
        elif operator == "gte":
            return value >= threshold
        elif operator == "lte":
            return value <= threshold
        return False

    # ------------------------------------------------------------------
    # Scaling Execution
    # ------------------------------------------------------------------

    def _execute_scaling(self, rule: Dict[str, Any], resource_id: str, metric_value: float) -> Dict[str, Any]:
        """Execute the scaling action defined in the rule."""
        start_time = time.time()
        execution = {
            "id": str(uuid.uuid4())[:8],
            "rule_id": rule["id"],
            "rule_name": rule["name"],
            "resource_id": resource_id,
            "triggered_at": datetime.now(timezone.utc).isoformat(),
            "metric_value_at_trigger": metric_value,
            "threshold_value": rule["threshold_value"],
            "action_taken": {},
            "previous_state": {},
            "new_state": {},
            "status": "failed",
            "error_message": None,
            "cost_impact": None,
            "execution_duration_ms": None,
        }

        try:
            service = rule["service_type"]
            action = rule["scaling_action"]

            if service == "ebs":
                execution = self._scale_ebs(rule, resource_id, action, execution)
            elif service == "ec2":
                execution = self._scale_ec2(rule, resource_id, action, execution)
            elif service == "rds":
                execution = self._scale_rds(rule, resource_id, action, execution)
            else:
                execution["error_message"] = f"Unsupported service type: {service}"

        except Exception as e:
            execution["status"] = "failed"
            execution["error_message"] = str(e)
            logger.error(f"Scaling execution failed for {resource_id}: {e}")

        execution["execution_duration_ms"] = int((time.time() - start_time) * 1000)
        self._executions.append(execution)
        return execution

    def _scale_ebs(self, rule: Dict, resource_id: str, action: Dict, execution: Dict) -> Dict:
        """Scale EBS volume — increase/decrease size."""
        if not self.session:
            execution["error_message"] = "No AWS session available"
            return execution

        ec2 = self.session.client("ec2", region_name=self.region)

        # Get current volume info
        vol_resp = ec2.describe_volumes(VolumeIds=[resource_id])
        volumes = vol_resp.get("Volumes", [])
        if not volumes:
            execution["error_message"] = f"Volume {resource_id} not found"
            return execution

        vol = volumes[0]
        current_size = vol["Size"]
        current_type = vol.get("VolumeType", "gp3")

        execution["previous_state"] = {"size_gb": current_size, "volume_type": current_type}

        # Calculate new size
        amount_gb = int(action.get("amount_gb", 4))
        if rule["scaling_direction"] == "scale_up":
            new_size = current_size + amount_gb
        else:
            new_size = max(1, current_size - amount_gb)

        # Check max limit
        max_size = rule.get("max_scaling_limit", {}).get("max_size_gb")
        if max_size and new_size > max_size:
            execution["status"] = "limit_reached"
            execution["error_message"] = f"New size {new_size}GB exceeds limit {max_size}GB"
            return execution

        # EBS volumes can only be increased, not decreased
        if new_size <= current_size and rule["scaling_direction"] == "scale_up":
            execution["status"] = "limit_reached"
            execution["error_message"] = "Volume already at or above target size"
            return execution

        # Execute modification
        modify_params = {"VolumeId": resource_id, "Size": new_size}

        # Optionally change volume type
        target_type = action.get("target_volume_type")
        if target_type:
            modify_params["VolumeType"] = target_type

        ec2.modify_volume(**modify_params)

        execution["action_taken"] = {
            "action": "modify_volume",
            "api_call": "ec2.modify_volume",
            "params": modify_params,
        }
        execution["new_state"] = {"size_gb": new_size, "volume_type": target_type or current_type}
        execution["status"] = "success"

        # Estimate cost impact (EBS gp3 ~$0.08/GB/month)
        cost_per_gb = {"gp3": 0.08, "gp2": 0.10, "io1": 0.125, "io2": 0.125, "st1": 0.045, "sc1": 0.015}
        price = cost_per_gb.get(current_type, 0.08)
        execution["cost_impact"] = round((new_size - current_size) * price, 2)

        logger.info(f"EBS volume {resource_id} scaled from {current_size}GB to {new_size}GB")
        return execution

    def _scale_ec2(self, rule: Dict, resource_id: str, action: Dict, execution: Dict) -> Dict:
        """Scale EC2 instance — resize instance type (stop → modify → start)."""
        if not self.session:
            execution["error_message"] = "No AWS session available"
            return execution

        ec2 = self.session.client("ec2", region_name=self.region)

        # Get current instance info
        desc = ec2.describe_instances(InstanceIds=[resource_id])
        reservations = desc.get("Reservations", [])
        if not reservations:
            execution["error_message"] = f"Instance {resource_id} not found"
            return execution

        instance = reservations[0]["Instances"][0]
        current_type = instance.get("InstanceType", "unknown")
        current_state = instance.get("State", {}).get("Name", "unknown")

        execution["previous_state"] = {"instance_type": current_type, "state": current_state}

        target_type = action.get("target_instance_type")
        if not target_type:
            execution["error_message"] = "No target_instance_type specified in scaling action"
            return execution

        # Check max limit
        max_type = rule.get("max_scaling_limit", {}).get("max_instance_type")
        if max_type:
            if _instance_size_index(target_type) > _instance_size_index(max_type):
                execution["status"] = "limit_reached"
                execution["error_message"] = f"Target type {target_type} exceeds limit {max_type}"
                return execution

        # Instance must be stopped to modify type
        was_running = current_state == "running"
        if was_running:
            ec2.stop_instances(InstanceIds=[resource_id])
            # Wait for stop (simplified — in production, use waiter)
            waiter = ec2.get_waiter("instance_stopped")
            waiter.wait(InstanceIds=[resource_id], WaiterConfig={"Delay": 10, "MaxAttempts": 30})

        # Modify instance type
        ec2.modify_instance_attribute(
            InstanceId=resource_id,
            InstanceType={"Value": target_type}
        )

        # Restart if it was running
        if was_running:
            ec2.start_instances(InstanceIds=[resource_id])

        execution["action_taken"] = {
            "action": "resize_instance",
            "api_call": "ec2.modify_instance_attribute",
            "previous_type": current_type,
            "new_type": target_type,
            "was_running": was_running,
        }
        execution["new_state"] = {"instance_type": target_type, "state": "running" if was_running else current_state}
        execution["status"] = "success"

        logger.info(f"EC2 instance {resource_id} resized from {current_type} to {target_type}")
        return execution

    def _scale_rds(self, rule: Dict, resource_id: str, action: Dict, execution: Dict) -> Dict:
        """Scale RDS instance — modify DB instance class."""
        if not self.session:
            execution["error_message"] = "No AWS session available"
            return execution

        rds = self.session.client("rds", region_name=self.region)

        # Get current DB info
        desc = rds.describe_db_instances(DBInstanceIdentifier=resource_id)
        dbs = desc.get("DBInstances", [])
        if not dbs:
            execution["error_message"] = f"RDS instance {resource_id} not found"
            return execution

        db = dbs[0]
        current_class = db.get("DBInstanceClass", "unknown")
        current_storage = db.get("AllocatedStorage", 0)

        execution["previous_state"] = {"db_instance_class": current_class, "allocated_storage": current_storage}

        target_class = action.get("target_db_instance_class")
        target_storage = action.get("target_allocated_storage")

        modify_params: Dict[str, Any] = {
            "DBInstanceIdentifier": resource_id,
            "ApplyImmediately": True,
        }

        if target_class:
            # Check max limit
            max_class = rule.get("max_scaling_limit", {}).get("max_instance_class")
            if max_class:
                if _instance_size_index(target_class) > _instance_size_index(max_class):
                    execution["status"] = "limit_reached"
                    execution["error_message"] = f"Target class {target_class} exceeds limit {max_class}"
                    return execution
            modify_params["DBInstanceClass"] = target_class

        if target_storage:
            max_storage = rule.get("max_scaling_limit", {}).get("max_storage_gb")
            if max_storage and target_storage > max_storage:
                execution["status"] = "limit_reached"
                execution["error_message"] = f"Target storage {target_storage}GB exceeds limit {max_storage}GB"
                return execution
            modify_params["AllocatedStorage"] = int(target_storage)
        elif action.get("increase_storage_gb"):
            new_storage = current_storage + int(action["increase_storage_gb"])
            max_storage = rule.get("max_scaling_limit", {}).get("max_storage_gb")
            if max_storage and new_storage > max_storage:
                execution["status"] = "limit_reached"
                execution["error_message"] = f"New storage {new_storage}GB exceeds limit {max_storage}GB"
                return execution
            modify_params["AllocatedStorage"] = new_storage

        rds.modify_db_instance(**modify_params)

        execution["action_taken"] = {
            "action": "modify_db_instance",
            "api_call": "rds.modify_db_instance",
            "params": {k: v for k, v in modify_params.items() if k != "DBInstanceIdentifier"},
        }
        execution["new_state"] = {
            "db_instance_class": target_class or current_class,
            "allocated_storage": modify_params.get("AllocatedStorage", current_storage),
        }
        execution["status"] = "success"

        logger.info(f"RDS instance {resource_id} modified: {modify_params}")
        return execution

    # ------------------------------------------------------------------
    # Dry-Run / Test
    # ------------------------------------------------------------------

    def test_rule(self, rule_id: str) -> Dict[str, Any]:
        """Dry-run test a rule — checks metrics but doesn't execute."""
        rule = self._rules.get(rule_id)
        if not rule:
            return {"error": "Rule not found"}

        resource_ids = self._discover_resources(rule)
        results = []
        for resource_id in resource_ids:
            metric_value = self._get_metric_value(rule, resource_id)
            breached = False
            if metric_value is not None:
                breached = self._check_threshold(metric_value, rule["threshold_operator"], rule["threshold_value"])
            results.append({
                "resource_id": resource_id,
                "metric_value": metric_value,
                "threshold": rule["threshold_value"],
                "operator": rule["threshold_operator"],
                "breached": breached,
                "would_trigger": breached,
            })

        return {
            "rule_id": rule_id,
            "rule_name": rule["name"],
            "dry_run": True,
            "resources_checked": len(resource_ids),
            "resources_breached": len([r for r in results if r["breached"]]),
            "details": results,
        }

    # ------------------------------------------------------------------
    # Execution History
    # ------------------------------------------------------------------

    def get_executions(self, rule_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get execution history, optionally filtered by rule ID."""
        if rule_id:
            return [e for e in reversed(self._executions) if e["rule_id"] == rule_id]
        return list(reversed(self._executions))

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics for scaling rules."""
        all_rules = list(self._rules.values())
        enabled_rules = [r for r in all_rules if r["is_enabled"]]
        successful_execs = [e for e in self._executions if e["status"] == "success"]
        failed_execs = [e for e in self._executions if e["status"] == "failed"]

        total_cost_impact = sum(
            e.get("cost_impact") or 0 for e in self._executions
            if e.get("cost_impact") is not None
        )

        return {
            "total_rules": len(all_rules),
            "active_rules": len(enabled_rules),
            "total_executions": len(self._executions),
            "successful_executions": len(successful_execs),
            "failed_executions": len(failed_execs),
            "total_cost_impact": round(total_cost_impact, 2),
            "success_rate": round(
                (len(successful_execs) / max(len(self._executions), 1)) * 100, 1
            ),
        }
