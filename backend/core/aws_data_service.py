"""
AWS Data Service - Centralized service for fetching real data from connected AWS accounts.

This module provides a single service class that wraps boto3 calls and provides
all the data that API endpoints need. Results are cached in-memory with a TTL
to avoid excessive AWS API calls.
"""

import boto3
import time
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Tuple
from decimal import Decimal

logger = logging.getLogger(__name__)


class CacheEntry:
    """Simple TTL cache entry."""
    def __init__(self, data: Any, ttl_seconds: int = 300):
        self.data = data
        self.expires_at = time.time() + ttl_seconds

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at


class AWSDataService:
    """
    Centralized AWS data service that fetches real data from a connected AWS account.
    
    All methods return real data via boto3. Results are cached in-memory for 5 minutes
    to avoid excessive API calls to AWS.
    """

    def __init__(self, access_key_id: str, secret_access_key: str, region: str = "us-east-1",
                 session_token: Optional[str] = None):
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.region = region
        self.session_token = session_token
        self._cache: Dict[str, CacheEntry] = {}
        self._account_id: Optional[str] = None

        # Create boto3 session
        session_kwargs = {
            "aws_access_key_id": access_key_id,
            "aws_secret_access_key": secret_access_key,
            "region_name": region,
        }
        if session_token:
            session_kwargs["aws_session_token"] = session_token

        self.session = boto3.Session(**session_kwargs)
        logger.info(f"AWSDataService initialized for region {region}")

    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached data if not expired."""
        entry = self._cache.get(key)
        if entry and not entry.is_expired:
            return entry.data
        return None

    def _set_cached(self, key: str, data: Any, ttl: int = 300):
        """Cache data with TTL (default 5 minutes)."""
        self._cache[key] = CacheEntry(data, ttl)

    # ---------------------------------------------------------------
    # Connection & Account Info
    # ---------------------------------------------------------------

    def test_connection(self) -> Dict[str, Any]:
        """Test AWS credentials and return account info."""
        try:
            sts = self.session.client("sts")
            identity = sts.get_caller_identity()
            self._account_id = identity["Account"]
            return {
                "success": True,
                "account_id": identity["Account"],
                "arn": identity["Arn"],
                "user_id": identity["UserId"],
            }
        except Exception as e:
            logger.error(f"AWS connection test failed: {e}")
            return {"success": False, "error": str(e)}

    def get_account_id(self) -> str:
        """Get the AWS account ID."""
        if not self._account_id:
            result = self.test_connection()
            self._account_id = result.get("account_id", "unknown")
        return self._account_id

    def get_accounts(self) -> List[Dict[str, Any]]:
        """
        Get AWS accounts. Tries Organizations first, falls back to single account from STS.
        """
        cached = self._get_cached("accounts")
        if cached is not None:
            return cached

        accounts = []
        try:
            orgs = self.session.client("organizations")
            paginator = orgs.get_paginator("list_accounts")
            for page in paginator.paginate():
                for account in page["Accounts"]:
                    accounts.append({
                        "id": account["Id"],
                        "name": account.get("Name", account["Id"]),
                        "provider": "aws",
                        "region": self.region,
                        "status": account.get("Status", "ACTIVE").lower(),
                    })
        except Exception as e:
            logger.info(f"Organizations API not available ({e}), using single account")
            account_id = self.get_account_id()
            accounts = [{
                "id": account_id,
                "name": f"AWS Account {account_id}",
                "provider": "aws",
                "region": self.region,
                "status": "active",
            }]

        # Enrich with cost data
        cost_by_account = self._get_cost_by_account(30)
        for account in accounts:
            acct_cost = cost_by_account.get(account["id"], {})
            account["monthly_cost"] = acct_cost.get("cost", 0)
            account["potential_savings"] = round(acct_cost.get("cost", 0) * 0.08, 2)  # estimate 8% potential savings

        self._set_cached("accounts", accounts)
        return accounts

    # ---------------------------------------------------------------
    # Cost Data
    # ---------------------------------------------------------------

    def _get_cost_by_account(self, days: int) -> Dict[str, Dict[str, float]]:
        """Get cost breakdown by linked account."""
        try:
            ce = self.session.client("ce")
            end_date = date.today()
            start_date = end_date - timedelta(days=days)

            response = ce.get_cost_and_usage(
                TimePeriod={
                    "Start": start_date.isoformat(),
                    "End": end_date.isoformat(),
                },
                Granularity="MONTHLY",
                Metrics=["UnblendedCost"],
                GroupBy=[{"Type": "DIMENSION", "Key": "LINKED_ACCOUNT"}],
            )

            result = {}
            for time_period in response.get("ResultsByTime", []):
                for group in time_period.get("Groups", []):
                    account_id = group["Keys"][0]
                    cost = float(group["Metrics"]["UnblendedCost"]["Amount"])
                    if account_id in result:
                        result[account_id]["cost"] += cost
                    else:
                        result[account_id] = {"cost": cost}
            return result
        except Exception as e:
            logger.error(f"Error getting cost by account: {e}")
            return {}

    def get_cost_trend(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get daily cost trend data from Cost Explorer."""
        cache_key = f"cost_trend_{days}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            ce = self.session.client("ce")
            end_date = date.today()
            start_date = end_date - timedelta(days=days)

            response = ce.get_cost_and_usage(
                TimePeriod={
                    "Start": start_date.isoformat(),
                    "End": end_date.isoformat(),
                },
                Granularity="DAILY",
                Metrics=["UnblendedCost"],
            )

            trend = []
            for result in response.get("ResultsByTime", []):
                cost = float(result["Metrics"]["UnblendedCost"]["Amount"])
                trend.append({
                    "date": result["TimePeriod"]["Start"],
                    "cost": round(cost, 2),
                    "savings": round(cost * 0.05, 2),  # estimated savings based on cost
                })

            self._set_cached(cache_key, trend)
            return trend
        except Exception as e:
            logger.error(f"Error getting cost trend: {e}")
            return []

    def get_service_breakdown(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get cost breakdown by AWS service."""
        cache_key = f"service_breakdown_{days}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            ce = self.session.client("ce")
            end_date = date.today()
            start_date = end_date - timedelta(days=days)

            response = ce.get_cost_and_usage(
                TimePeriod={
                    "Start": start_date.isoformat(),
                    "End": end_date.isoformat(),
                },
                Granularity="MONTHLY",
                Metrics=["UnblendedCost"],
                GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
            )

            services = {}
            for time_period in response.get("ResultsByTime", []):
                for group in time_period.get("Groups", []):
                    service_name = group["Keys"][0]
                    cost = float(group["Metrics"]["UnblendedCost"]["Amount"])
                    if service_name in services:
                        services[service_name] += cost
                    else:
                        services[service_name] = cost

            total = sum(services.values()) or 1
            colors = ["#ff9800", "#4caf50", "#2196f3", "#9c27b0", "#607d8b", "#795548",
                       "#e91e63", "#00bcd4", "#8bc34a", "#ff5722"]

            breakdown = []
            for i, (name, cost) in enumerate(sorted(services.items(), key=lambda x: -x[1])):
                if cost < 0.01:
                    continue
                breakdown.append({
                    "name": name,
                    "cost": round(cost, 2),
                    "value": round((cost / total) * 100, 1),
                    "color": colors[i % len(colors)],
                })

            self._set_cached(cache_key, breakdown)
            return breakdown
        except Exception as e:
            logger.error(f"Error getting service breakdown: {e}")
            return []

    def get_total_monthly_cost(self) -> float:
        """Get total cost for the current month."""
        trend = self.get_cost_trend(30)
        return round(sum(d["cost"] for d in trend), 2)

    # ---------------------------------------------------------------
    # Budgets
    # ---------------------------------------------------------------

    def get_budgets(self) -> List[Dict[str, Any]]:
        """Get AWS Budgets data."""
        cached = self._get_cached("budgets")
        if cached is not None:
            return cached

        try:
            budgets_client = self.session.client("budgets")
            account_id = self.get_account_id()

            response = budgets_client.describe_budgets(AccountId=account_id)
            budgets = []

            for budget in response.get("Budgets", []):
                limit = float(budget["BudgetLimit"]["Amount"])
                spent_data = budget.get("CalculatedSpend", {})
                spent = float(spent_data.get("ActualSpend", {}).get("Amount", 0))
                utilization = round((spent / limit) * 100, 1) if limit > 0 else 0

                status = "good"
                if utilization > 95:
                    status = "critical"
                elif utilization > 80:
                    status = "warning"

                budgets.append({
                    "id": budget["BudgetName"],
                    "name": budget["BudgetName"],
                    "amount": limit,
                    "spent": spent,
                    "remaining": round(limit - spent, 2),
                    "utilization": utilization,
                    "period": budget.get("TimeUnit", "MONTHLY").lower(),
                    "status": status,
                    "team": budget.get("CostFilters", {}).get("TagKeyValue", ["General"])[0] if budget.get("CostFilters") else "General",
                    "alerts": [],
                })

                # Get notifications for this budget
                try:
                    notifs = budgets_client.describe_notifications_for_budget(
                        AccountId=account_id,
                        BudgetName=budget["BudgetName"],
                    )
                    for notif in notifs.get("Notifications", []):
                        threshold = notif.get("Threshold", 0)
                        budgets[-1]["alerts"].append({
                            "threshold": threshold,
                            "triggered": utilization >= threshold,
                        })
                except Exception:
                    pass

            self._set_cached("budgets", budgets)
            return budgets
        except Exception as e:
            logger.error(f"Error getting budgets: {e}")
            return []

    # ---------------------------------------------------------------
    # Optimization Recommendations
    # ---------------------------------------------------------------

    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Get rightsizing and cost optimization recommendations."""
        cached = self._get_cached("optimization")
        if cached is not None:
            return cached

        recommendations = []

        # Rightsizing recommendations from Cost Explorer
        try:
            ce = self.session.client("ce")
            response = ce.get_rightsizing_recommendation(
                Service="AmazonEC2",
                Configuration={
                    "RecommendationTarget": "SAME_INSTANCE_FAMILY",
                    "BenefitsConsidered": True,
                }
            )

            for rec in response.get("RightsizingRecommendations", []):
                current = rec.get("CurrentInstance", {})
                modify_rec = rec.get("ModifyRecommendation", {})
                terminate_rec = rec.get("TerminateRecommendation", {})

                if rec.get("RightsizingType") == "TERMINATE":
                    monthly_savings = float(terminate_rec.get("EstimatedMonthlySavings", 0))
                    recommendations.append({
                        "action_id": current.get("ResourceId", "unknown"),
                        "action_type": "terminate_instance",
                        "name": f"Terminate Idle Instance {current.get('ResourceId', '')}",
                        "description": f"Instance {current.get('InstanceName', current.get('ResourceId', ''))} appears to be idle or underutilized",
                        "potential_savings": round(monthly_savings, 2),
                        "status": "enabled",
                        "resource_type": "ec2_instance",
                        "risk_level": "medium",
                        "estimated_execution_time": "5 minutes",
                        "last_executed": None,
                        "success_rate": 0,
                        "current_config": f"{current.get('ResourceDetails', {}).get('EC2ResourceDetails', {}).get('InstanceType', 'unknown')}",
                        "recommended_config": "Terminate",
                        "confidence": 90,
                    })
                elif modify_rec:
                    target = modify_rec.get("TargetInstances", [{}])[0]
                    monthly_savings = float(target.get("EstimatedMonthlySavings", 0))
                    current_type = current.get("ResourceDetails", {}).get("EC2ResourceDetails", {}).get("InstanceType", "unknown")
                    target_type = target.get("ResourceDetails", {}).get("EC2ResourceDetails", {}).get("InstanceType", "unknown")

                    recommendations.append({
                        "action_id": current.get("ResourceId", "unknown"),
                        "action_type": "resize_underutilized_instances",
                        "name": f"Rightsize {current.get('ResourceId', '')} from {current_type} to {target_type}",
                        "description": f"Instance is underutilized. Resize from {current_type} to {target_type}",
                        "potential_savings": round(monthly_savings, 2),
                        "status": "enabled",
                        "resource_type": "ec2_instance",
                        "risk_level": "low",
                        "estimated_execution_time": "10 minutes",
                        "last_executed": None,
                        "success_rate": 0,
                        "current_config": current_type,
                        "recommended_config": target_type,
                        "confidence": 85,
                    })
        except Exception as e:
            logger.error(f"Error getting rightsizing recommendations: {e}")

        # Check for unattached EBS volumes
        try:
            ec2 = self.session.client("ec2")
            volumes = ec2.describe_volumes(
                Filters=[{"Name": "status", "Values": ["available"]}]
            )
            for vol in volumes.get("Volumes", []):
                size = vol.get("Size", 0)
                vol_type = vol.get("VolumeType", "gp3")
                # Estimate cost per GB per month
                cost_per_gb = {"gp3": 0.08, "gp2": 0.10, "io1": 0.125, "io2": 0.125, "st1": 0.045, "sc1": 0.015, "standard": 0.05}
                monthly_cost = size * cost_per_gb.get(vol_type, 0.08)
                
                recommendations.append({
                    "action_id": vol["VolumeId"],
                    "action_type": "delete_volumes",
                    "name": f"Delete Unattached Volume {vol['VolumeId']}",
                    "description": f"Unattached {vol_type} volume ({size} GB) incurring unnecessary costs",
                    "potential_savings": round(monthly_cost, 2),
                    "status": "enabled",
                    "resource_type": "ebs_volume",
                    "risk_level": "medium",
                    "estimated_execution_time": "2 minutes",
                    "last_executed": None,
                    "success_rate": 0,
                    "current_config": f"{vol_type} {size}GB (Unattached)",
                    "recommended_config": "Delete unused volume",
                    "confidence": 100,
                })
        except Exception as e:
            logger.error(f"Error checking unattached volumes: {e}")

        # Check for unused Elastic IPs
        try:
            ec2 = self.session.client("ec2")
            addresses = ec2.describe_addresses()
            for eip in addresses.get("Addresses", []):
                if not eip.get("AssociationId"):
                    recommendations.append({
                        "action_id": eip.get("AllocationId", eip.get("PublicIp", "unknown")),
                        "action_type": "release_elastic_ips",
                        "name": f"Release Unused Elastic IP {eip.get('PublicIp', '')}",
                        "description": "Unassociated Elastic IP address incurring charges",
                        "potential_savings": 3.60,  # ~$0.005/hour = ~$3.60/month
                        "status": "enabled",
                        "resource_type": "elastic_ip",
                        "risk_level": "low",
                        "estimated_execution_time": "1 minute",
                        "last_executed": None,
                        "success_rate": 0,
                        "current_config": f"Elastic IP {eip.get('PublicIp', '')} (Unassociated)",
                        "recommended_config": "Release IP",
                        "confidence": 100,
                    })
        except Exception as e:
            logger.error(f"Error checking elastic IPs: {e}")

        self._set_cached("optimization", recommendations)
        return recommendations

    def get_automation_stats(self) -> Dict[str, Any]:
        """Calculate automation stats from real recommendations."""
        recommendations = self.get_optimization_recommendations()
        total_savings = sum(r["potential_savings"] for r in recommendations)

        return {
            "total_actions": len(recommendations),
            "active_actions": len([r for r in recommendations if r["status"] == "enabled"]),
            "monthly_savings": round(total_savings, 2),
            "actions_this_month": len(recommendations),
            "success_rate": 0,
            "automation_enabled": True,
            "last_execution": None,
            "pending_approvals": len(recommendations),
            "failed_actions": 0,
        }

    # ---------------------------------------------------------------
    # Cost Anomalies / Alerts
    # ---------------------------------------------------------------

    def get_cost_anomalies(self) -> List[Dict[str, Any]]:
        """Get cost anomalies from AWS Cost Anomaly Detection."""
        cached = self._get_cached("anomalies")
        if cached is not None:
            return cached

        alerts = []
        try:
            ce = self.session.client("ce")
            end_date = date.today().isoformat()
            start_date = (date.today() - timedelta(days=30)).isoformat()

            response = ce.get_anomalies(
                DateInterval={"StartDate": start_date, "EndDate": end_date},
                MaxResults=20,
            )

            for i, anomaly in enumerate(response.get("Anomalies", [])):
                impact = anomaly.get("Impact", {})
                total_impact = float(impact.get("TotalImpact", 0))
                severity = "high" if total_impact > 100 else "warning" if total_impact > 20 else "info"

                alerts.append({
                    "id": anomaly.get("AnomalyId", f"anomaly_{i}"),
                    "type": "cost_anomaly",
                    "message": f"Cost anomaly detected: ${total_impact:.2f} unexpected cost in {anomaly.get('DimensionValue', 'unknown service')}",
                    "severity": severity,
                    "timestamp": anomaly.get("AnomalyStartDate", datetime.utcnow().isoformat()),
                })
        except Exception as e:
            logger.info(f"Cost Anomaly Detection not available: {e}")

        # Also check budget alerts
        budgets = self.get_budgets()
        for budget in budgets:
            if budget["utilization"] > 80:
                severity = "high" if budget["utilization"] > 95 else "warning"
                alerts.append({
                    "id": f"budget_{budget['id']}",
                    "type": "budget_exceeded",
                    "message": f"Budget '{budget['name']}' is {budget['utilization']}% utilized",
                    "severity": severity,
                    "timestamp": datetime.utcnow().isoformat(),
                })

        self._set_cached("anomalies", alerts)
        return alerts

    # ---------------------------------------------------------------
    # EC2 Instances (for workloads / multi-cloud)
    # ---------------------------------------------------------------

    def get_ec2_instances(self) -> List[Dict[str, Any]]:
        """Get EC2 instances as workloads."""
        cached = self._get_cached("ec2_instances")
        if cached is not None:
            return cached

        instances = []
        try:
            ec2 = self.session.client("ec2")
            paginator = ec2.get_paginator("describe_instances")

            for page in paginator.paginate():
                for reservation in page.get("Reservations", []):
                    for instance in reservation.get("Instances", []):
                        name_tag = ""
                        for tag in instance.get("Tags", []):
                            if tag["Key"] == "Name":
                                name_tag = tag["Value"]
                                break

                        instances.append({
                            "id": instance["InstanceId"],
                            "name": name_tag or instance["InstanceId"],
                            "description": f"{instance.get('InstanceType', 'unknown')} instance in {instance.get('Placement', {}).get('AvailabilityZone', self.region)}",
                            "created_at": instance.get("LaunchTime", datetime.utcnow()).isoformat() if isinstance(instance.get("LaunchTime"), datetime) else str(instance.get("LaunchTime", "")),
                            "updated_at": datetime.utcnow().isoformat(),
                            "regions": [instance.get("Placement", {}).get("AvailabilityZone", self.region)],
                            "compliance_requirements": [],
                            "instance_type": instance.get("InstanceType", "unknown"),
                            "state": instance.get("State", {}).get("Name", "unknown"),
                        })
        except Exception as e:
            logger.error(f"Error getting EC2 instances: {e}")

        self._set_cached("ec2_instances", instances)
        return instances

    # ---------------------------------------------------------------
    # Compliance & Governance
    # ---------------------------------------------------------------

    def get_compliance_data(self) -> Dict[str, Any]:
        """Scan AWS resources for tagging compliance and security violations."""
        cached = self._get_cached("compliance")
        if cached is not None:
            return cached

        required_tags = ["Name", "Environment", "Owner", "CostCenter", "Project"]
        violations = []
        service_stats = {}
        total_resources = 0
        total_compliant = 0

        # --- EC2 Instances ---
        ec2_total = 0
        ec2_compliant = 0
        try:
            ec2 = self.session.client("ec2")
            paginator = ec2.get_paginator("describe_instances")
            for page in paginator.paginate():
                for res in page.get("Reservations", []):
                    for inst in res.get("Instances", []):
                        ec2_total += 1
                        tags = {t["Key"]: t["Value"] for t in inst.get("Tags", [])}
                        missing = [t for t in required_tags if t not in tags]
                        if not missing:
                            ec2_compliant += 1
                        else:
                            violations.append({
                                "id": inst["InstanceId"],
                                "resource": inst["InstanceId"],
                                "resourceType": "EC2 Instance",
                                "policy": "Required Tags Policy",
                                "violation": f"Missing tags: {', '.join(missing)}",
                                "severity": "medium",
                                "team": tags.get("Owner", "Unknown"),
                                "detected": datetime.utcnow().isoformat(),
                                "status": "open",
                            })
        except Exception as e:
            logger.error(f"Compliance scan EC2 error: {e}")

        service_stats["EC2"] = {"total": ec2_total, "compliant": ec2_compliant,
                                "compliance": round((ec2_compliant / ec2_total * 100) if ec2_total else 100, 1),
                                "color": "#ff9800"}

        # --- EBS Volumes ---
        ebs_total = 0
        ebs_compliant = 0
        try:
            ec2 = self.session.client("ec2")
            volumes = ec2.describe_volumes()
            for vol in volumes.get("Volumes", []):
                ebs_total += 1
                tags = {t["Key"]: t["Value"] for t in vol.get("Tags", [])}
                missing = [t for t in required_tags if t not in tags]
                is_encrypted = vol.get("Encrypted", False)
                issues = []
                if missing:
                    issues.append(f"Missing tags: {', '.join(missing)}")
                if not is_encrypted:
                    issues.append("Volume not encrypted")
                if not issues:
                    ebs_compliant += 1
                else:
                    for issue in issues:
                        severity = "high" if "encrypt" in issue.lower() else "medium"
                        violations.append({
                            "id": vol["VolumeId"],
                            "resource": vol["VolumeId"],
                            "resourceType": "EBS Volume",
                            "policy": "Encryption Policy" if "encrypt" in issue.lower() else "Required Tags Policy",
                            "violation": issue,
                            "severity": severity,
                            "team": tags.get("Owner", "Unknown"),
                            "detected": datetime.utcnow().isoformat(),
                            "status": "open",
                        })
        except Exception as e:
            logger.error(f"Compliance scan EBS error: {e}")

        service_stats["EBS"] = {"total": ebs_total, "compliant": ebs_compliant,
                                "compliance": round((ebs_compliant / ebs_total * 100) if ebs_total else 100, 1),
                                "color": "#2196f3"}

        # --- S3 Buckets ---
        s3_total = 0
        s3_compliant = 0
        try:
            s3 = self.session.client("s3")
            buckets = s3.list_buckets().get("Buckets", [])
            for bucket in buckets:
                s3_total += 1
                bucket_name = bucket["Name"]
                is_compliant = True
                # Check tagging
                try:
                    tag_resp = s3.get_bucket_tagging(Bucket=bucket_name)
                    tags = {t["Key"]: t["Value"] for t in tag_resp.get("TagSet", [])}
                    missing = [t for t in required_tags if t not in tags]
                    if missing:
                        is_compliant = False
                        violations.append({
                            "id": bucket_name, "resource": bucket_name,
                            "resourceType": "S3 Bucket",
                            "policy": "Required Tags Policy",
                            "violation": f"Missing tags: {', '.join(missing)}",
                            "severity": "medium",
                            "team": tags.get("Owner", "Unknown"),
                            "detected": datetime.utcnow().isoformat(), "status": "open",
                        })
                except s3.exceptions.ClientError:
                    is_compliant = False
                    violations.append({
                        "id": bucket_name, "resource": bucket_name,
                        "resourceType": "S3 Bucket",
                        "policy": "Required Tags Policy",
                        "violation": "No tags configured",
                        "severity": "medium", "team": "Unknown",
                        "detected": datetime.utcnow().isoformat(), "status": "open",
                    })
                except Exception:
                    pass
                # Check public access
                try:
                    acl = s3.get_bucket_acl(Bucket=bucket_name)
                    for grant in acl.get("Grants", []):
                        grantee = grant.get("Grantee", {})
                        if grantee.get("URI", "").endswith("AllUsers") or grantee.get("URI", "").endswith("AuthenticatedUsers"):
                            is_compliant = False
                            violations.append({
                                "id": f"{bucket_name}_public", "resource": bucket_name,
                                "resourceType": "S3 Bucket",
                                "policy": "Public Access Policy",
                                "violation": "Public access enabled",
                                "severity": "critical", "team": "Unknown",
                                "detected": datetime.utcnow().isoformat(), "status": "open",
                            })
                            break
                except Exception:
                    pass
                if is_compliant:
                    s3_compliant += 1
        except Exception as e:
            logger.error(f"Compliance scan S3 error: {e}")

        service_stats["S3"] = {"total": s3_total, "compliant": s3_compliant,
                               "compliance": round((s3_compliant / s3_total * 100) if s3_total else 100, 1),
                               "color": "#4caf50"}

        total_resources = sum(s["total"] for s in service_stats.values())
        total_compliant = sum(s["compliant"] for s in service_stats.values())
        overall_score = round((total_compliant / total_resources * 100) if total_resources else 100, 1)

        # Build compliance by service for charts
        compliance_by_service = [
            {"service": svc, "compliant": data["compliance"], "nonCompliant": round(100 - data["compliance"], 1), "color": data["color"]}
            for svc, data in service_stats.items() if data["total"] > 0
        ]

        # Build tagging compliance by owner/team
        owner_stats: Dict[str, Dict[str, int]] = {}
        for svc, data in service_stats.items():
            owner_stats.setdefault("All Resources", {"total": 0, "compliant": 0})
            owner_stats["All Resources"]["total"] += data["total"]
            owner_stats["All Resources"]["compliant"] += data["compliant"]

        tagging_compliance = []
        for team, data in owner_stats.items():
            rate = round((data["compliant"] / data["total"] * 100) if data["total"] else 100, 1)
            tagging_compliance.append({"team": team, "total": data["total"], "compliant": data["compliant"], "compliance": rate})

        open_violations = len([v for v in violations if v["status"] == "open"])
        critical_count = len([v for v in violations if v["severity"] == "critical" and v["status"] == "open"])

        result = {
            "overview": {
                "overallScore": overall_score,
                "taggingCompliance": overall_score,
                "policyCompliance": round(100 - (open_violations / max(total_resources, 1)) * 100, 1),
                "securityCompliance": round(100 - (critical_count / max(total_resources, 1)) * 100, 1),
                "totalResources": total_resources,
                "compliantResources": total_compliant,
                "nonCompliantResources": total_resources - total_compliant,
            },
            "taggingCompliance": tagging_compliance,
            "policyViolations": violations[:50],  # cap at 50
            "complianceByService": compliance_by_service,
        }
        self._set_cached("compliance", result, ttl=600)  # cache 10 min — compliance scans are expensive
        return result

    # ---------------------------------------------------------------
    # Dashboard Summary
    # ---------------------------------------------------------------

    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get complete dashboard summary data."""
        cost_trend = self.get_cost_trend(30)
        service_breakdown = self.get_service_breakdown(30)
        recommendations = self.get_optimization_recommendations()
        anomalies = self.get_cost_anomalies()
        accounts = self.get_accounts()

        total_cost = sum(d["cost"] for d in cost_trend)
        total_savings = sum(r["potential_savings"] for r in recommendations)

        # Calculate budget utilization from actual budgets
        budgets = self.get_budgets()
        total_budget = sum(b["amount"] for b in budgets) if budgets else total_cost * 1.2  # estimate 120% as budget
        budget_utilization = round((total_cost / total_budget) * 100, 1) if total_budget > 0 else 0
        waste_pct = round((total_savings / total_cost) * 100, 1) if total_cost > 0 else 0

        return {
            "overview": {
                "total_monthly_cost": round(total_cost, 2),
                "potential_savings": round(total_savings, 2),
                "savings_percentage": round((total_savings / total_cost) * 100, 1) if total_cost > 0 else 0,
                "active_accounts": len(accounts),
            },
            "cost_trend": cost_trend,
            "service_breakdown": service_breakdown,
            "top_opportunities": [
                {"type": cat, "savings": round(sav, 2), "count": cnt}
                for cat, sav, cnt in _categorize_recommendations(recommendations)
            ],
            "finops_summary": {
                "totalMonthlyCost": round(total_cost, 2),
                "monthlyBudget": round(total_budget, 2),
                "monthlySavings": round(total_savings, 2),
                "wastePercentage": waste_pct,
                "budgetUtilization": budget_utilization,
                "costTrend": "decreasing" if len(cost_trend) >= 2 and cost_trend[-1]["cost"] < cost_trend[0]["cost"] else "increasing",
                "anomaliesCount": len(anomalies),
                "optimizationOpportunities": len(recommendations),
            },
        }

    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()


def _categorize_recommendations(recommendations: List[Dict]) -> List[tuple]:
    """Categorize recommendations into groups for dashboard display."""
    categories: Dict[str, Dict[str, float]] = {}

    type_labels = {
        "terminate_instance": "Unused Instances",
        "resize_underutilized_instances": "Right-sizing",
        "delete_volumes": "Storage Optimization",
        "release_elastic_ips": "Network Cleanup",
        "upgrade_storage": "Storage Optimization",
    }

    for rec in recommendations:
        label = type_labels.get(rec.get("action_type", ""), "Other")
        if label not in categories:
            categories[label] = {"savings": 0, "count": 0}
        categories[label]["savings"] += rec["potential_savings"]
        categories[label]["count"] += 1

    return [(cat, data["savings"], int(data["count"])) for cat, data in categories.items()]
