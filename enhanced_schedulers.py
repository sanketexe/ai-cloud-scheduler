# enhanced_schedulers.py
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
from enhanced_models import (
    EnhancedWorkload, EnhancedVirtualMachine, WorkloadPriority,
    CostConstraints, PerformanceRequirements, ComplianceRequirements
)
from schedulers import BaseScheduler


@dataclass
class PlacementPrediction:
    """Prediction of placement outcome"""
    vm_id: int
    workload_id: int
    predicted_cost: float
    predicted_performance_score: float
    confidence_score: float  # 0-1
    estimated_completion_time: Optional[datetime] = None
    risk_factors: List[str] = None
    
    def __post_init__(self):
        if self.risk_factors is None:
            self.risk_factors = []


@dataclass
class Recommendation:
    """Optimization recommendation"""
    recommendation_id: str
    recommendation_type: str  # "cost_optimization", "performance_improvement", "scaling"
    description: str
    current_state: Dict[str, Any]
    recommended_action: str
    estimated_impact: Dict[str, float]  # e.g., {"cost_savings": 100.0, "performance_gain": 15.0}
    confidence_score: float
    implementation_effort: str  # "low", "medium", "high"
    created_at: datetime


class EnhancedScheduler(BaseScheduler):
    """Enhanced base scheduler with cost and performance awareness"""
    
    def __init__(self, name: str = "EnhancedScheduler"):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.placement_history: List[Dict[str, Any]] = []
        self.performance_weights = {
            'cost': 0.4,
            'performance': 0.3,
            'availability': 0.2,
            'compliance': 0.1
        }
    
    def select_vm(self, workload, vms):
        """Enhanced VM selection with cost and performance considerations"""
        if isinstance(workload, EnhancedWorkload):
            return self.select_vm_enhanced(workload, vms)
        else:
            # Fallback to basic selection for legacy workloads
            return self._select_vm_basic(workload, vms)
    
    def select_vm_enhanced(self, workload: EnhancedWorkload, 
                          vms: List[EnhancedVirtualMachine]) -> Optional[EnhancedVirtualMachine]:
        """Enhanced VM selection for EnhancedWorkload"""
        # Filter VMs that can accommodate the workload
        eligible_vms = [vm for vm in vms if vm.can_accommodate_enhanced(workload)]
        
        if not eligible_vms:
            self.logger.warning(f"No eligible VMs found for workload {workload.id}")
            return None
        
        # Score and rank VMs
        scored_vms = []
        for vm in eligible_vms:
            score = self._calculate_vm_score(workload, vm)
            prediction = self.predict_placement_outcome(workload, vm)
            scored_vms.append((vm, score, prediction))
        
        # Sort by score (higher is better)
        scored_vms.sort(key=lambda x: x[1], reverse=True)
        
        selected_vm = scored_vms[0][0]
        
        # Log placement decision
        self._log_placement_decision(workload, selected_vm, scored_vms[0][2])
        
        return selected_vm
    
    def _select_vm_basic(self, workload, vms) -> Optional:
        """Basic VM selection for legacy workloads"""
        eligible_vms = [vm for vm in vms if vm.can_accommodate(workload)]
        if not eligible_vms:
            return None
        return eligible_vms[0]  # Simple first-fit
    
    def _calculate_vm_score(self, workload: EnhancedWorkload, 
                           vm: EnhancedVirtualMachine) -> float:
        """Calculate overall score for VM placement"""
        scores = {}
        
        # Cost score (lower cost = higher score)
        cost_score = self._calculate_cost_score(workload, vm)
        scores['cost'] = cost_score
        
        # Performance score
        performance_score = self._calculate_performance_score(workload, vm)
        scores['performance'] = performance_score
        
        # Availability score
        availability_score = self._calculate_availability_score(vm)
        scores['availability'] = availability_score
        
        # Compliance score
        compliance_score = self._calculate_compliance_score(workload, vm)
        scores['compliance'] = compliance_score
        
        # Weighted total score
        total_score = sum(
            scores[factor] * weight 
            for factor, weight in self.performance_weights.items()
        )
        
        return total_score
    
    def _calculate_cost_score(self, workload: EnhancedWorkload, 
                             vm: EnhancedVirtualMachine) -> float:
        """Calculate cost-based score (0-100)"""
        current_cost = vm.get_current_hourly_cost()
        
        if workload.cost_constraints:
            max_cost = workload.cost_constraints.max_hourly_cost
            if current_cost >= max_cost:
                return 0.0  # Exceeds budget
            # Score based on how much under budget we are
            return ((max_cost - current_cost) / max_cost) * 100
        
        # If no constraints, prefer lower cost
        # Normalize against a reasonable maximum (e.g., $10/hour)
        max_reasonable_cost = 10.0
        return max(0, (max_reasonable_cost - current_cost) / max_reasonable_cost * 100)
    
    def _calculate_performance_score(self, workload: EnhancedWorkload, 
                                   vm: EnhancedVirtualMachine) -> float:
        """Calculate performance-based score (0-100)"""
        if not workload.performance_requirements:
            return vm.performance_score  # Use VM's inherent performance score
        
        if not vm.current_performance_metrics:
            return vm.performance_score * 0.8  # Penalty for no metrics
        
        score = 0.0
        factors = 0
        
        # CPU performance
        if vm.performance_score >= workload.performance_requirements.min_cpu_performance:
            score += vm.performance_score
        else:
            score += vm.performance_score * 0.5  # Penalty for not meeting requirements
        factors += 1
        
        # Latency (lower is better)
        metrics = vm.current_performance_metrics
        required_latency = workload.performance_requirements.max_latency_ms
        if metrics.response_time_ms <= required_latency:
            latency_score = 100 - (metrics.response_time_ms / required_latency * 50)
            score += max(50, latency_score)
        else:
            score += 25  # Penalty for exceeding latency requirement
        factors += 1
        
        # Throughput
        if workload.performance_requirements.throughput_requirement:
            if metrics.throughput >= workload.performance_requirements.throughput_requirement:
                score += 100
            else:
                ratio = metrics.throughput / workload.performance_requirements.throughput_requirement
                score += ratio * 100
            factors += 1
        
        return score / factors if factors > 0 else 50.0
    
    def _calculate_availability_score(self, vm: EnhancedVirtualMachine) -> float:
        """Calculate availability-based score (0-100)"""
        # Base score on health status
        health_scores = {
            vm.health_status.HEALTHY: 100.0,
            vm.health_status.WARNING: 70.0,
            vm.health_status.UNHEALTHY: 30.0,
            vm.health_status.UNKNOWN: 50.0
        }
        
        base_score = health_scores.get(vm.health_status, 50.0)
        
        # Adjust based on utilization (prefer less utilized VMs)
        if vm.current_performance_metrics:
            avg_utilization = (
                vm.current_performance_metrics.cpu_utilization + 
                vm.current_performance_metrics.memory_utilization
            ) / 2
            utilization_penalty = avg_utilization * 0.3  # Up to 30% penalty
            base_score = max(0, base_score - utilization_penalty)
        
        return base_score
    
    def _calculate_compliance_score(self, workload: EnhancedWorkload, 
                                  vm: EnhancedVirtualMachine) -> float:
        """Calculate compliance-based score (0-100)"""
        if vm.meets_compliance_requirements(workload.compliance_requirements):
            return 100.0
        else:
            return 0.0  # Binary: either compliant or not
    
    def predict_placement_outcome(self, workload: EnhancedWorkload, 
                                vm: EnhancedVirtualMachine) -> PlacementPrediction:
        """Predict the outcome of placing workload on VM"""
        predicted_cost = vm.get_current_hourly_cost()
        if workload.estimated_duration_hours:
            predicted_cost *= workload.estimated_duration_hours
        
        predicted_performance = self._calculate_performance_score(workload, vm)
        
        # Calculate confidence based on available data
        confidence = 0.7  # Base confidence
        if vm.current_performance_metrics:
            confidence += 0.2
        if vm.cost_history:
            confidence += 0.1
        
        # Identify risk factors
        risk_factors = []
        if vm.health_status != vm.health_status.HEALTHY:
            risk_factors.append(f"VM health status: {vm.health_status.value}")
        
        if vm.current_performance_metrics:
            if vm.current_performance_metrics.cpu_utilization > 80:
                risk_factors.append("High CPU utilization")
            if vm.current_performance_metrics.memory_utilization > 80:
                risk_factors.append("High memory utilization")
        
        return PlacementPrediction(
            vm_id=vm.vm_id,
            workload_id=workload.id,
            predicted_cost=predicted_cost,
            predicted_performance_score=predicted_performance,
            confidence_score=min(1.0, confidence),
            risk_factors=risk_factors
        )
    
    def get_optimization_recommendations(self, vms: List[EnhancedVirtualMachine]) -> List[Recommendation]:
        """Generate optimization recommendations based on current state"""
        recommendations = []
        
        for vm in vms:
            # Cost optimization recommendations
            if vm.current_performance_metrics:
                avg_util = (
                    vm.current_performance_metrics.cpu_utilization + 
                    vm.current_performance_metrics.memory_utilization
                ) / 2
                
                if avg_util < 20:
                    recommendations.append(Recommendation(
                        recommendation_id=f"cost_opt_{vm.vm_id}_{datetime.now().timestamp()}",
                        recommendation_type="cost_optimization",
                        description=f"VM {vm.vm_id} is underutilized ({avg_util:.1f}% average)",
                        current_state={"utilization": avg_util, "cost": vm.get_current_hourly_cost()},
                        recommended_action="Consider downsizing or consolidating workloads",
                        estimated_impact={"cost_savings": vm.get_current_hourly_cost() * 0.3},
                        confidence_score=0.8,
                        implementation_effort="medium",
                        created_at=datetime.now()
                    ))
                
                elif avg_util > 85:
                    recommendations.append(Recommendation(
                        recommendation_id=f"perf_opt_{vm.vm_id}_{datetime.now().timestamp()}",
                        recommendation_type="performance_improvement",
                        description=f"VM {vm.vm_id} is highly utilized ({avg_util:.1f}% average)",
                        current_state={"utilization": avg_util, "performance": vm.performance_score},
                        recommended_action="Consider scaling up or load balancing",
                        estimated_impact={"performance_gain": 25.0},
                        confidence_score=0.9,
                        implementation_effort="low",
                        created_at=datetime.now()
                    ))
        
        return recommendations
    
    def _log_placement_decision(self, workload: EnhancedWorkload, 
                              vm: EnhancedVirtualMachine, 
                              prediction: PlacementPrediction):
        """Log placement decision for analysis"""
        log_entry = {
            'timestamp': datetime.now(),
            'workload_id': workload.id,
            'vm_id': vm.vm_id,
            'scheduler': self.name,
            'predicted_cost': prediction.predicted_cost,
            'predicted_performance': prediction.predicted_performance_score,
            'confidence': prediction.confidence_score,
            'risk_factors': prediction.risk_factors
        }
        self.placement_history.append(log_entry)
        
        self.logger.info(f"Placed workload {workload.id} on VM {vm.vm_id} "
                        f"(cost: ${prediction.predicted_cost:.2f}, "
                        f"performance: {prediction.predicted_performance_score:.1f})")


class CostAwareScheduler(EnhancedScheduler):
    """Scheduler that optimizes placement based on real-time pricing"""
    
    def __init__(self):
        super().__init__("CostAwareScheduler")
        self.performance_weights = {
            'cost': 0.7,  # Higher weight on cost
            'performance': 0.15,
            'availability': 0.1,
            'compliance': 0.05
        }
    
    def select_vm_enhanced(self, workload: EnhancedWorkload, 
                          vms: List[EnhancedVirtualMachine]) -> Optional[EnhancedVirtualMachine]:
        """Cost-optimized VM selection"""
        # Filter by cost constraints first
        eligible_vms = []
        for vm in vms:
            if vm.can_accommodate_enhanced(workload):
                # Additional cost filtering
                if workload.cost_constraints:
                    current_cost = vm.get_current_hourly_cost()
                    max_cost = workload.cost_constraints.max_hourly_cost
                    
                    # Apply cost optimization preference
                    if workload.cost_constraints.cost_optimization_preference.value == "aggressive":
                        # Only consider VMs using <70% of budget
                        if current_cost <= max_cost * 0.7:
                            eligible_vms.append(vm)
                    elif workload.cost_constraints.cost_optimization_preference.value == "moderate":
                        # Consider VMs using <90% of budget
                        if current_cost <= max_cost * 0.9:
                            eligible_vms.append(vm)
                    else:
                        # Use full budget if needed
                        if current_cost <= max_cost:
                            eligible_vms.append(vm)
                else:
                    eligible_vms.append(vm)
        
        if not eligible_vms:
            return None
        
        # Use parent class scoring but with cost-focused weights
        return super().select_vm_enhanced(workload, eligible_vms)


class PerformanceScheduler(EnhancedScheduler):
    """Scheduler that considers historical performance data"""
    
    def __init__(self):
        super().__init__("PerformanceScheduler")
        self.performance_weights = {
            'cost': 0.1,
            'performance': 0.6,  # Higher weight on performance
            'availability': 0.25,
            'compliance': 0.05
        }
    
    def select_vm_enhanced(self, workload: EnhancedWorkload, 
                          vms: List[EnhancedVirtualMachine]) -> Optional[EnhancedVirtualMachine]:
        """Performance-optimized VM selection"""
        # Filter VMs that meet performance requirements
        eligible_vms = []
        for vm in vms:
            if vm.can_accommodate_enhanced(workload):
                # Additional performance filtering
                if workload.performance_requirements:
                    if vm.meets_performance_requirements(workload.performance_requirements):
                        # Prefer VMs with better performance history
                        if vm.utilization_trends:
                            # Avoid VMs with consistently high utilization
                            if vm.utilization_trends.avg_cpu_utilization_24h < 80:
                                eligible_vms.append(vm)
                        else:
                            eligible_vms.append(vm)
                else:
                    eligible_vms.append(vm)
        
        if not eligible_vms:
            # Fallback to basic accommodation if no VMs meet strict performance criteria
            eligible_vms = [vm for vm in vms if vm.can_accommodate_enhanced(workload)]
        
        if not eligible_vms:
            return None
        
        return super().select_vm_enhanced(workload, eligible_vms)
    
    def _calculate_performance_score(self, workload: EnhancedWorkload, 
                                   vm: EnhancedVirtualMachine) -> float:
        """Enhanced performance scoring with historical data"""
        base_score = super()._calculate_performance_score(workload, vm)
        
        # Bonus for good historical performance
        if vm.utilization_trends:
            trends = vm.utilization_trends
            
            # Bonus for stable performance
            if trends.trend_direction == "stable":
                base_score += 10
            
            # Penalty for consistently high utilization
            if trends.avg_cpu_utilization_24h > 85 or trends.avg_memory_utilization_24h > 85:
                base_score -= 20
            
            # Bonus for low average utilization (more headroom)
            if trends.avg_cpu_utilization_24h < 50 and trends.avg_memory_utilization_24h < 50:
                base_score += 15
        
        return min(100, max(0, base_score))