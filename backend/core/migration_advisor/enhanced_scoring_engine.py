"""
Enhanced Scoring Engine for Cloud Provider Recommendations

Implements weighted scoring algorithm with hard eliminators as specified in PRD Section 2.
"""

from typing import Dict, List, Tuple, Optional
from enum import Enum
from pydantic import BaseModel


class Provider(str, Enum):
    """Supported cloud providers"""
    AWS = "AWS"
    AZURE = "Azure"
    GCP = "GCP"
    IBM = "IBM"
    ORACLE = "Oracle"


class CategoryWeight(BaseModel):
    """Category multipliers for weighted scoring (PRD Section 2.1)"""
    compliance_regulatory: float = 3.0
    workload_type: float = 2.5
    hybrid_integration: float = 2.5
    data_volume_storage: float = 2.0
    budget_cost_model: float = 2.0
    security_encryption: float = 2.0
    ai_ml_requirements: float = 1.5
    support_level: float = 1.5
    vendor_ecosystem: float = 1.5
    scalability: float = 1.5
    open_source_preference: float = 1.0
    existing_vendor_relationship: float = 0.5


class HardEliminator:
    """Pre-scoring filters that eliminate incompatible providers (PRD Section 2.2)"""
    
    @staticmethod
    def check_fedramp(answers: Dict, providers: List[Provider]) -> List[Provider]:
        """Eliminate providers without FedRAMP certification"""
        compliance = answers.get('compliance', {})
        frameworks = compliance.get('regulatory_frameworks', [])
        
        if 'FedRAMP' in frameworks or 'fedramp' in [f.lower() for f in frameworks]:
            # Only AWS and Azure have FedRAMP
            return [p for p in providers if p in [Provider.AWS, Provider.AZURE]]
        return providers
    
    @staticmethod
    def check_data_residency(answers: Dict, providers: List[Provider]) -> List[Provider]:
        """Eliminate providers without required data center"""
        performance = answers.get('performance', {})
        geo_dist = performance.get('geographic_distribution', [])
        
        # Check if specific country requirements exist
        # For now, all major providers have global coverage
        # This would be enhanced with actual region availability data
        return providers
    
    @staticmethod
    def check_hipaa(answers: Dict, providers: List[Provider]) -> List[Provider]:
        """Eliminate providers without HIPAA BAA"""
        compliance = answers.get('compliance', {})
        frameworks = compliance.get('regulatory_frameworks', [])
        
        if 'HIPAA' in frameworks or 'hipaa' in [f.lower() for f in frameworks]:
            # All major providers support HIPAA with BAA
            # IBM and Oracle also support HIPAA
            return providers
        return providers
    
    @staticmethod
    def check_budget(answers: Dict, providers: List[Provider]) -> List[Provider]:
        """Eliminate providers based on budget constraints"""
        budget = answers.get('budget', {})
        monthly_budget = budget.get('target_monthly_cost', 0)
        
        # Very low budgets might not be suitable for enterprise providers
        # Only filter if budget is explicitly very low (< $500/month)
        if monthly_budget > 0 and monthly_budget < 500:
            # Filter out enterprise-focused providers for very small budgets
            return [p for p in providers if p != Provider.IBM]
        return providers
    
    @staticmethod
    def apply_all(answers: Dict) -> List[Provider]:
        """Apply all hard eliminators"""
        providers = list(Provider)
        providers = HardEliminator.check_fedramp(answers, providers)
        providers = HardEliminator.check_data_residency(answers, providers)
        providers = HardEliminator.check_hipaa(answers, providers)
        providers = HardEliminator.check_budget(answers, providers)
        return providers


class EnhancedScoringEngine:
    """Weighted scoring algorithm with category multipliers"""
    
    def __init__(self):
        self.weights = CategoryWeight()
        self.max_possible_score = 100.0  # Normalized to 100
    
    def calculate_scores(self, answers: Dict) -> Dict[Provider, float]:
        """
        Calculate weighted scores for all providers
        
        Args:
            answers: Dictionary containing all user answers
            
        Returns:
            Dictionary mapping providers to scores (0-100, or -1 if eliminated)
        """
        # Initialize scores
        scores = {provider: 0.0 for provider in Provider}
        
        # Apply hard eliminators first
        eligible_providers = HardEliminator.apply_all(answers)
        
        # Mark eliminated providers
        for provider in Provider:
            if provider not in eligible_providers:
                scores[provider] = -1
        
        # Calculate weighted scores for eligible providers
        for provider in eligible_providers:
            score = 0.0
            
            # Workload type scoring
            score += self._score_workload(answers, provider) * self.weights.workload_type
            
            # Technical stack scoring
            score += self._score_tech_stack(answers, provider) * self.weights.vendor_ecosystem
            
            # Compliance scoring
            score += self._score_compliance(answers, provider) * self.weights.compliance_regulatory
            
            # Budget scoring
            score += self._score_budget(answers, provider) * self.weights.budget_cost_model
            
            # AI/ML scoring
            score += self._score_ai_ml(answers, provider) * self.weights.ai_ml_requirements
            
            # Scalability scoring
            score += self._score_scalability(answers, provider) * self.weights.scalability
            
            # Hybrid integration scoring
            score += self._score_hybrid(answers, provider) * self.weights.hybrid_integration
            
            # Data volume scoring
            score += self._score_data_volume(answers, provider) * self.weights.data_volume_storage
            
            # Support level scoring
            score += self._score_support(answers, provider) * self.weights.support_level
            
            # Open source preference
            score += self._score_open_source(answers, provider) * self.weights.open_source_preference
            
            # Normalize to 0-100 scale
            # Max possible raw score is sum of all max deltas × weights
            # Approximate max: 4 × 2.5 + 4 × 2.0 + 3 × 1.5 + 2 × 1.0 = 32.5
            scores[provider] = min(100.0, (score / 32.5) * 100)
        
        return scores

    
    def _score_workload(self, answers: Dict, provider: Provider) -> float:
        """Score based on workload type (PRD Section 2.4)"""
        # Get workload from organization or technical requirements
        org_data = answers.get('organization', {})
        tech_data = answers.get('technical', {})
        
        # Try to infer workload from various fields
        required_services = tech_data.get('required_services', [])
        ml_required = tech_data.get('ml_ai_required', False)
        analytics_required = tech_data.get('analytics_required', False)
        container_required = tech_data.get('container_orchestration', False)
        
        score = 0.0
        
        # Web Apps/APIs (if Compute is primary)
        if 'Compute' in required_services:
            scoring = {Provider.AWS: 3, Provider.GCP: 2, Provider.AZURE: 1, Provider.IBM: 0, Provider.ORACLE: 0}
            score += scoring.get(provider, 0)
        
        # Data Analytics
        if analytics_required or 'Analytics' in required_services:
            scoring = {Provider.AWS: 2, Provider.GCP: 3, Provider.AZURE: 1, Provider.IBM: 0, Provider.ORACLE: 0}
            score += scoring.get(provider, 0)
        
        # AI/ML
        if ml_required:
            scoring = {Provider.AWS: 2, Provider.GCP: 3, Provider.AZURE: 2, Provider.IBM: 1, Provider.ORACLE: 0}
            score += scoring.get(provider, 0)
        
        # Containers/Kubernetes
        if container_required:
            scoring = {Provider.AWS: 2, Provider.GCP: 3, Provider.AZURE: 2, Provider.IBM: 0, Provider.ORACLE: 0}
            score += scoring.get(provider, 0)
        
        # Database workloads
        if 'Database' in required_services:
            scoring = {Provider.AWS: 2, Provider.AZURE: 2, Provider.GCP: 1, Provider.IBM: 1, Provider.ORACLE: 4}
            score += scoring.get(provider, 0)
        
        return min(score, 4.0)  # Cap at 4
    
    def _score_tech_stack(self, answers: Dict, provider: Provider) -> float:
        """Score based on technology stack"""
        org_data = answers.get('organization', {})
        tech_data = answers.get('technical', {})
        
        # Infer tech stack from various signals
        score = 0.0
        
        # Check for Microsoft ecosystem indicators
        industry = org_data.get('industry', '')
        if industry in ['Finance', 'Healthcare', 'Government']:
            # These industries often use Microsoft stack
            if provider == Provider.AZURE:
                score += 2
        
        # Open source preference
        required_services = tech_data.get('required_services', [])
        if 'Compute' in required_services and 'Database' in required_services:
            # General open source workload
            scoring = {Provider.AWS: 3, Provider.GCP: 3, Provider.AZURE: 1, Provider.IBM: 0, Provider.ORACLE: 0}
            score += scoring.get(provider, 0)
        
        return min(score, 4.0)
    
    def _score_compliance(self, answers: Dict, provider: Provider) -> float:
        """Score based on compliance requirements"""
        compliance = answers.get('compliance', {})
        frameworks = compliance.get('regulatory_frameworks', [])
        
        if not frameworks:
            return 0.0
        
        # More compliance requirements favor Azure and AWS
        compliance_count = len(frameworks)
        
        if compliance_count >= 3:
            scoring = {Provider.AZURE: 3, Provider.AWS: 3, Provider.GCP: 1, Provider.IBM: 2, Provider.ORACLE: 1}
        elif compliance_count >= 1:
            scoring = {Provider.AZURE: 2, Provider.AWS: 2, Provider.GCP: 1, Provider.IBM: 1, Provider.ORACLE: 1}
        else:
            return 0.0
        
        return scoring.get(provider, 0)
    
    def _score_budget(self, answers: Dict, provider: Provider) -> float:
        """Score based on budget sensitivity"""
        budget = answers.get('budget', {})
        priority = budget.get('cost_optimization_priority', 'MEDIUM')
        
        if priority == 'HIGH':
            scoring = {Provider.GCP: 3, Provider.AWS: 1, Provider.AZURE: 1, Provider.IBM: 0, Provider.ORACLE: 2}
        elif priority == 'LOW':
            scoring = {Provider.AWS: 2, Provider.AZURE: 2, Provider.GCP: 1, Provider.IBM: 1, Provider.ORACLE: 1}
        else:  # MEDIUM
            scoring = {Provider.AWS: 2, Provider.GCP: 2, Provider.AZURE: 2, Provider.IBM: 1, Provider.ORACLE: 1}
        
        return scoring.get(provider, 0)
    
    def _score_ai_ml(self, answers: Dict, provider: Provider) -> float:
        """Score based on AI/ML requirements"""
        tech_data = answers.get('technical', {})
        ml_required = tech_data.get('ml_ai_required', False)
        
        if ml_required:
            scoring = {Provider.GCP: 3, Provider.AWS: 2, Provider.AZURE: 2, Provider.IBM: 1, Provider.ORACLE: 0}
            return scoring.get(provider, 0)
        return 0.0
    
    def _score_scalability(self, answers: Dict, provider: Provider) -> float:
        """Score based on scalability needs"""
        performance = answers.get('performance', {})
        availability = performance.get('availability_target', 99.0)
        
        # High availability requirements indicate need for strong scalability
        if availability >= 99.9:
            scoring = {Provider.AWS: 3, Provider.GCP: 2, Provider.AZURE: 2, Provider.IBM: 1, Provider.ORACLE: 1}
            return scoring.get(provider, 0)
        elif availability >= 99.5:
            scoring = {Provider.AWS: 2, Provider.GCP: 2, Provider.AZURE: 2, Provider.IBM: 1, Provider.ORACLE: 1}
            return scoring.get(provider, 0)
        return 0.0
    
    def _score_hybrid(self, answers: Dict, provider: Provider) -> float:
        """Score based on hybrid cloud requirements"""
        org_data = answers.get('organization', {})
        current_infra = org_data.get('current_infrastructure', '')
        
        # If currently on-premises or hybrid, favor providers with strong hybrid support
        if current_infra in ['ON_PREMISES', 'HYBRID']:
            scoring = {Provider.AZURE: 3, Provider.AWS: 2, Provider.GCP: 1, Provider.IBM: 3, Provider.ORACLE: 1}
            return scoring.get(provider, 0)
        return 0.0
    
    def _score_data_volume(self, answers: Dict, provider: Provider) -> float:
        """Score based on data volume and storage needs"""
        workload = answers.get('workload', {})
        storage_tb = workload.get('total_storage_tb', 0)
        
        # Large data volumes favor providers with better storage pricing
        if storage_tb > 100:
            scoring = {Provider.AWS: 2, Provider.GCP: 3, Provider.AZURE: 2, Provider.IBM: 1, Provider.ORACLE: 1}
            return scoring.get(provider, 0)
        elif storage_tb > 10:
            scoring = {Provider.AWS: 2, Provider.GCP: 2, Provider.AZURE: 2, Provider.IBM: 1, Provider.ORACLE: 1}
            return scoring.get(provider, 0)
        return 0.0
    
    def _score_support(self, answers: Dict, provider: Provider) -> float:
        """Score based on support level requirements"""
        org_data = answers.get('organization', {})
        company_size = org_data.get('company_size', 'MEDIUM')
        
        # Larger companies need better support
        if company_size in ['ENTERPRISE', 'LARGE']:
            scoring = {Provider.AWS: 3, Provider.AZURE: 3, Provider.GCP: 2, Provider.IBM: 2, Provider.ORACLE: 2}
            return scoring.get(provider, 0)
        return 0.0
    
    def _score_open_source(self, answers: Dict, provider: Provider) -> float:
        """Score based on open source preference"""
        tech_data = answers.get('technical', {})
        container_required = tech_data.get('container_orchestration', False)
        
        # Container orchestration indicates open source preference
        if container_required:
            scoring = {Provider.GCP: 2, Provider.AWS: 2, Provider.AZURE: 1, Provider.IBM: 1, Provider.ORACLE: 0}
            return scoring.get(provider, 0)
        return 0.0
    
    def get_recommendation(self, answers: Dict) -> Tuple[Provider, float, Dict]:
        """
        Get top recommendation with score and evidence
        
        Args:
            answers: Dictionary containing all user answers
            
        Returns:
            Tuple of (provider, score, evidence_dict)
        """
        scores = self.calculate_scores(answers)
        
        # Filter out eliminated providers
        eligible_scores = {p: s for p, s in scores.items() if s >= 0}
        
        if not eligible_scores:
            raise ValueError("No eligible providers found based on requirements")
        
        # Get top provider
        top_provider = max(eligible_scores, key=eligible_scores.get)
        top_score = eligible_scores[top_provider]
        
        # Generate evidence
        evidence = self._generate_evidence(answers, top_provider, top_score)
        
        return top_provider, top_score, evidence
    
    def _generate_evidence(self, answers: Dict, provider: Provider, score: float) -> Dict:
        """Generate explanation for recommendation"""
        evidence_points = []
        
        # Analyze which factors contributed most
        org_data = answers.get('organization', {})
        tech_data = answers.get('technical', {})
        budget_data = answers.get('budget', {})
        compliance_data = answers.get('compliance', {})
        
        # Workload evidence
        required_services = tech_data.get('required_services', [])
        if required_services:
            services_str = ', '.join(required_services[:3])
            evidence_points.append(f"Your workload requires {services_str} → {provider.value} excels in these areas")
        
        # Tech stack evidence
        if tech_data.get('ml_ai_required'):
            evidence_points.append(f"You require AI/ML capabilities → {provider.value} has industry-leading ML services")
        
        if tech_data.get('container_orchestration'):
            evidence_points.append(f"You need container orchestration → {provider.value} has mature Kubernetes support")
        
        # Budget evidence
        cost_priority = budget_data.get('cost_optimization_priority', 'MEDIUM')
        if cost_priority == 'HIGH':
            evidence_points.append(f"Cost optimization is a priority → {provider.value} offers competitive pricing")
        
        # Compliance evidence
        frameworks = compliance_data.get('regulatory_frameworks', [])
        if frameworks:
            evidence_points.append(f"You require {len(frameworks)} compliance framework(s) → {provider.value} meets these requirements")
        
        # Company size evidence
        company_size = org_data.get('company_size', 'MEDIUM')
        if company_size in ['ENTERPRISE', 'LARGE']:
            evidence_points.append(f"Your organization size ({company_size}) → {provider.value} has enterprise-grade support")
        
        # Hybrid evidence
        current_infra = org_data.get('current_infrastructure', '')
        if current_infra in ['ON_PREMISES', 'HYBRID']:
            evidence_points.append(f"Your current infrastructure is {current_infra} → {provider.value} excels at hybrid cloud")
        
        return {
            'evidence_points': evidence_points[:4],  # Top 4 factors
            'provider': provider.value,
            'score': round(score, 1)
        }
    
    def get_all_scores_sorted(self, answers: Dict) -> List[Tuple[Provider, float]]:
        """Get all provider scores sorted by score (highest first)"""
        scores = self.calculate_scores(answers)
        
        # Filter eligible providers and sort
        eligible = [(p, s) for p, s in scores.items() if s >= 0]
        return sorted(eligible, key=lambda x: x[1], reverse=True)
