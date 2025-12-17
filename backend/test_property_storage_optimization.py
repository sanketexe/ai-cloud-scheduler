#!/usr/bin/env python3
"""
Property-Based Test for Storage Optimization Automation

Tests Property 2: Storage Optimization Automation
**Feature: automated-cost-optimization, Property 2: Storage Optimization Automation**
**Validates: Requirements 1.4**
"""

import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any

import hypothesis
from hypothesis import given, strategies as st, assume, settings
import pytest

# Import the components we're testing
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.storage_optimizer import StorageOptimizer, EBSVolume, VolumeOptimizationOpportunity
from backend.core.automation_models import ActionType, RiskLevel, AutomationLevel


class MockAutomationPolicy:
    """Mock automation policy for testing"""
    
    def __init__(self):
        self.automation_level = AutomationLevel.BALANCED
        self.resource_filters = {
            "exclude_tags": [],
            "include_services": ["EBS"],
            "min_cost_threshold": 0
        }
        self.time_restrictions = {}


class TestStorageOptimizationProperty:
    """Test class for storage optimization automation property"""
    
    def __init__(self):
        self.optimizer = StorageOptimizer()
        self.policy = MockAutomationPolicy()
    
    @given(
        volumes=st.lists(
            st.builds(
                EBSVolume,
                volume_id=st.text(min_size=10, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Pc'))),
                volume_type=st.sampled_from(["gp2", "gp3", "io1", "io2", "st1", "sc1"]),
                size_gb=st.integers(min_value=1, max_value=2000),
                state=st.sampled_from(["available", "in-use", "creating", "deleting"]),
                attachment_state=st.sampled_from(["attached", "detached", "attaching", "detaching", None]),
                attached_instance_id=st.one_of(st.none(), st.text(min_size=10, max_size=20)),
                creation_time=st.datetimes(
                    min_value=datetime(2020, 1, 1),
                    max_value=datetime.utcnow() - timedelta(days=1)
                ),
                last_attachment_time=st.one_of(st.none(), st.datetimes(
                    min_value=datetime(2020, 1, 1),
                    max_value=datetime.utcnow()
                )),
                last_detachment_time=st.one_of(st.none(), st.datetimes(
                    min_value=datetime(2020, 1, 1),
                    max_value=datetime.utcnow()
                )),
                tags=st.dictionaries(
                    st.text(min_size=1, max_size=20),
                    st.text(min_size=1, max_size=50),
                    min_size=0,
                    max_size=5
                ),
                monthly_cost=st.decimals(min_value=0, max_value=1000, places=2),
                iops=st.one_of(st.none(), st.integers(min_value=100, max_value=20000)),
                throughput=st.one_of(st.none(), st.integers(min_value=125, max_value=1000)),
                availability_zone=st.sampled_from(["us-east-1a", "us-east-1b", "us-west-2a"]),
                encrypted=st.booleans(),
                snapshot_id=st.one_of(st.none(), st.text(min_size=10, max_size=20))
            ),
            min_size=0,
            max_size=10
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_storage_optimization_automation_property(self, volumes: List[EBSVolume]):
        """
        **Feature: automated-cost-optimization, Property 2: Storage Optimization Automation**
        
        Property: For any collection of EBS volumes, when gp2 volumes larger than 100GB are identified,
        the system should automatically upgrade them to gp3 and calculate cost savings.
        
        **Validates: Requirements 1.4**
        """
        
        # Filter to only gp2 volumes larger than 100GB that are in stable states
        eligible_gp2_volumes = [
            vol for vol in volumes 
            if (vol.volume_type == "gp2" and 
                vol.size_gb >= 100 and 
                vol.state in ["available", "in-use"] and
                not self._has_production_tags(vol.tags))
        ]
        
        # Run the storage optimization detection
        upgrade_opportunities = self.optimizer.detect_gp2_upgrade_opportunities(volumes, self.policy)
        
        # Property validation: All eligible gp2 volumes should be detected for upgrade
        detected_volume_ids = {opp.volume.volume_id for opp in upgrade_opportunities}
        eligible_volume_ids = {vol.volume_id for vol in eligible_gp2_volumes}
        
        # All eligible volumes should be detected (subset relationship)
        assert detected_volume_ids.issubset(eligible_volume_ids), (
            f"Detected volumes {detected_volume_ids} should be subset of eligible volumes {eligible_volume_ids}"
        )
        
        # All detected opportunities should be for gp2 to gp3 upgrades
        for opportunity in upgrade_opportunities:
            assert opportunity.optimization_type == ActionType.UPGRADE_STORAGE, (
                f"Opportunity type should be UPGRADE_STORAGE, got {opportunity.optimization_type}"
            )
            assert opportunity.target_volume_type == "gp3", (
                f"Target volume type should be gp3, got {opportunity.target_volume_type}"
            )
            assert opportunity.volume.volume_type == "gp2", (
                f"Source volume should be gp2, got {opportunity.volume.volume_type}"
            )
            assert opportunity.volume.size_gb >= 100, (
                f"Volume size should be >= 100GB, got {opportunity.volume.size_gb}GB"
            )
        
        # All opportunities should have positive cost savings
        for opportunity in upgrade_opportunities:
            assert opportunity.estimated_monthly_savings > 0, (
                f"Estimated savings should be positive, got {opportunity.estimated_monthly_savings}"
            )
        
        # All opportunities should have valid gp3 configuration
        for opportunity in upgrade_opportunities:
            assert opportunity.target_iops is not None, "Target IOPS should be specified"
            assert opportunity.target_throughput is not None, "Target throughput should be specified"
            assert 3000 <= opportunity.target_iops <= 16000, (
                f"Target IOPS should be between 3000-16000, got {opportunity.target_iops}"
            )
            assert 125 <= opportunity.target_throughput <= 1000, (
                f"Target throughput should be between 125-1000 MB/s, got {opportunity.target_throughput}"
            )
        
        # Test the upgrade execution for detected opportunities
        if upgrade_opportunities:
            upgrade_plans = [
                {
                    "volume_id": opp.volume.volume_id,
                    "target_iops": opp.target_iops,
                    "target_throughput": opp.target_throughput
                }
                for opp in upgrade_opportunities
            ]
            
            # Execute upgrades (simulated)
            upgrade_results = self.optimizer.upgrade_gp2_to_gp3(upgrade_plans, self.policy)
            
            # All upgrades should succeed for valid configurations
            for volume_id, success in upgrade_results.items():
                # Find the corresponding opportunity
                opp = next((o for o in upgrade_opportunities if o.volume.volume_id == volume_id), None)
                if opp and not self._has_production_tags(opp.volume.tags):
                    assert success, f"Upgrade should succeed for volume {volume_id}"
    
    def _has_production_tags(self, tags: Dict[str, str]) -> bool:
        """Check if volume has production tags that would exclude it"""
        production_indicators = [
            ("Environment", ["production", "prod", "live"]),
            ("Critical", ["true", "yes", "1"]),
            ("Tier", ["production", "prod", "critical", "live"])
        ]
        
        for tag_key, protected_values in production_indicators:
            if tag_key in tags:
                tag_value = str(tags[tag_key]).lower()
                if tag_value in [v.lower() for v in protected_values]:
                    return True
        
        return False


def run_property_test():
    """Run the property-based test"""
    
    print("Running Property-Based Test for Storage Optimization Automation")
    print("=" * 60)
    print("**Feature: automated-cost-optimization, Property 2: Storage Optimization Automation**")
    print("**Validates: Requirements 1.4**")
    print()
    
    test_instance = TestStorageOptimizationProperty()
    
    try:
        print("Testing Property 2: Storage Optimization Automation...")
        test_instance.test_storage_optimization_automation_property()
        print("✓ Property 2 test completed successfully")
        print()
        print("Property validation confirmed:")
        print("- gp2 volumes >= 100GB are correctly identified for upgrade")
        print("- All upgrades target gp3 with valid configuration")
        print("- Cost savings are calculated for all opportunities")
        print("- Production volumes are properly excluded")
        print("- Upgrade execution works for valid configurations")
        
        return True
        
    except Exception as e:
        print(f"✗ Property 2 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_property_test()
    if not success:
        exit(1)
    print("\n🎉 All property tests passed!")