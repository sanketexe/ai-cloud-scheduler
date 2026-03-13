import pytest
from app.optimizers.resource_optimizer import ResourceOptimizer

@pytest.fixture
def optimizer():
    return ResourceOptimizer()

def test_detect_idle_ec2(optimizer: ResourceOptimizer):
    instances = [
        {
            "instance_id": "i-12345",
            "launch_time": "2023-01-01T00:00:00+00:00",
            "metrics": {
                "CPUUtilization": [{"Value": 1.0}, {"Value": 2.0}],
                "NetworkIn": [{"Value": 1000}],
                "NetworkOut": [{"Value": 1000}]
            }
        },
        {
            "instance_id": "i-active",
            "launch_time": "2023-01-01T00:00:00+00:00",
            "metrics": {
                "CPUUtilization": [{"Value": 50.0}],
                "NetworkIn": [{"Value": 1000000}], # > 5MB
                "NetworkOut": [{"Value": 1000000}]
            }
        }
    ]
    recs = optimizer.detect_idle_instances(instances)
    assert len(recs) == 1
    assert recs[0]["resource_id"] == "i-12345"
    assert recs[0]["issue"] == "Idle Instance"

def test_detect_unused_ebs(optimizer: ResourceOptimizer):
    volumes = [
        {
            "volume_id": "vol-unused",
            "state": "available",
            "metrics": {}
        },
        {
            "volume_id": "vol-active",
            "state": "in-use",
            "metrics": {
                "VolumeReadOps": [{"Value": 10}],
                "VolumeWriteOps": [{"Value": 5}]
            }
        }
    ]
    recs = optimizer.detect_unused_volumes(volumes)
    assert len(recs) == 1
    assert recs[0]["resource_id"] == "vol-unused"
    assert recs[0]["recommendation"] == "Delete"
