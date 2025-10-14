"""
Test Security Framework

Basic tests to verify the security framework components work correctly.
"""

from datetime import datetime, timedelta
from security_manager import security_manager, Permission
from security_framework import UserRole, MFAMethod, SecurityEventType
from compliance_framework import ComplianceStandard, AuditEventType, audit_logger
from security_monitoring import security_monitoring_system
from data_protection import data_protection_manager

def test_user_authentication():
    """Test user authentication flow"""
    print("Testing user authentication...")
    
    # Create a test user
    user = security_manager.auth_manager.create_user(
        username="testuser",
        email="test@example.com",
        password="testpass123",
        role=UserRole.ENGINEER
    )
    
    # Test authentication
    token = security_manager.authenticate_user(
        "testuser",
        "testpass123",
        "127.0.0.1",
        "test-agent"
    )
    
    assert token is not None, "Authentication should succeed"
    
    # Verify session
    session_info = security_manager.verify_session(token)
    assert session_info is not None, "Session should be valid"
    assert session_info['username'] == "testuser", "Username should match"
    
    print("✓ User authentication test passed")

def test_permission_system():
    """Test role-based permission system"""
    print("Testing permission system...")
    
    # Create users with different roles
    admin_user = security_manager.auth_manager.create_user(
        username="admin",
        email="admin@example.com", 
        password="adminpass123",
        role=UserRole.ADMIN
    )
    
    viewer_user = security_manager.auth_manager.create_user(
        username="viewer",
        email="viewer@example.com",
        password="viewerpass123", 
        role=UserRole.VIEWER
    )
    
    # Test admin permissions
    assert security_manager.check_permission(admin_user.user_id, Permission.USER_MANAGE), \
        "Admin should have user management permission"
    
    # Test viewer permissions
    assert not security_manager.check_permission(viewer_user.user_id, Permission.USER_MANAGE), \
        "Viewer should not have user management permission"
    
    assert security_manager.check_permission(viewer_user.user_id, Permission.WORKLOAD_READ), \
        "Viewer should have read permission"
    
    print("✓ Permission system test passed")

def test_api_key_management():
    """Test API key creation and verification"""
    print("Testing API key management...")
    
    # Create a user
    user = security_manager.auth_manager.create_user(
        username="apiuser",
        email="api@example.com",
        password="apipass123",
        role=UserRole.ENGINEER
    )
    
    # Create API key
    permissions = [Permission.API_READ, Permission.API_WRITE]
    full_key, api_key = security_manager.create_api_key(
        user.user_id,
        "test-key",
        permissions,
        creator_user_id=user.user_id,
        ip_address="127.0.0.1",
        user_agent="test-agent"
    )
    
    assert full_key is not None, "API key should be created"
    assert api_key.name == "test-key", "API key name should match"
    
    # Verify API key
    verified_key = security_manager.verify_api_key(
        full_key,
        "127.0.0.1",
        "test-agent"
    )
    
    assert verified_key is not None, "API key should be verified"
    assert verified_key.user_id == user.user_id, "User ID should match"
    
    print("✓ API key management test passed")

def test_audit_logging():
    """Test audit logging functionality"""
    print("Testing audit logging...")
    
    # Log some audit events
    event_id = audit_logger.log_event(
        AuditEventType.USER_LOGIN,
        "test-user-123",
        "authentication",
        None,
        "login_success",
        "SUCCESS",
        "127.0.0.1",
        "test-agent",
        details={"method": "password"}
    )
    
    assert event_id is not None, "Audit event should be logged"
    
    # Retrieve audit events
    events = audit_logger.get_events(
        start_time=datetime.now() - timedelta(minutes=5),
        limit=10
    )
    
    assert len(events) > 0, "Should retrieve audit events"
    assert any(e.event_id == event_id for e in events), "Should find logged event"
    
    print("✓ Audit logging test passed")

def test_data_encryption():
    """Test data encryption functionality"""
    print("Testing data encryption...")
    
    # Initialize encryption keys if not present
    from data_protection import KeyType, EncryptionAlgorithm
    if not data_protection_manager.key_manager.get_active_key(KeyType.DATA_ENCRYPTION_KEY):
        data_protection_manager.key_manager.create_key(
            KeyType.DATA_ENCRYPTION_KEY,
            EncryptionAlgorithm.FERNET
        )
    
    # Test string encryption
    test_data = "sensitive information"
    encrypted = data_protection_manager.encrypt_data(test_data)
    
    assert encrypted is not None, "Data should be encrypted"
    
    decrypted = data_protection_manager.decrypt_data(encrypted)
    decrypted_str = decrypted.decode('utf-8')
    
    assert decrypted_str == test_data, "Decrypted data should match original"
    
    # Test field-level encryption
    test_dict = {
        "username": "john_doe",
        "password": "secret123",
        "email": "john@example.com"
    }
    
    encrypted_dict = data_protection_manager.encrypt_data(
        test_dict,
        sensitive_fields=["password"]
    )
    
    assert encrypted_dict["username"] == "john_doe", "Non-sensitive field should be unchanged"
    assert encrypted_dict["password"] != "secret123", "Sensitive field should be encrypted"
    
    decrypted_dict = data_protection_manager.decrypt_data(
        encrypted_dict,
        sensitive_fields=["password"]
    )
    
    assert decrypted_dict["password"] == "secret123", "Decrypted password should match original"
    
    print("✓ Data encryption test passed")

def test_security_monitoring():
    """Test security monitoring system"""
    print("Testing security monitoring...")
    
    # Process a security event
    event_data = {
        'event_type': 'login_failure',
        'user_id': 'test-user',
        'ip_address': '192.168.1.100',
        'user_agent': 'test-browser',
        'timestamp': datetime.now()
    }
    
    incident_ids = security_monitoring_system.process_event(event_data)
    
    # Process multiple failed login attempts to trigger brute force detection
    for i in range(6):
        event_data['timestamp'] = datetime.now()
        incident_ids.extend(security_monitoring_system.process_event(event_data))
    
    # Should have created incidents for brute force attack
    assert len(incident_ids) > 0, "Should create incidents for brute force attack"
    
    # Get security dashboard
    dashboard = security_monitoring_system.get_security_dashboard()
    
    assert 'summary' in dashboard, "Dashboard should have summary"
    assert 'monitoring_status' in dashboard, "Dashboard should have monitoring status"
    
    print("✓ Security monitoring test passed")

def test_compliance_reporting():
    """Test compliance reporting"""
    print("Testing compliance reporting...")
    
    # Create an admin user for compliance reporting
    admin_user = security_manager.auth_manager.create_user(
        username="compliance_admin",
        email="compliance@example.com",
        password="adminpass123",
        role=UserRole.ADMIN
    )
    
    # Generate a compliance report
    start_date = datetime.now() - timedelta(days=30)
    end_date = datetime.now()
    
    report = security_manager.generate_compliance_report(
        ComplianceStandard.SOC2.value,
        start_date,
        end_date,
        admin_user.user_id
    )
    
    assert report is not None, "Compliance report should be generated"
    assert 'report_type' in report, "Report should have type"
    assert 'summary' in report, "Report should have summary"
    assert 'compliance_checks' in report, "Report should have compliance checks"
    
    print("✓ Compliance reporting test passed")

def run_all_tests():
    """Run all security framework tests"""
    print("Running Security Framework Tests")
    print("=" * 50)
    
    try:
        test_user_authentication()
        test_permission_system()
        test_api_key_management()
        test_audit_logging()
        test_data_encryption()
        test_security_monitoring()
        test_compliance_reporting()
        
        print("\n" + "=" * 50)
        print("✓ All security framework tests passed!")
        print("Security framework is working correctly.")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        raise

if __name__ == "__main__":
    run_all_tests()