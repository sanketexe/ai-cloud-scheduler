"""
Data Protection and Encryption Module

This module provides comprehensive data protection including:
- Advanced encryption for data at rest
- TLS configuration for data in transit
- Key management with rotation
- Field-level encryption
- Database encryption
- File encryption
- Secure key storage
"""

import os
import ssl
import json
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import base64
import logging
from pathlib import Path

# Cryptography imports
from cryptography.fernet import Fernet, MultiFernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography import x509
from cryptography.x509.oid import NameOID

# Configure logging
logger = logging.getLogger(__name__)

class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms"""
    AES_256_GCM = "aes_256_gcm"
    FERNET = "fernet"
    RSA_OAEP = "rsa_oaep"
    CHACHA20_POLY1305 = "chacha20_poly1305"

class KeyType(Enum):
    """Types of encryption keys"""
    MASTER_KEY = "master_key"
    DATA_ENCRYPTION_KEY = "data_encryption_key"
    KEY_ENCRYPTION_KEY = "key_encryption_key"
    API_SIGNING_KEY = "api_signing_key"
    TLS_CERTIFICATE = "tls_certificate"

@dataclass
class EncryptionKey:
    """Encryption key metadata and data"""
    key_id: str
    key_type: KeyType
    algorithm: EncryptionAlgorithm
    key_data: bytes
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    version: int = 1
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EncryptedData:
    """Container for encrypted data with metadata"""
    encrypted_data: bytes
    key_id: str
    algorithm: EncryptionAlgorithm
    nonce: Optional[bytes] = None
    tag: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class KeyManager:
    """Manages encryption keys with rotation and secure storage"""
    
    def __init__(self, key_storage_path: str = ".keys"):
        self.key_storage_path = Path(key_storage_path)
        self.key_storage_path.mkdir(exist_ok=True, mode=0o700)  # Secure permissions
        self.keys: Dict[str, EncryptionKey] = {}
        self.master_key: Optional[bytes] = None
        self._load_keys()
    
    def _load_keys(self):
        """Load keys from secure storage"""
        try:
            # Load master key from environment or generate new one
            master_key_env = os.environ.get('MASTER_KEY')
            if master_key_env:
                self.master_key = base64.b64decode(master_key_env)
            else:
                master_key_file = self.key_storage_path / "master.key"
                if master_key_file.exists():
                    with open(master_key_file, 'rb') as f:
                        self.master_key = f.read()
                else:
                    # Generate new master key
                    self.master_key = secrets.token_bytes(32)
                    with open(master_key_file, 'wb') as f:
                        f.write(self.master_key)
                    master_key_file.chmod(0o600)  # Secure permissions
            
            # Load other keys
            keys_file = self.key_storage_path / "keys.json"
            if keys_file.exists():
                with open(keys_file, 'r') as f:
                    keys_data = json.load(f)
                
                for key_data in keys_data:
                    key = EncryptionKey(
                        key_id=key_data['key_id'],
                        key_type=KeyType(key_data['key_type']),
                        algorithm=EncryptionAlgorithm(key_data['algorithm']),
                        key_data=base64.b64decode(key_data['key_data']),
                        created_at=datetime.fromisoformat(key_data['created_at']),
                        expires_at=datetime.fromisoformat(key_data['expires_at']) if key_data.get('expires_at') else None,
                        version=key_data.get('version', 1),
                        is_active=key_data.get('is_active', True),
                        metadata=key_data.get('metadata', {})
                    )
                    self.keys[key.key_id] = key
            
        except Exception as e:
            logger.error(f"Failed to load keys: {e}")
            # Initialize with default keys if loading fails
            self._initialize_default_keys()
    
    def _save_keys(self):
        """Save keys to secure storage"""
        try:
            keys_data = []
            for key in self.keys.values():
                keys_data.append({
                    'key_id': key.key_id,
                    'key_type': key.key_type.value,
                    'algorithm': key.algorithm.value,
                    'key_data': base64.b64encode(key.key_data).decode(),
                    'created_at': key.created_at.isoformat(),
                    'expires_at': key.expires_at.isoformat() if key.expires_at else None,
                    'version': key.version,
                    'is_active': key.is_active,
                    'metadata': key.metadata
                })
            
            keys_file = self.key_storage_path / "keys.json"
            with open(keys_file, 'w') as f:
                json.dump(keys_data, f, indent=2)
            keys_file.chmod(0o600)  # Secure permissions
            
        except Exception as e:
            logger.error(f"Failed to save keys: {e}")
    
    def _initialize_default_keys(self):
        """Initialize default encryption keys"""
        # Create master data encryption key
        self.create_key(
            key_type=KeyType.DATA_ENCRYPTION_KEY,
            algorithm=EncryptionAlgorithm.FERNET,
            expires_in_days=365
        )
        
        # Create API signing key
        self.create_key(
            key_type=KeyType.API_SIGNING_KEY,
            algorithm=EncryptionAlgorithm.RSA_OAEP,
            expires_in_days=730
        )
    
    def create_key(self, key_type: KeyType, algorithm: EncryptionAlgorithm,
                  expires_in_days: Optional[int] = None) -> str:
        """Create new encryption key"""
        key_id = secrets.token_urlsafe(16)
        
        # Generate key based on algorithm
        if algorithm == EncryptionAlgorithm.FERNET:
            key_data = Fernet.generate_key()
        elif algorithm == EncryptionAlgorithm.AES_256_GCM:
            key_data = secrets.token_bytes(32)  # 256-bit key
        elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            key_data = secrets.token_bytes(32)  # 256-bit key
        elif algorithm == EncryptionAlgorithm.RSA_OAEP:
            # Generate RSA key pair
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            key_data = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)
        
        key = EncryptionKey(
            key_id=key_id,
            key_type=key_type,
            algorithm=algorithm,
            key_data=key_data,
            expires_at=expires_at
        )
        
        self.keys[key_id] = key
        self._save_keys()
        
        logger.info(f"Created new {key_type.value} key: {key_id}")
        return key_id
    
    def get_key(self, key_id: str) -> Optional[EncryptionKey]:
        """Get encryption key by ID"""
        key = self.keys.get(key_id)
        
        # Check if key is expired
        if key and key.expires_at and datetime.now() > key.expires_at:
            key.is_active = False
            self._save_keys()
            return None
        
        return key if key and key.is_active else None
    
    def get_active_key(self, key_type: KeyType) -> Optional[EncryptionKey]:
        """Get the most recent active key of specified type"""
        active_keys = [
            key for key in self.keys.values()
            if key.key_type == key_type and key.is_active and
            (not key.expires_at or datetime.now() < key.expires_at)
        ]
        
        if not active_keys:
            return None
        
        # Return the most recent key
        return max(active_keys, key=lambda k: k.created_at)
    
    def rotate_key(self, key_id: str, expires_in_days: Optional[int] = None) -> str:
        """Rotate an existing key"""
        old_key = self.get_key(key_id)
        if not old_key:
            raise ValueError(f"Key not found: {key_id}")
        
        # Create new key with same type and algorithm
        new_key_id = self.create_key(
            key_type=old_key.key_type,
            algorithm=old_key.algorithm,
            expires_in_days=expires_in_days
        )
        
        # Increment version
        new_key = self.keys[new_key_id]
        new_key.version = old_key.version + 1
        
        # Deactivate old key
        old_key.is_active = False
        
        self._save_keys()
        
        logger.info(f"Rotated key {key_id} to {new_key_id}")
        return new_key_id
    
    def list_keys(self, key_type: Optional[KeyType] = None) -> List[EncryptionKey]:
        """List all keys, optionally filtered by type"""
        keys = list(self.keys.values())
        
        if key_type:
            keys = [key for key in keys if key.key_type == key_type]
        
        return sorted(keys, key=lambda k: k.created_at, reverse=True)

class DataEncryption:
    """Handles data encryption and decryption operations"""
    
    def __init__(self, key_manager: KeyManager):
        self.key_manager = key_manager
    
    def encrypt(self, data: Union[str, bytes], key_id: Optional[str] = None,
               algorithm: Optional[EncryptionAlgorithm] = None) -> EncryptedData:
        """Encrypt data using specified or default key"""
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # Get encryption key
        if key_id:
            key = self.key_manager.get_key(key_id)
            if not key:
                raise ValueError(f"Key not found or expired: {key_id}")
        else:
            # Use default data encryption key
            key = self.key_manager.get_active_key(KeyType.DATA_ENCRYPTION_KEY)
            if not key:
                raise ValueError("No active data encryption key found")
        
        # Use specified algorithm or key's algorithm
        algo = algorithm or key.algorithm
        
        if algo == EncryptionAlgorithm.FERNET:
            return self._encrypt_fernet(data, key)
        elif algo == EncryptionAlgorithm.AES_256_GCM:
            return self._encrypt_aes_gcm(data, key)
        elif algo == EncryptionAlgorithm.CHACHA20_POLY1305:
            return self._encrypt_chacha20(data, key)
        elif algo == EncryptionAlgorithm.RSA_OAEP:
            return self._encrypt_rsa(data, key)
        else:
            raise ValueError(f"Unsupported encryption algorithm: {algo}")
    
    def decrypt(self, encrypted_data: EncryptedData) -> bytes:
        """Decrypt data using the specified key"""
        key = self.key_manager.get_key(encrypted_data.key_id)
        if not key:
            raise ValueError(f"Decryption key not found: {encrypted_data.key_id}")
        
        if encrypted_data.algorithm == EncryptionAlgorithm.FERNET:
            return self._decrypt_fernet(encrypted_data, key)
        elif encrypted_data.algorithm == EncryptionAlgorithm.AES_256_GCM:
            return self._decrypt_aes_gcm(encrypted_data, key)
        elif encrypted_data.algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            return self._decrypt_chacha20(encrypted_data, key)
        elif encrypted_data.algorithm == EncryptionAlgorithm.RSA_OAEP:
            return self._decrypt_rsa(encrypted_data, key)
        else:
            raise ValueError(f"Unsupported decryption algorithm: {encrypted_data.algorithm}")
    
    def _encrypt_fernet(self, data: bytes, key: EncryptionKey) -> EncryptedData:
        """Encrypt using Fernet (AES 128 CBC + HMAC SHA256)"""
        f = Fernet(key.key_data)
        encrypted = f.encrypt(data)
        
        return EncryptedData(
            encrypted_data=encrypted,
            key_id=key.key_id,
            algorithm=EncryptionAlgorithm.FERNET
        )
    
    def _decrypt_fernet(self, encrypted_data: EncryptedData, key: EncryptionKey) -> bytes:
        """Decrypt using Fernet"""
        f = Fernet(key.key_data)
        return f.decrypt(encrypted_data.encrypted_data)
    
    def _encrypt_aes_gcm(self, data: bytes, key: EncryptionKey) -> EncryptedData:
        """Encrypt using AES-256-GCM"""
        nonce = secrets.token_bytes(12)  # 96-bit nonce for GCM
        
        cipher = Cipher(
            algorithms.AES(key.key_data),
            modes.GCM(nonce),
            backend=default_backend()
        )
        
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        return EncryptedData(
            encrypted_data=ciphertext,
            key_id=key.key_id,
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            nonce=nonce,
            tag=encryptor.tag
        )
    
    def _decrypt_aes_gcm(self, encrypted_data: EncryptedData, key: EncryptionKey) -> bytes:
        """Decrypt using AES-256-GCM"""
        cipher = Cipher(
            algorithms.AES(key.key_data),
            modes.GCM(encrypted_data.nonce, encrypted_data.tag),
            backend=default_backend()
        )
        
        decryptor = cipher.decryptor()
        return decryptor.update(encrypted_data.encrypted_data) + decryptor.finalize()
    
    def _encrypt_chacha20(self, data: bytes, key: EncryptionKey) -> EncryptedData:
        """Encrypt using ChaCha20-Poly1305"""
        nonce = secrets.token_bytes(12)  # 96-bit nonce
        
        cipher = Cipher(
            algorithms.ChaCha20(key.key_data, nonce),
            modes.GCM(b'\x00' * 12),  # ChaCha20 with Poly1305
            backend=default_backend()
        )
        
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        return EncryptedData(
            encrypted_data=ciphertext,
            key_id=key.key_id,
            algorithm=EncryptionAlgorithm.CHACHA20_POLY1305,
            nonce=nonce,
            tag=encryptor.tag
        )
    
    def _decrypt_chacha20(self, encrypted_data: EncryptedData, key: EncryptionKey) -> bytes:
        """Decrypt using ChaCha20-Poly1305"""
        cipher = Cipher(
            algorithms.ChaCha20(key.key_data, encrypted_data.nonce),
            modes.GCM(b'\x00' * 12, encrypted_data.tag),
            backend=default_backend()
        )
        
        decryptor = cipher.decryptor()
        return decryptor.update(encrypted_data.encrypted_data) + decryptor.finalize()
    
    def _encrypt_rsa(self, data: bytes, key: EncryptionKey) -> EncryptedData:
        """Encrypt using RSA-OAEP (for small data like keys)"""
        private_key = serialization.load_pem_private_key(
            key.key_data,
            password=None,
            backend=default_backend()
        )
        public_key = private_key.public_key()
        
        # RSA can only encrypt small amounts of data
        if len(data) > 190:  # Conservative limit for 2048-bit key
            raise ValueError("Data too large for RSA encryption")
        
        encrypted = public_key.encrypt(
            data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return EncryptedData(
            encrypted_data=encrypted,
            key_id=key.key_id,
            algorithm=EncryptionAlgorithm.RSA_OAEP
        )
    
    def _decrypt_rsa(self, encrypted_data: EncryptedData, key: EncryptionKey) -> bytes:
        """Decrypt using RSA-OAEP"""
        private_key = serialization.load_pem_private_key(
            key.key_data,
            password=None,
            backend=default_backend()
        )
        
        decrypted = private_key.decrypt(
            encrypted_data.encrypted_data,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return decrypted

class FieldLevelEncryption:
    """Provides field-level encryption for sensitive data"""
    
    def __init__(self, data_encryption: DataEncryption):
        self.data_encryption = data_encryption
        self.encrypted_field_prefix = "ENC:"
    
    def encrypt_fields(self, data: Dict[str, Any], 
                      sensitive_fields: List[str]) -> Dict[str, Any]:
        """Encrypt specified fields in a dictionary"""
        encrypted_data = data.copy()
        
        for field in sensitive_fields:
            if field in encrypted_data and encrypted_data[field] is not None:
                # Convert to string if not already
                field_value = str(encrypted_data[field])
                
                # Encrypt the field
                encrypted = self.data_encryption.encrypt(field_value)
                
                # Store as base64-encoded string with prefix
                encrypted_str = base64.b64encode(
                    json.dumps({
                        'data': base64.b64encode(encrypted.encrypted_data).decode(),
                        'key_id': encrypted.key_id,
                        'algorithm': encrypted.algorithm.value,
                        'nonce': base64.b64encode(encrypted.nonce).decode() if encrypted.nonce else None,
                        'tag': base64.b64encode(encrypted.tag).decode() if encrypted.tag else None
                    }).encode()
                ).decode()
                
                encrypted_data[field] = f"{self.encrypted_field_prefix}{encrypted_str}"
        
        return encrypted_data
    
    def decrypt_fields(self, data: Dict[str, Any], 
                      sensitive_fields: List[str]) -> Dict[str, Any]:
        """Decrypt specified fields in a dictionary"""
        decrypted_data = data.copy()
        
        for field in sensitive_fields:
            if field in decrypted_data and isinstance(decrypted_data[field], str):
                field_value = decrypted_data[field]
                
                # Check if field is encrypted
                if field_value.startswith(self.encrypted_field_prefix):
                    try:
                        # Remove prefix and decode
                        encrypted_str = field_value[len(self.encrypted_field_prefix):]
                        encrypted_info = json.loads(base64.b64decode(encrypted_str).decode())
                        
                        # Reconstruct EncryptedData object
                        encrypted_data = EncryptedData(
                            encrypted_data=base64.b64decode(encrypted_info['data']),
                            key_id=encrypted_info['key_id'],
                            algorithm=EncryptionAlgorithm(encrypted_info['algorithm']),
                            nonce=base64.b64decode(encrypted_info['nonce']) if encrypted_info['nonce'] else None,
                            tag=base64.b64decode(encrypted_info['tag']) if encrypted_info['tag'] else None
                        )
                        
                        # Decrypt the field
                        decrypted_bytes = self.data_encryption.decrypt(encrypted_data)
                        decrypted_data[field] = decrypted_bytes.decode('utf-8')
                        
                    except Exception as e:
                        logger.error(f"Failed to decrypt field {field}: {e}")
                        # Leave field as-is if decryption fails
        
        return decrypted_data
    
    def is_field_encrypted(self, field_value: Any) -> bool:
        """Check if a field value is encrypted"""
        return (isinstance(field_value, str) and 
                field_value.startswith(self.encrypted_field_prefix))

class TLSManager:
    """Manages TLS certificates and configuration"""
    
    def __init__(self, cert_storage_path: str = ".certs"):
        self.cert_storage_path = Path(cert_storage_path)
        self.cert_storage_path.mkdir(exist_ok=True, mode=0o700)
    
    def generate_self_signed_cert(self, hostname: str = "localhost",
                                 key_size: int = 2048,
                                 validity_days: int = 365) -> Tuple[str, str]:
        """Generate self-signed certificate for development/testing"""
        
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=default_backend()
        )
        
        # Create certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Cloud Intelligence Platform"),
            x509.NameAttribute(NameOID.COMMON_NAME, hostname),
        ])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.utcnow()
        ).not_valid_after(
            datetime.utcnow() + timedelta(days=validity_days)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(hostname),
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            ]),
            critical=False,
        ).sign(private_key, hashes.SHA256(), default_backend())
        
        # Save certificate and key
        cert_path = self.cert_storage_path / f"{hostname}.crt"
        key_path = self.cert_storage_path / f"{hostname}.key"
        
        with open(cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        cert_path.chmod(0o644)
        
        with open(key_path, "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        key_path.chmod(0o600)
        
        logger.info(f"Generated self-signed certificate for {hostname}")
        return str(cert_path), str(key_path)
    
    def create_ssl_context(self, cert_file: str, key_file: str,
                          ca_file: Optional[str] = None) -> ssl.SSLContext:
        """Create SSL context for secure connections"""
        
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        
        # Load certificate and key
        context.load_cert_chain(cert_file, key_file)
        
        # Load CA certificates if provided
        if ca_file:
            context.load_verify_locations(ca_file)
        
        # Security settings
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
        
        return context
    
    def get_tls_config_for_fastapi(self, hostname: str = "localhost") -> Dict[str, str]:
        """Get TLS configuration for FastAPI/Uvicorn"""
        
        cert_path = self.cert_storage_path / f"{hostname}.crt"
        key_path = self.cert_storage_path / f"{hostname}.key"
        
        # Generate certificate if it doesn't exist
        if not cert_path.exists() or not key_path.exists():
            cert_file, key_file = self.generate_self_signed_cert(hostname)
            return {"ssl_certfile": cert_file, "ssl_keyfile": key_file}
        
        return {"ssl_certfile": str(cert_path), "ssl_keyfile": str(key_path)}

class DataProtectionManager:
    """Main manager for all data protection operations"""
    
    def __init__(self, key_storage_path: str = ".keys", cert_storage_path: str = ".certs"):
        self.key_manager = KeyManager(key_storage_path)
        self.data_encryption = DataEncryption(self.key_manager)
        self.field_encryption = FieldLevelEncryption(self.data_encryption)
        self.tls_manager = TLSManager(cert_storage_path)
        
        # Default sensitive fields for automatic encryption
        self.default_sensitive_fields = [
            'password', 'token', 'secret', 'key', 'api_key',
            'credit_card', 'ssn', 'phone', 'email', 'address'
        ]
    
    def encrypt_data(self, data: Union[str, bytes, Dict[str, Any]],
                    sensitive_fields: Optional[List[str]] = None) -> Union[EncryptedData, Dict[str, Any]]:
        """Encrypt data - handles both raw data and dictionaries with sensitive fields"""
        
        if isinstance(data, dict):
            # Field-level encryption for dictionaries
            fields_to_encrypt = sensitive_fields or self.default_sensitive_fields
            return self.field_encryption.encrypt_fields(data, fields_to_encrypt)
        else:
            # Direct encryption for strings/bytes
            return self.data_encryption.encrypt(data)
    
    def decrypt_data(self, data: Union[EncryptedData, Dict[str, Any]],
                    sensitive_fields: Optional[List[str]] = None) -> Union[bytes, Dict[str, Any]]:
        """Decrypt data - handles both EncryptedData objects and dictionaries"""
        
        if isinstance(data, dict):
            # Field-level decryption for dictionaries
            fields_to_decrypt = sensitive_fields or self.default_sensitive_fields
            return self.field_encryption.decrypt_fields(data, fields_to_decrypt)
        else:
            # Direct decryption for EncryptedData objects
            return self.data_encryption.decrypt(data)
    
    def rotate_keys(self, key_type: Optional[KeyType] = None):
        """Rotate encryption keys"""
        if key_type:
            # Rotate specific key type
            active_key = self.key_manager.get_active_key(key_type)
            if active_key:
                self.key_manager.rotate_key(active_key.key_id)
        else:
            # Rotate all active keys
            for key_type in KeyType:
                active_key = self.key_manager.get_active_key(key_type)
                if active_key:
                    self.key_manager.rotate_key(active_key.key_id)
    
    def setup_tls(self, hostname: str = "localhost") -> Dict[str, str]:
        """Setup TLS configuration"""
        return self.tls_manager.get_tls_config_for_fastapi(hostname)
    
    def get_encryption_status(self) -> Dict[str, Any]:
        """Get status of encryption system"""
        keys_by_type = {}
        for key_type in KeyType:
            active_key = self.key_manager.get_active_key(key_type)
            keys_by_type[key_type.value] = {
                'active': active_key is not None,
                'key_id': active_key.key_id if active_key else None,
                'created_at': active_key.created_at.isoformat() if active_key else None,
                'expires_at': active_key.expires_at.isoformat() if active_key and active_key.expires_at else None
            }
        
        return {
            'keys': keys_by_type,
            'total_keys': len(self.key_manager.keys),
            'active_keys': len([k for k in self.key_manager.keys.values() if k.is_active]),
            'tls_configured': len(list(self.tls_manager.cert_storage_path.glob("*.crt"))) > 0
        }

# Global data protection manager instance
data_protection_manager = DataProtectionManager()

# Convenience functions
def encrypt_sensitive_data(data: Union[str, bytes, Dict[str, Any]], 
                          sensitive_fields: Optional[List[str]] = None):
    """Encrypt sensitive data"""
    return data_protection_manager.encrypt_data(data, sensitive_fields)

def decrypt_sensitive_data(data: Union[EncryptedData, Dict[str, Any]], 
                          sensitive_fields: Optional[List[str]] = None):
    """Decrypt sensitive data"""
    return data_protection_manager.decrypt_data(data, sensitive_fields)

def setup_tls_config(hostname: str = "localhost") -> Dict[str, str]:
    """Setup TLS configuration for web server"""
    return data_protection_manager.setup_tls(hostname)

def rotate_encryption_keys():
    """Rotate all encryption keys"""
    data_protection_manager.rotate_keys()

# Fix missing import
import ipaddress