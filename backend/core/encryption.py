"""
Encryption service for secure credential storage
"""

import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import structlog

logger = structlog.get_logger(__name__)

class EncryptionService:
    """Service for encrypting/decrypting sensitive data like cloud credentials"""
    
    def __init__(self, encryption_key: str = None):
        """Initialize encryption service with key"""
        if encryption_key:
            self.key = encryption_key.encode()
        else:
            # Get from environment or generate
            env_key = os.getenv("ENCRYPTION_KEY")
            if env_key:
                self.key = env_key.encode()
            else:
                # Generate a new key (in production, this should be stored securely)
                self.key = Fernet.generate_key()
                logger.warning("Generated new encryption key - store this securely!")
        
        # Derive Fernet key from the provided key
        if len(self.key) != 44:  # Fernet key length
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'finops_salt_2024',  # In production, use random salt per key
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.key))
            self.fernet = Fernet(key)
        else:
            self.fernet = Fernet(self.key)
    
    def encrypt(self, data: str) -> str:
        """Encrypt string data"""
        try:
            encrypted_data = self.fernet.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error("Encryption failed", error=str(e))
            raise ValueError(f"Failed to encrypt data: {str(e)}")
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt string data"""
        try:
            decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self.fernet.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            logger.error("Decryption failed", error=str(e))
            raise ValueError(f"Failed to decrypt data: {str(e)}")
    
    def encrypt_dict(self, data: dict) -> str:
        """Encrypt dictionary as JSON string"""
        import json
        json_data = json.dumps(data)
        return self.encrypt(json_data)
    
    def decrypt_dict(self, encrypted_data: str) -> dict:
        """Decrypt to dictionary"""
        import json
        json_data = self.decrypt(encrypted_data)
        return json.loads(json_data)

# Global encryption service instance
encryption_service = EncryptionService()