"""
YMERA Enterprise - Security Configuration
Production-Ready Security Management - v4.0

File: backend/app/CORE_CONFIGURATION/config_security.py
FIXED: Line ~894 - Import path corrected from .settings to .config_settings
"""

# ===============================================================================
# STANDARD IMPORTS SECTION
# ===============================================================================

import hashlib
import hmac
import os
import secrets
import time
from datetime import datetime, timedelta, timezone
from functools import lru_cache, wraps
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

# Third-party imports
import bcrypt
import jwt
import structlog
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from pydantic import BaseModel, Field, validator

# ===============================================================================
# LOGGING CONFIGURATION
# ===============================================================================

logger = structlog.get_logger("ymera.security")

# ===============================================================================
# CONSTANTS & CONFIGURATION
# ===============================================================================

DEFAULT_ALGORITHM = "HS256"
DEFAULT_ACCESS_TOKEN_EXPIRE_MINUTES = 30
DEFAULT_REFRESH_TOKEN_EXPIRE_DAYS = 7

MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 128
PASSWORD_ROUNDS = 12

DEFAULT_RATE_LIMIT = 1000
DEFAULT_RATE_WINDOW = 3600

SALT_LENGTH = 32
KEY_LENGTH = 32

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'self'",
    "Referrer-Policy": "strict-origin-when-cross-origin"
}

# ===============================================================================
# ENUMS & DATA MODELS
# ===============================================================================

class TokenType(str, Enum):
    """Token type enumeration"""
    ACCESS = "access"
    REFRESH = "refresh"
    RESET = "reset"
    VERIFICATION = "verification"

class PermissionLevel(str, Enum):
    """Permission level enumeration"""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

@dataclass
class SecurityConfig:
    """Comprehensive security configuration"""
    
    # JWT Configuration
    secret_key: str
    jwt_secret_key: str
    jwt_algorithm: str = DEFAULT_ALGORITHM
    access_token_expire_minutes: int = DEFAULT_ACCESS_TOKEN_EXPIRE_MINUTES
    refresh_token_expire_days: int = DEFAULT_REFRESH_TOKEN_EXPIRE_DAYS
    
    # Password Configuration
    password_min_length: int = MIN_PASSWORD_LENGTH
    password_max_length: int = MAX_PASSWORD_LENGTH
    password_rounds: int = PASSWORD_ROUNDS
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_numbers: bool = True
    require_special_chars: bool = True
    
    # Rate Limiting
    rate_limit_requests: int = DEFAULT_RATE_LIMIT
    rate_limit_window: int = DEFAULT_RATE_WINDOW
    rate_limit_enabled: bool = True
    
    # Session Configuration
    session_timeout_minutes: int = 480
    max_concurrent_sessions: int = 5
    
    # Encryption
    encryption_key: Optional[str] = None
    
    # Security Headers
    security_headers_enabled: bool = True
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    
    extra_options: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and initialize security configuration"""
        self._validate_keys()
        self._validate_password_policy()
        self._setup_encryption()
        self._validate_rate_limiting()
    
    def _validate_keys(self) -> None:
        """Validate secret keys"""
        if not self.secret_key or len(self.secret_key) < 32:
            raise ValueError("Secret key must be at least 32 characters long")
        
        if not self.jwt_secret_key or len(self.jwt_secret_key) < 32:
            raise ValueError("JWT secret key must be at least 32 characters long")
    
    def _validate_password_policy(self) -> None:
        """Validate password policy settings"""
        if self.password_min_length < 8:
            logger.warning("Password minimum length is less than recommended 8 characters")
        
        if self.password_max_length > 256:
            logger.warning("Password maximum length is very high")
    
    def _setup_encryption(self) -> None:
        """Setup encryption configuration"""
        if not self.encryption_key:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=KEY_LENGTH,
                salt=self.secret_key[:SALT_LENGTH].encode(),
                iterations=100000,
            )
            key = kdf.derive(self.secret_key.encode())
            self.encryption_key = Fernet.generate_key().decode()
    
    def _validate_rate_limiting(self) -> None:
        """Validate rate limiting configuration"""
        if self.rate_limit_requests < 1:
            raise ValueError("Rate limit requests must be at least 1")
        
        if self.rate_limit_window < 60:
            logger.warning("Rate limit window is very short")

# ===============================================================================
# AUTHENTICATION MODELS
# ===============================================================================

class TokenPayload(BaseModel):
    """JWT token payload model"""
    sub: str = Field(..., description="Subject (user ID)")
    exp: int = Field(..., description="Expiration timestamp")
    iat: int = Field(..., description="Issued at timestamp")
    type: TokenType = Field(..., description="Token type")
    permissions: List[str] = Field(default_factory=list, description="User permissions")
    session_id: Optional[str] = Field(None, description="Session identifier")
    
    class Config:
        use_enum_values = True

class AuthenticationResponse(BaseModel):
    """Authentication response model"""
    access_token: str = Field(..., description="Access token")
    refresh_token: str = Field(..., description="Refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")
    user_id: str = Field(..., description="User identifier")
    permissions: List[str] = Field(default_factory=list, description="User permissions")

class User(BaseModel):
    """User model for authentication"""
    id: str = Field(..., description="User identifier")
    email: str = Field(..., description="User email")
    username: Optional[str] = Field(None, description="Username")
    permissions: List[str] = Field(default_factory=list, description="User permissions")
    is_active: bool = Field(default=True, description="User active status")
    is_verified: bool = Field(default=False, description="User verification status")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")

# ===============================================================================
# SECURITY MANAGERS
# ===============================================================================

class PasswordManager:
    """Production-ready password management"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logger.bind(component="PasswordManager")
        
        self.pwd_context = CryptContext(
            schemes=["bcrypt"],
            deprecated="auto",
            bcrypt__rounds=config.password_rounds
        )
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        self.validate_password_strength(password)
        hashed = self.pwd_context.hash(password)
        self.logger.debug("Password hashed successfully")
        return hashed
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        try:
            is_valid = self.pwd_context.verify(plain_password, hashed_password)
            if is_valid:
                self.logger.debug("Password verification successful")
            else:
                self.logger.debug("Password verification failed")
            return is_valid
        except Exception as e:
            self.logger.error("Password verification error", error=str(e))
            return False
    
    def validate_password_strength(self, password: str) -> None:
        """Validate password strength"""
        errors = []
        
        if len(password) < self.config.password_min_length:
            errors.append(f"Password must be at least {self.config.password_min_length} characters")
        
        if len(password) > self.config.password_max_length:
            errors.append(f"Password must not exceed {self.config.password_max_length} characters")
        
        if self.config.require_uppercase and not any(c.isupper() for c in password):
            errors.append("Password must contain uppercase letter")
        
        if self.config.require_lowercase and not any(c.islower() for c in password):
            errors.append("Password must contain lowercase letter")
        
        if self.config.require_numbers and not any(c.isdigit() for c in password):
            errors.append("Password must contain number")
        
        if self.config.require_special_chars and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            errors.append("Password must contain special character")
        
        if errors:
            raise ValueError("; ".join(errors))
    
    def generate_secure_password(self, length: int = 16) -> str:
        """Generate cryptographically secure password"""
        if length < 12:
            length = 12
        
        import string
        
        chars = (
            string.ascii_lowercase +
            string.ascii_uppercase +
            string.digits +
            "!@#$%^&*()_+-="
        )
        
        password = [
            secrets.choice(string.ascii_lowercase),
            secrets.choice(string.ascii_uppercase),
            secrets.choice(string.digits),
            secrets.choice("!@#$%^&*()_+-=")
        ]
        
        for _ in range(length - 4):
            password.append(secrets.choice(chars))
        
        secrets.SystemRandom().shuffle(password)
        generated = ''.join(password)
        self.logger.debug("Secure password generated", length=length)
        return generated

class JWTManager:
    """Production-ready JWT token management"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logger.bind(component="JWTManager")
        self._blacklisted_tokens: set = set()
    
    def create_access_token(
        self, 
        user_id: str, 
        permissions: List[str] = None,
        session_id: Optional[str] = None,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token"""
        if permissions is None:
            permissions = []
        
        if expires_delta is None:
            expires_delta = timedelta(minutes=self.config.access_token_expire_minutes)
        
        now = datetime.now(timezone.utc)
        expire = now + expires_delta
        
        payload = TokenPayload(
            sub=user_id,
            exp=int(expire.timestamp()),
            iat=int(now.timestamp()),
            type=TokenType.ACCESS,
            permissions=permissions,
            session_id=session_id
        )
        
        token = jwt.encode(
            payload.dict(),
            self.config.jwt_secret_key,
            algorithm=self.config.jwt_algorithm
        )
        
        self.logger.debug("Access token created", user_id=user_id)
        return token
    
    def create_refresh_token(
        self, 
        user_id: str,
        session_id: Optional[str] = None,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT refresh token"""
        if expires_delta is None:
            expires_delta = timedelta(days=self.config.refresh_token_expire_days)
        
        now = datetime.now(timezone.utc)
        expire = now + expires_delta
        
        payload = TokenPayload(
            sub=user_id,
            exp=int(expire.timestamp()),
            iat=int(now.timestamp()),
            type=TokenType.REFRESH,
            permissions=[],
            session_id=session_id
        )
        
        token = jwt.encode(
            payload.dict(),
            self.config.jwt_secret_key,
            algorithm=self.config.jwt_algorithm
        )
        
        self.logger.debug("Refresh token created", user_id=user_id)
        return token
    
    def verify_token(self, token: str, token_type: TokenType = TokenType.ACCESS) -> TokenPayload:
        """Verify and decode JWT token"""
        try:
            if token in self._blacklisted_tokens:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked"
                )
            
            payload = jwt.decode(
                token,
                self.config.jwt_secret_key,
                algorithms=[self.config.jwt_algorithm]
            )
            
            token_payload = TokenPayload(**payload)
            
            if token_payload.type != token_type:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Invalid token type"
                )
            
            if token_payload.exp < int(time.time()):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has expired"
                )
            
            self.logger.debug("Token verified successfully", user_id=token_payload.sub)
            return token_payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError as e:
            self.logger.warning("Invalid token provided", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        """Create new access token from refresh token"""
        payload = self.verify_token(refresh_token, TokenType.REFRESH)
        
        new_access_token = self.create_access_token(
            user_id=payload.sub,
            permissions=payload.permissions,
            session_id=payload.session_id
        )
        
        new_refresh_token = self.create_refresh_token(
            user_id=payload.sub,
            session_id=payload.session_id
        )
        
        self._blacklisted_tokens.add(refresh_token)
        
        return {
            "access_token": new_access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer",
            "expires_in": self.config.access_token_expire_minutes * 60
        }
    
    def revoke_token(self, token: str) -> None:
        """Revoke token"""
        self._blacklisted_tokens.add(token)
        self.logger.debug("Token revoked")

class EncryptionManager:
    """Production-ready encryption manager"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logger.bind(component="EncryptionManager")
        
        if config.encryption_key:
            self.fernet = Fernet(config.encryption_key.encode())
        else:
            raise ValueError("Encryption key not provided")
    
    def encrypt(self, data: str) -> str:
        """Encrypt string data"""
        try:
            encrypted_data = self.fernet.encrypt(data.encode())
            self.logger.debug("Data encrypted successfully")
            return encrypted_data.decode()
        except Exception as e:
            self.logger.error("Encryption failed", error=str(e))
            raise
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt string data"""
        try:
            decrypted_data = self.fernet.decrypt(encrypted_data.encode())
            self.logger.debug("Data decrypted successfully")
            return decrypted_data.decode()
        except InvalidToken:
            self.logger.error("Decryption failed - invalid token")
            raise
        except Exception as e:
            self.logger.error("Decryption failed", error=str(e))
            raise

class RateLimiter:
    """Production-ready rate limiting"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logger.bind(component="RateLimiter")
        self._requests: Dict[str, List[float]] = {}
        self._cleanup_interval = 300
        self._last_cleanup = time.time()
    
    def is_allowed(self, identifier: str, limit: Optional[int] = None, window: Optional[int] = None) -> bool:
        """Check if request is allowed"""
        if not self.config.rate_limit_enabled:
            return True
        
        current_time = time.time()
        limit = limit or self.config.rate_limit_requests
        window = window or self.config.rate_limit_window
        
        if current_time - self._last_cleanup > self._cleanup_interval:
            self._cleanup_old_entries(current_time)
            self._last_cleanup = current_time
        
        if identifier not in self._requests:
            self._requests[identifier] = []
        
        request_times = self._requests[identifier]
        cutoff_time = current_time - window
        request_times[:] = [t for t in request_times if t > cutoff_time]
        
        if len(request_times) >= limit:
            self.logger.debug("Rate limit exceeded", identifier=identifier)
            return False
        
        request_times.append(current_time)
        return True
    
    def _cleanup_old_entries(self, current_time: float) -> None:
        """Remove old entries"""
        cutoff_time = current_time - self.config.rate_limit_window
        
        for identifier in list(self._requests.keys()):
            request_times = self._requests[identifier]
            request_times[:] = [t for t in request_times if t > cutoff_time]
            
            if not request_times:
                del self._requests[identifier]

class SecurityDependencies:
    """FastAPI security dependencies"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.jwt_manager = JWTManager(config)
        self.rate_limiter = RateLimiter(config)
        self.logger = logger.bind(component="SecurityDependencies")
        self.security_scheme = HTTPBearer(auto_error=False)
    
    async def get_current_user(
        self, 
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
    ) -> User:
        """Get current authenticated user"""
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication credentials required",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        token_payload = self.jwt_manager.verify_token(credentials.credentials)
        
        user = User(
            id=token_payload.sub,
            email=f"user_{token_payload.sub}@example.com",
            permissions=token_payload.permissions,
            is_active=True,
            is_verified=True
        )
        
        self.logger.debug("User authenticated", user_id=user.id)
        return user

class SecurityMiddleware:
    """Security middleware"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logger.bind(component="SecurityMiddleware")
    
    async def __call__(self, request: Request, call_next):
        """Process request with security headers"""
        response = await call_next(request)
        
        if self.config.security_headers_enabled:
            for header, value in SECURITY_HEADERS.items():
                response.headers[header] = value
        
        return response

# ===============================================================================
# UTILITY FUNCTIONS
# ===============================================================================

def generate_session_id() -> str:
    """Generate session ID"""
    return secrets.token_urlsafe(32)

def constant_time_compare(a: str, b: str) -> bool:
    """Constant time string comparison"""
    return hmac.compare_digest(a.encode(), b.encode())

def hash_api_key(api_key: str, salt: Optional[str] = None) -> str:
    """Hash API key"""
    if salt is None:
        salt = secrets.token_hex(16)
    
    key_hash = hashlib.pbkdf2_hmac('sha256', api_key.encode(), salt.encode(), 100000)
    return f"{salt}:{key_hash.hex()}"

def verify_api_key(api_key: str, stored_hash: str) -> bool:
    """Verify API key"""
    try:
        salt, key_hash = stored_hash.split(':', 1)
        expected_hash = hashlib.pbkdf2_hmac('sha256', api_key.encode(), salt.encode(), 100000)
        return constant_time_compare(key_hash, expected_hash.hex())
    except (ValueError, AttributeError):
        return False

# ===============================================================================
# CONFIGURATION FACTORY - LINE ~894 FIXED
# ===============================================================================

@lru_cache()
def get_security_config() -> SecurityConfig:
    """
    Get security configuration with caching.
    
    FIXED: Changed import from .settings to .config_settings (Line ~894)
    
    Returns:
        SecurityConfig: Configured security settings
    """
    # ✅ FIXED: Import path corrected
    from .config_settings import get_settings
    
    settings = get_settings()
    
    config = SecurityConfig(
        secret_key=settings.SECRET_KEY,
        jwt_secret_key=settings.JWT_SECRET_KEY,
        jwt_algorithm=settings.JWT_ALGORITHM,
        access_token_expire_minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
        refresh_token_expire_days=settings.REFRESH_TOKEN_EXPIRE_DAYS,
        rate_limit_requests=settings.RATE_LIMIT_REQUESTS,
        rate_limit_window=settings.RATE_LIMIT_WINDOW,
        cors_origins=settings.CORS_ORIGINS
    )
    
    logger.info(
        "Security configuration initialized",
        jwt_algorithm=config.jwt_algorithm,
        access_token_expire_minutes=config.access_token_expire_minutes,
        rate_limit_enabled=config.rate_limit_enabled
    )
    
    return config

# ===============================================================================
# INITIALIZATION FUNCTIONS
# ===============================================================================

def initialize_security() -> Dict[str, Any]:
    """Initialize security components"""
    config = get_security_config()
    
    managers = {
        "password_manager": PasswordManager(config),
        "jwt_manager": JWTManager(config),
        "encryption_manager": EncryptionManager(config),
        "rate_limiter": RateLimiter(config),
        "dependencies": SecurityDependencies(config),
        "middleware": SecurityMiddleware(config)
    }
    
    logger.info("Security components initialized successfully")
    return managers

# ===============================================================================
# EXPORTS
# ===============================================================================

__all__ = [
    "SecurityConfig",
    "TokenType",
    "PermissionLevel",
    "TokenPayload",
    "AuthenticationResponse",
    "User",
    "PasswordManager",
    "JWTManager",
    "EncryptionManager",
    "RateLimiter",
    "SecurityDependencies",
    "SecurityMiddleware",
    "get_security_config",
    "initialize_security",
    "generate_session_id",
    "constant_time_compare",
    "hash_api_key",
    "verify_api_key"
]
