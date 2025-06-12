# ai-service/app/core/security.py
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer()

class TokenValidator:
    """JWT Token validation"""
    
    def __init__(self):
        self.settings = get_settings()
        self.algorithm = self.settings.jwt_algorithm
        self.secret = self.settings.jwt_secret
    
    def decode_token(self, token: str) -> Dict[str, Any]:
        """Decode and validate JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.secret,
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def verify_token(self, credentials: HTTPAuthorizationCredentials) -> Dict[str, Any]:
        """Verify bearer token"""
        token = credentials.credentials
        return self.decode_token(token)

# Global token validator instance
token_validator = TokenValidator()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """Get current user from JWT token"""
    payload = token_validator.verify_token(credentials)
    
    # Extract user information
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload"
        )
    
    return {
        "id": user_id,
        "email": payload.get("email"),
        "roles": payload.get("roles", []),
        "permissions": payload.get("permissions", [])
    }

async def require_permission(permission: str):
    """Dependency to require specific permission"""
    async def permission_checker(
        current_user: Dict[str, Any] = Depends(get_current_user)
    ):
        permissions = current_user.get("permissions", [])
        if permission not in permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required"
            )
        return current_user
    
    return permission_checker

async def require_role(role: str):
    """Dependency to require specific role"""
    async def role_checker(
        current_user: Dict[str, Any] = Depends(get_current_user)
    ):
        roles = current_user.get("roles", [])
        if role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role}' required"
            )
        return current_user
    
    return role_checker

def create_access_token(user_id: str, email: str, 
                       roles: List[str] = None, 
                       permissions: List[str] = None,
                       expires_delta: Optional[timedelta] = None) -> str:
    """Create a new access token (for testing/internal use)"""
    settings = get_settings()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=24)
    
    payload = {
        "sub": user_id,
        "email": email,
        "roles": roles or [],
        "permissions": permissions or [],
        "exp": expire,
        "iat": datetime.utcnow()
    }
    
    token = jwt.encode(
        payload,
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm
    )
    
    return token