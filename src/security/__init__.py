"""
Security utilities for SANAD API.

Provides:
- API Key authentication
- Rate limiting
- Input sanitization
- Security headers
- CORS configuration
"""

import os
import time
import re
from typing import Optional
from functools import lru_cache

from fastapi import Request, HTTPException, Depends
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp


# =============================================================================
# API Key Authentication
# =============================================================================

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


@lru_cache
def get_valid_api_keys() -> set[str]:
    """
    Load valid API keys from environment.
    Format: SANAD_API_KEYS=key1,key2,key3
    Returns empty set if not configured (dev mode - no auth required).
    """
    keys_str = os.environ.get("SANAD_API_KEYS", "").strip()
    if not keys_str:
        return set()
    return {k.strip() for k in keys_str.split(",") if k.strip()}


def clear_api_keys_cache():
    """Clear the API keys cache for testing."""
    get_valid_api_keys.cache_clear()


async def verify_api_key(
    request: Request,
    api_key: Optional[str] = Depends(api_key_header),
) -> Optional[str]:
    """
    Verify API key if authentication is enabled.
    Returns the API key if valid, None if auth is disabled.
    Raises 401 if auth is enabled but key is missing/invalid.
    """
    valid_keys = get_valid_api_keys()
    
    # If no keys configured, run in dev mode (no auth required)
    if not valid_keys:
        return None
    
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Provide via X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    if api_key not in valid_keys:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    return api_key


# =============================================================================
# Rate Limiting (Token Bucket)
# =============================================================================

class TokenBucket:
    """Simple in-memory token bucket for rate limiting."""
    
    def __init__(self, rate: float, burst: int):
        self.rate = rate  # tokens per second
        self.burst = burst  # max tokens
        self.tokens = float(burst)
        self.last_update = time.monotonic()
    
    def consume(self, tokens: int = 1) -> bool:
        now = time.monotonic()
        elapsed = now - self.last_update
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        self.last_update = now
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class RateLimiter:
    """Per-IP rate limiter using token bucket."""
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        burst: int = 10,
        exempt_ips: Optional[set[str]] = None,
    ):
        self.buckets: dict[str, TokenBucket] = {}
        self.requests_per_minute = requests_per_minute
        self.burst = burst
        self.exempt_ips = exempt_ips or {"127.0.0.1", "::1"}
    
    def _get_bucket(self, ip: str) -> TokenBucket:
        if ip not in self.buckets:
            self.buckets[ip] = TokenBucket(
                rate=self.requests_per_minute / 60.0,
                burst=self.burst,
            )
        return self.buckets[ip]
    
    def check_rate_limit(self, ip: str) -> tuple[bool, dict]:
        """
        Check if request is allowed.
        Returns (allowed, headers_dict).
        """
        if ip in self.exempt_ips:
            return True, {}
        
        bucket = self._get_bucket(ip)
        allowed = bucket.consume()
        
        headers = {
            "X-RateLimit-Limit": str(self.requests_per_minute),
            "X-RateLimit-Remaining": str(int(bucket.tokens)),
            "X-RateLimit-Reset": str(int(time.monotonic()) + 60),
        }
        
        return allowed, headers


# Global rate limiter instance
rate_limiter = RateLimiter(
    requests_per_minute=int(os.environ.get("SANAD_RATE_LIMIT", "60")),
    burst=int(os.environ.get("SANAD_RATE_BURST", "10")),
)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""
    
    def __init__(
        self,
        app: ASGIApp,
        limiter: Optional[RateLimiter] = None,
    ):
        super().__init__(app)
        self.limiter = limiter or rate_limiter
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/docs", "/openapi.json"]:
            return await call_next(request)
        
        client_ip = request.client.host if request.client else "unknown"
        allowed, headers = self.limiter.check_rate_limit(client_ip)
        
        if not allowed:
            return Response(
                content='{"detail": "Rate limit exceeded"}',
                status_code=429,
                media_type="application/json",
                headers=headers,
            )
        
        response = await call_next(request)
        # Add rate limit headers to response
        for k, v in headers.items():
            response.headers[k] = v
        return response


# =============================================================================
# Security Headers Middleware
# =============================================================================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )
        
        # HSTS (only if using HTTPS)
        # response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response


# =============================================================================
# Input Sanitization
# =============================================================================

# Regex patterns for dangerous inputs
SQL_INJECTION_PATTERNS = [
    r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|CREATE)\b)",
    r"(--|;|\/\*|\*\/)",
    r"(\bOR\b\s+\d+\s*=\s*\d+)",
    r"(\bAND\b\s+\d+\s*=\s*\d+)",
]

XSS_PATTERNS = [
    r"<script[^>]*>.*?</script>",
    r"javascript:",
    r"on\w+\s*=",
    r"<iframe[^>]*>",
    r"<object[^>]*>",
    r"<embed[^>]*>",
]

PATH_TRAVERSAL_PATTERNS = [
    r"\.\./",
    r"\.\.\\",
    r"%2e%2e%2f",
    r"%2e%2e%5c",
]


def sanitize_input(text: str, max_length: int = 10000) -> str:
    """
    Sanitize user input for safety.
    - Truncates to max_length
    - Removes null bytes
    - Strips control characters (except newlines/tabs)
    """
    if not text:
        return ""
    
    # Truncate
    text = text[:max_length]
    
    # Remove null bytes
    text = text.replace("\x00", "")
    
    # Remove control characters except \n, \r, \t
    text = "".join(
        char for char in text
        if ord(char) >= 32 or char in "\n\r\t"
    )
    
    return text.strip()


def detect_injection_attempt(text: str) -> tuple[bool, list[str]]:
    """
    Detect potential injection attempts.
    Returns (is_suspicious, list_of_matched_patterns).
    """
    if not text:
        return False, []
    
    matched = []
    text_lower = text.lower()
    
    for pattern in SQL_INJECTION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            matched.append(f"sql:{pattern}")
    
    for pattern in XSS_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            matched.append(f"xss:{pattern}")
    
    for pattern in PATH_TRAVERSAL_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            matched.append(f"path_traversal:{pattern}")
    
    return len(matched) > 0, matched


# =============================================================================
# CORS Configuration
# =============================================================================

def get_cors_origins() -> list[str]:
    """
    Get allowed CORS origins from environment.
    Format: SANAD_CORS_ORIGINS=http://localhost:3000,https://app.example.com
    Defaults to localhost for development.
    """
    origins_str = os.environ.get("SANAD_CORS_ORIGINS", "").strip()
    if not origins_str:
        return [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
        ]
    return [o.strip() for o in origins_str.split(",") if o.strip()]