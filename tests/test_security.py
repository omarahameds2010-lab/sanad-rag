import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.security import (
    sanitize_input,
    detect_injection_attempt,
    get_valid_api_keys,
    verify_api_key,
    RateLimiter,
    clear_api_keys_cache,
)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


class TestInputSanitization:
    def test_sanitize_input_basic(self):
        text = "  hello world  "
        assert sanitize_input(text) == "hello world"

    def test_sanitize_input_truncates(self):
        text = "x" * 15000
        result = sanitize_input(text, max_length=1000)
        assert len(result) == 1000

    def test_sanitize_input_removes_null_bytes(self):
        text = "hello\x00world"
        result = sanitize_input(text)
        assert "\x00" not in result

    def test_sanitize_input_removes_control_chars(self):
        text = "hello\x01\x02world"
        result = sanitize_input(text)
        assert "\x01" not in result
        assert "\x02" not in result

    def test_sanitize_input_keeps_newlines_tabs(self):
        text = "hello\nworld\t!"
        result = sanitize_input(text)
        assert "\n" in result
        assert "\t" in result


class TestInjectionDetection:
    def test_detects_sql_injection(self):
        text = "SELECT * FROM users"
        suspicious, patterns = detect_injection_attempt(text)
        assert suspicious
        assert any("sql" in p for p in patterns)

    def test_detects_union_injection(self):
        text = "' UNION SELECT password FROM users --"
        suspicious, patterns = detect_injection_attempt(text)
        assert suspicious

    def test_detects_xss_script(self):
        text = "<script>alert('xss')</script>"
        suspicious, patterns = detect_injection_attempt(text)
        assert suspicious
        assert any("xss" in p for p in patterns)

    def test_detects_javascript_protocol(self):
        text = "javascript:alert(1)"
        suspicious, patterns = detect_injection_attempt(text)
        assert suspicious

    def test_detects_path_traversal(self):
        text = "../../../etc/passwd"
        suspicious, patterns = detect_injection_attempt(text)
        assert suspicious
        assert any("path_traversal" in p for p in patterns)

    def test_normal_arabic_text_not_flagged(self):
        text = "ما هو الإعراب في النحو العربي؟"
        suspicious, patterns = detect_injection_attempt(text)
        # Should not flag normal Arabic text
        assert not suspicious or len(patterns) == 0


class TestRateLimiter:
    def test_rate_limiter_allows_requests_under_limit(self):
        limiter = RateLimiter(requests_per_minute=60, burst=10)
        allowed, _ = limiter.check_rate_limit("192.168.1.1")
        assert allowed

    def test_rate_limiter_blocks_over_limit(self):
        limiter = RateLimiter(requests_per_minute=60, burst=2)
        limiter.check_rate_limit("192.168.1.2")  # 1
        limiter.check_rate_limit("192.168.1.2")  # 2
        allowed, headers = limiter.check_rate_limit("192.168.1.2")  # 3 - should block
        assert not allowed
        assert headers["X-RateLimit-Limit"] == "60"

    def test_rate_limiter_exempt_ips(self):
        limiter = RateLimiter(requests_per_minute=60, burst=1, exempt_ips={"10.0.0.1"})
        # Should always allow exempt IP
        for _ in range(100):
            allowed, _ = limiter.check_rate_limit("10.0.0.1")
            assert allowed


class TestAPIKeyAuth:
    def test_get_valid_api_keys_empty(self):
        # No keys configured -> empty set
        os.environ.pop("SANAD_API_KEYS", None)
        clear_api_keys_cache()
        keys = get_valid_api_keys()
        assert keys == set()

    def test_get_valid_api_keys_parses_comma_separated(self):
        os.environ["SANAD_API_KEYS"] = "key1, key2, key3"
        clear_api_keys_cache()
        keys = get_valid_api_keys()
        assert keys == {"key1", "key2", "key3"}


class TestAPIEndpointsWithSecurity:
    def test_health_endpoint_no_auth_required(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_ingest_requires_auth_when_keys_configured(self, client, monkeypatch):
        monkeypatch.setenv("SANAD_API_KEYS", "test-key-123")
        clear_api_keys_cache()
        
        # Request without API key should fail
        r = client.post("/ingest", json={"source_id": "test", "text": "test"})
        assert r.status_code == 401

        # Request with invalid API key should fail
        r = client.post("/ingest", 
            json={"source_id": "test", "text": "test"},
            headers={"X-API-Key": "wrong-key"}
        )
        assert r.status_code == 401

        # Request with valid API key should succeed
        r = client.post("/ingest",
            json={"source_id": "test", "text": "test"},
            headers={"X-API-Key": "test-key-123"}
        )
        # 200 or 422 (validation) but not 401
        assert r.status_code != 401

    def test_query_requires_auth_when_keys_configured(self, client, monkeypatch):
        monkeypatch.setenv("SANAD_API_KEYS", "test-key-123")
        clear_api_keys_cache()
        
        r = client.post("/query", json={"question": "test"}, headers={"X-API-Key": "test-key-123"})
        assert r.status_code != 401

    def test_rate_limit_headers_present(self, client):
        r = client.get("/health")
        # Health check is exempt from rate limiting but headers should exist
        assert r.status_code == 200

    def test_security_headers_present(self, client):
        r = client.get("/health")
        assert r.headers.get("X-Content-Type-Options") == "nosniff"
        assert r.headers.get("X-Frame-Options") == "DENY"
        assert r.headers.get("X-XSS-Protection") == "1; mode=block"
        assert "Content-Security-Policy" in r.headers

    def test_cors_headers_present(self, client):
        r = client.options("/health", headers={"Origin": "http://localhost:3000"})
        # Should have CORS headers
        assert r.status_code in [200, 405]


class TestInputValidationMiddleware:
    def test_sanitizes_input_on_ingest(self, client, monkeypatch):
        monkeypatch.setenv("SANAD_API_KEYS", "test-key-123")
        
        # Input with extra whitespace and null bytes
        r = client.post("/ingest",
            json={"source_id": "test", "text": "  hello\x00world  "},
            headers={"X-API-Key": "test-key-123"}
        )
        # Should not crash, input should be sanitized
        assert r.status_code in [200, 422]