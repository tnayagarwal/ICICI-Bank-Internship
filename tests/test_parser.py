"""
Tests for the resume/document parsing utilities used during
candidate intake in the HR assistant pipeline.
"""

import re
import pytest


def extract_email(text: str) -> str:
    """Simple email extraction regex."""
    match = re.search(r'[\w.+-]+@[\w-]+\.[\w.]+', text)
    return match.group() if match else ""


def test_email_extraction():
    """Verify email extraction from raw resume text."""
    text = "Candidate: John Doe. Contact: johndoe@example.com. Skills: Python."
    assert extract_email(text) == "johndoe@example.com"


def test_email_missing_returns_empty():
    """If no email is present, return empty string (not None)."""
    assert extract_email("No contact info here.") == ""


def test_pdf_malformed_input_guard():
    """Malformed or binary input should not crash the parser."""
    try:
        result = extract_email("\x00\xff\xfe invalid bytes")
        assert isinstance(result, str)
    except Exception as e:
        pytest.fail(f"Parser crashed on malformed input: {e}")
