"""
Webhook Orchestration Utilities
================================
Provides retry-safe n8n webhook interceptors and latency tracking
for monitoring the agentic LLM pipeline performance.
"""

import time
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

MAX_RETRY_COUNT = 3
RETRY_BACKOFF_BASE = 2  # seconds


def with_retry(fn: Callable, *args, max_retries: int = MAX_RETRY_COUNT, **kwargs) -> Any:
    """
    Execute a callable with exponential backoff retry logic.

    Args:
        fn: The function to call.
        max_retries: Maximum number of retry attempts.
        *args, **kwargs: Arguments passed to the function.

    Returns:
        The result of the function call.

    Raises:
        TimeoutError: If all retries are exhausted.
    """
    for attempt in range(max_retries):
        try:
            start = time.monotonic()
            result = fn(*args, **kwargs)
            latency = round((time.monotonic() - start) * 1000, 2)
            logger.info("Call succeeded in %.2fms (attempt %d).", latency, attempt + 1)
            return result
        except Exception as e:
            wait = RETRY_BACKOFF_BASE ** attempt
            logger.warning("Attempt %d failed: %s. Retrying in %ds...", attempt + 1, e, wait)
            time.sleep(wait)

    raise TimeoutError(f"Function '{fn.__name__}' failed after {max_retries} retries.")


def handle_n8n_webhook(payload: dict) -> dict:
    """
    Process an incoming n8n webhook payload and route it to the orchestrator.

    Args:
        payload: Webhook data dict containing employee_id and query fields.

    Returns:
        Response dict from the LLM orchestrator.
    """
    employee_id = payload.get("employee_id")
    query = payload.get("query", "")

    if not employee_id:
        raise ValueError("Webhook payload must contain 'employee_id'.")
    if not query.strip():
        raise ValueError("Webhook payload must contain a non-empty 'query'.")

    logger.info("Webhook received from employee: %s", employee_id)

    # Dynamic import to avoid circular deps
    from agents.langgraph_orchestrator import process_dynamic_request_full
    return with_retry(process_dynamic_request_full, employee_id, query)
