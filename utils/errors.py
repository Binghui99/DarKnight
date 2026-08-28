"""
utils/errors.py
Custom exceptions and handle errors for Dark Knight.
"""

CONNECTION_ERROR_KEY = "connection_error"


class ConnectionFailure(RuntimeError):
    """
    Raised when every retry attempt for an LLM API call fails with a
    network / connection / timeout error.

    Catching this in pipeline.py allows the orchestrator to skip recording
    the sample so that --resume will naturally rerun it.
    """
    pass


def is_connection_error(exc: Exception) -> bool:
    """
    Return True if *exc* represents a network issue that is worth
    retrying / skipping rather than logging as a permanent failure.

    Covers:
    - openai.APIConnectionError, openai.APITimeoutError (SDK v1)
    - built-in ConnectionError, TimeoutError, OSError (ECONNREFUSED etc.)
    - httpx transport exceptions that bubble up as generic exceptions
    - keyword matching on the exception message as a last resort
    """
    # OpenAI SDK v1 typed exceptions
    try:
        import openai
        if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
            return True
    except (ImportError, AttributeError):
        pass

    # httpx exceptions (used internally by the OpenAI SDK)
    try:
        import httpx
        if isinstance(exc, (httpx.ConnectError, httpx.ReadTimeout,
                             httpx.WriteTimeout, httpx.PoolTimeout)):
            return True
    except (ImportError, AttributeError):
        pass

    # Built-in Python network exceptions
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True

    # Keyword fallback for wrapped / string-only exceptions
    msg = str(exc).lower()
    return any(kw in msg for kw in (
        "connection", "timeout", "timed out", "refused", "unreachable",
        "network", "socket", "eof", "reset by peer", "broken pipe",
        "name or service not known", "temporary failure",
    ))
