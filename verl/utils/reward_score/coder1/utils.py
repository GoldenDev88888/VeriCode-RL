"""Shared execution defaults for the Code-R1 fork."""
_ERROR_MSG_PREFIX = 'Failed to execute program: '
_DEFAULT_TIMEOUT_SECONDS = 30


def check_executor_alive(executor):
    import requests
    try:
        return requests.get(executor + '/', timeout=3).status_code in (200, 404)
    except requests.RequestException:
        return False
