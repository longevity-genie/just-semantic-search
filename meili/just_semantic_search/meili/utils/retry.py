from eliot import start_action
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from meilisearch_python_sdk.errors import MeilisearchApiError, MeilisearchCommunicationError

def create_retry_decorator(
    attempts: int = 5,
    multiplier: float = 1,
    min_wait: float = 4,
    max_wait: float = 40
) -> callable:
    """Create a retry decorator with specified parameters"""
    def log_retry_attempt(retry_state):
        """Log retry attempt with Eliot action"""
        with start_action(action_type="retry_attempt") as action:
            if retry_state.outcome is not None:
                exception = retry_state.outcome.exception()
                action.add_success_fields(
                    attempt_number=retry_state.attempt_number,
                    exception_type=type(exception).__name__ if exception else None,
                    exception_message=str(exception) if exception else None,
                    next_attempt_in=retry_state.next_action.sleep if hasattr(retry_state.next_action, 'sleep') else None
                )

    def log_retry_done(retry_state):
        """Log retry completion with Eliot action"""
        with start_action(action_type="retry_done") as action:
            action.add_success_fields(
                attempt_number=retry_state.attempt_number,
                was_successful=retry_state.outcome is not None and retry_state.outcome.exception() is None
            )

    return retry(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(
            multiplier=multiplier,
            min=min_wait,
            max=max_wait
        ),
        retry=retry_if_exception_type((MeilisearchApiError, MeilisearchCommunicationError)),
        before_sleep=log_retry_attempt,
        after=log_retry_done
    )