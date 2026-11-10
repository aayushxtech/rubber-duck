import json
import os
import time

from app.services.model_registery import LIGHT_WEIGHT_MODEL, REASONING_MODEL
from app.services.parsed_response import ParsedResponse, T
from dotenv import load_dotenv
from groq import (
    APIConnectionError,
    APITimeoutError,
    Groq,
    InternalServerError,
    RateLimitError,
)
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


class MissingAPIKeyError(RuntimeError):
    """Raised when GROQ_API_KEY is not set in the environment."""


_groq_client: Groq | None = None


def get_groq_client() -> Groq:
    """Return a lazily‑initialized Groq client.

    The API key is loaded from the environment on first use.  If the key is
    missing, a ``MissingAPIKeyError`` is raised.  This ensures that importing the
    module does not require credentials.
    """
    global _groq_client
    if _groq_client is not None:
        return _groq_client
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise MissingAPIKeyError("GROQ_API_KEY is not set")
    _groq_client = Groq(api_key=api_key)
    return _groq_client


def _is_transient_error(exc: Exception) -> bool:
    """Classify whether an exception is transient and eligible for retry.

    - MissingAPIKeyError is never transient.
    - Groq SDK APIConnectionError and APITimeoutError are transient (do not expose status_code).
    - Python built-in ConnectionError and TimeoutError are transient.
    - Groq SDK RateLimitError (429) and InternalServerError (500+) are transient.
    - Explicit HTTP status codes:
        * 408, 429, and 500-599 are transient.
        * All other HTTP status codes (e.g. 400 BadRequestError, 401 AuthenticationError,
          403 PermissionDeniedError, 404 NotFoundError) are permanent.
    - If no status code or known exception type is available, fallback to message keywords.
    """
    if isinstance(exc, MissingAPIKeyError):
        return False

    # Groq SDK connection failures and timeouts (which inherit from APIError without status_code)
    # as well as standard Python ConnectionError / TimeoutError
    if isinstance(
        exc, (APIConnectionError, APITimeoutError, ConnectionError, TimeoutError)
    ):
        return True

    # Explicit Groq SDK error types for rate limits and server errors
    if isinstance(exc, (RateLimitError, InternalServerError)):
        return True

    # Check for HTTP status code on exception or attached response
    status_code = getattr(exc, "status_code", None)
    if status_code is None and hasattr(exc, "response"):
        status_code = getattr(exc.response, "status_code", None)

    if status_code is not None:
        try:
            code = int(status_code)
            if code == 408 or code == 429 or (500 <= code <= 599):
                return True
            # Any explicit non-transient HTTP status code is permanent
            return False
        except (ValueError, TypeError):
            pass

    msg = str(exc).lower()
    keywords = [
        "rate limit",
        "timed out",
        "timeout",
        "service unavailable",
        "internal server error",
    ]
    return any(k in msg for k in keywords)


# ---------------------------------------------------------------------------
# Public LLM helper functions
# ---------------------------------------------------------------------------


def call_structured_llm(
    *,
    task_name: str,
    prompt: str,
    schema: type[T],
    model: str = REASONING_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    max_retries: int = 5,
) -> "ParsedResponse[T]":
    """Call a LLM that is expected to return JSON matching *schema*.

    The function retries on transient provider errors, applies a simple back‑off,
    and uses a repair prompt to fix JSON or schema validation issues.  The
    contract (return type and error collection) matches the original implementation.
    """
    attempts = 0
    errors: list[str] = []
    raw_text: str | None = None
    current_prompt = prompt
    current_temperature = temperature

    while attempts < max_retries:
        attempts += 1
        try:
            client = get_groq_client()
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": current_prompt}],
                temperature=current_temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            raw_text = completion.choices[0].message.content
            if raw_text is None or raw_text.strip() == "":
                raise ValueError("LLM returned empty content")
            raw_text = raw_text.strip()

            # Try to parse JSON
            try:
                parsed_json = json.loads(raw_text)
            except json.JSONDecodeError as e:
                error_msg = f"JSON decode error: {e!s}"
                errors.append(error_msg)
                current_prompt = _build_repair_prompt(
                    base_prompt=prompt,
                    error_message=error_msg,
                    attempt=attempts,
                    max_retries=max_retries,
                )
                current_temperature = max(0.1, current_temperature - 0.2)
                continue

            # Validate against schema
            try:
                validated = schema.model_validate(parsed_json)
            except ValidationError as e:
                error_msg = f"Schema validation error: {e!s}"
                errors.append(error_msg)
                current_prompt = _build_repair_prompt(
                    base_prompt=prompt,
                    error_message=error_msg,
                    attempt=attempts,
                    max_retries=max_retries,
                )
                current_temperature = max(0.1, current_temperature - 0.2)
                continue

            return ParsedResponse[T](
                value=validated,
                raw_text=raw_text,
                model=model,
                attempts=attempts,
                success=True,
                errors=None,
            )
        except Exception as e:
            # Provider-level error handling
            if _is_transient_error(e):
                error_msg = (
                    f"Transient LLM call error (retry {attempts}): {type(e).__name__}"
                )
                errors.append(error_msg)
                if attempts < max_retries:
                    backoff = min(2**attempts, 5)
                    time.sleep(backoff)
                    current_temperature = max(0.1, current_temperature - 0.2)
                    continue
                break
            else:
                # Permanent provider error (e.g. MissingAPIKeyError, 401 Auth, 400/404 Invalid Model)
                # Stop immediately without retrying
                error_msg = f"Permanent LLM call error: {type(e).__name__}"
                errors.append(error_msg)
                break

    # Exhausted retries or encountered permanent error – return a failure response.
    return ParsedResponse[T](
        value=None,
        raw_text=raw_text,
        model=model,
        attempts=attempts,
        success=False,
        errors=errors,
    )


def call_text_llm(
    *,
    task_name: str,
    prompt: str,
    model: str = LIGHT_WEIGHT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> str:
    """Call a lightweight LLM for plain‑text output.

    Retries are performed only for transient provider failures. Permanent
    failures stop immediately and raise a controlled exception.
    """
    attempts = 0
    max_retries = 3
    current_temperature = temperature
    while attempts < max_retries:
        attempts += 1
        try:
            client = get_groq_client()
            kwargs = {}
            if "gpt-oss" in model:
                kwargs["extra_body"] = {"reasoning_effort": "low"}
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=current_temperature,
                max_tokens=max_tokens,
                stream=False,
                **kwargs,
            )
            raw_text = completion.choices[0].message.content
            if raw_text is None or raw_text.strip() == "":
                if attempts < max_retries:
                    backoff = min(2**attempts, 5)
                    time.sleep(backoff)
                    current_temperature = max(0.1, current_temperature - 0.2)
                    continue
                raise RuntimeError("LLM call failed: empty content returned")
            return raw_text.strip()
        except Exception as e:
            if _is_transient_error(e) and attempts < max_retries:
                backoff = min(2**attempts, 5)
                time.sleep(backoff)
                current_temperature = max(0.1, current_temperature - 0.2)
                continue
            # Permanent error or exhausted retries – stop immediately and raise controlled exception.
            if isinstance(e, MissingAPIKeyError):
                raise
            raise RuntimeError(f"LLM call failed: {type(e).__name__}") from e


def _build_repair_prompt(
    *,
    base_prompt: str,
    error_message: str,
    attempt: int,
    max_retries: int,
) -> str:
    return f"""
The previous output was INVALID and did not match the required JSON schema.

ATTEMPT {attempt} OF {max_retries}

ERROR DETAILS:
{error_message}

INSTRUCTIONS:
- Fix the output so it EXACTLY matches the required schema
- Return ONLY corrected JSON
- Do NOT include explanations, markdown, or extra text
- Do NOT omit required fields
- Do NOT add new fields

ORIGINAL TASK:
{base_prompt}
""".strip()
