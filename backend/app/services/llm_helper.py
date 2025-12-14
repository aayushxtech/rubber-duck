from pydantic import BaseModel, ValidationError
from typing import Type, TypeVar, List, Optional
import os
from app.services.model_registery import REASONING_MODEL
from app.services.parsed_response import ParsedResponse, T
from app.schemas.request import DuckRequest
from groq import Groq
import json

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)


def call_structured_llm(
    *,
    task_name: str,
    prompt: str,
    schema: Type[T],
    model: str = REASONING_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    max_retries: int = 5,
) -> "ParsedResponse[T]":

    attempts = 0
    errors: List[str] = []
    raw_text: Optional[str] = None
    current_prompt = prompt
    current_temperature = temperature

    while attempts < max_retries:
        attempts += 1

        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": current_prompt,
                    }
                ],
                temperature=current_temperature,
                max_completion_tokens=max_tokens,
                stream=False,
            )

            raw_text = completion.choices[0].message.content
            if raw_text is None:
                raise ValueError("LLM returned empty content")
            raw_text = raw_text.strip()

            try:
                parsed_json = json.loads(raw_text)
            except json.JSONDecodeError as e:
                error_msg = f"JSON decode error: {str(e)}"
                errors.append(error_msg)
                current_prompt = _build_repair_prompt(
                    base_prompt=prompt,
                    error_message=error_msg,
                    attempt=attempts,
                    max_retries=max_retries,
                )
                current_temperature = max(0.1, current_temperature - 0.2)
                continue

            try:
                validated = schema.model_validate(parsed_json)
            except ValidationError as e:
                error_msg = f"Schema validation error: {str(e)}"
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
            error_msg = f"LLM call error: {str(e)}"
            errors.append(error_msg)
            current_temperature = max(0.1, current_temperature - 0.2)

    return ParsedResponse[T](
        value=None,
        raw_text=raw_text,
        model=model,
        attempts=attempts,
        success=False,
        errors=errors,
    )


def call_text_llm():
    pass


def _retry_and_repair():
    pass


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
