from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class ParsedResponse(BaseModel, Generic[T]):
    value: T | None
    raw_text: str | None
    model: str
    attempts: int
    success: bool
    errors: list[str] | None = None
