from pydantic import BaseModel
from typing import Optional, List, TypeVar, Generic


T = TypeVar("T", bound=BaseModel)


class ParsedResponse(BaseModel, Generic[T]):
    value: Optional[T]
    raw_text: Optional[str]
    model: str
    attempts: int
    success: bool
    errors: Optional[List[str]] = None
