from pydantic import BaseModel
from typing import Any, Dict

class Problem(BaseModel):
    features: Dict[str, Any]
    text: str | None = None

class Solution(BaseModel):
    payload: Dict[str, Any]

class Case(BaseModel):
    id: str
    problem: Problem
    solution: Solution
