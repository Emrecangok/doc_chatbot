from pydantic import BaseModel
from typing import Optional




class QueryRequest(BaseModel):
    question : str
    thread_id : Optional[str] = None


class QueryResponse(BaseModel):
    answer : str
    sources : Optional[list[str]] = []



