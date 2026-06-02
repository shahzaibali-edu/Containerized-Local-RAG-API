from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's question for the AI", min_length=1)


class QueryResponse(BaseModel):
    status: str
    response: str


class DocumentInfo(BaseModel):
    filename: str
    status: str   # "processing" | "ready" | "error"
    chunks: int


class UploadResponse(BaseModel):
    status: str
    filename: str
    size_mb: float
    message: str