from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's question for the AI", min_length=1)

class QueryResponse(BaseModel):
    status: str
    response: str