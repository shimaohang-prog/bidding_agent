from datetime import datetime

from pydantic import BaseModel, ConfigDict


class FileView(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    conversation_id: str
    original_name: str
    mime_type: str
    size_bytes: int
    sha256: str
    status: str
    error_code: str | None
    chunk_count: int
    created_at: datetime
