from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ConversationCreate(BaseModel):
    title: str = Field(default="新会话", min_length=1, max_length=160)


class ConversationRename(BaseModel):
    title: str = Field(min_length=1, max_length=160)


class ConversationView(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class CitationView(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    evidence_id: str
    source_type: str
    category: str
    title: str
    source_url: str | None
    source_id: str
    metadata_json: dict


class MessageView(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    role: str
    content: str
    request_id: str | None
    created_at: datetime
    citations: list[CitationView] = []
