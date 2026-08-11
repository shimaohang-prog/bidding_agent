"""严格校验双向 WebSocket 事件。"""

from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class ClientEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str
    request_id: UUID
    conversation_id: UUID


class AskEvent(ClientEvent):
    type: Literal["ask"]
    question: str = Field(min_length=1, max_length=6000)
    client_message_id: str = Field(min_length=1, max_length=64)
    file_ids: list[UUID] = Field(default_factory=list, max_length=20)


class StopEvent(ClientEvent):
    type: Literal["stop"]


class ResumeEvent(ClientEvent):
    type: Literal["resume"]
    last_seq: int = Field(default=0, ge=0)


class PingEvent(ClientEvent):
    type: Literal["ping"]


InboundEvent = Annotated[AskEvent | StopEvent | ResumeEvent | PingEvent, Field(discriminator="type")]
inbound_adapter = TypeAdapter(InboundEvent)


class DomainEvent(BaseModel):
    type: Literal["status", "token", "citations", "done", "cancelled", "error"]
    payload: dict[str, Any] = Field(default_factory=dict)


class ServerEvent(BaseModel):
    type: Literal["ack", "status", "token", "citations", "done", "cancelled", "error", "pong"]
    request_id: str
    conversation_id: str
    seq: int | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
