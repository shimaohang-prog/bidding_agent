export interface User { id: string; username: string }
export interface Conversation { id: string; title: string; created_at: string; updated_at: string }
export interface Citation {
  evidence_id: string; source_type: string; category: string; title: string;
  source_url?: string | null; source_id: string; metadata_json?: Record<string, unknown>;
  url?: string | null; metadata?: Record<string, unknown>;
}
export interface Message { id: string; role: 'user' | 'assistant'; content: string; request_id?: string; created_at: string; citations: Citation[] }
export interface UploadedFile { id: string; conversation_id: string; original_name: string; mime_type: string; size_bytes: number; sha256: string; status: string; error_code?: string; chunk_count: number; created_at: string }
export interface KnowledgeFile { name: string; relative_path: string; size_bytes: number; mime_type: string }
export interface ServerEvent {
  type: 'ack' | 'status' | 'token' | 'citations' | 'done' | 'cancelled' | 'error' | 'pong'
  request_id: string; conversation_id: string; seq: number | null; payload: Record<string, any>
}
