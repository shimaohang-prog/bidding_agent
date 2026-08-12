import type { Conversation, KnowledgeFile, Message, UploadedFile, User } from './types'

export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string) { super(message) }
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`/api/v1${path}`, {
    ...options,
    credentials: 'include',
    headers: options.body instanceof FormData ? options.headers : { 'Content-Type': 'application/json', ...options.headers },
  })
  if (response.status === 204) return undefined as T
  const body = await response.json().catch(() => ({}))
  if (!response.ok) throw new ApiError(response.status, body.error_code ?? 'REQUEST_FAILED', body.message ?? '请求失败')
  return body as T
}

export const api = {
  me: () => request<User>('/auth/me'),
  login: (username: string, password: string) => request<User>('/auth/login', { method: 'POST', body: JSON.stringify({ username, password }) }),
  logout: () => request<void>('/auth/logout', { method: 'POST' }),
  conversations: () => request<Conversation[]>('/conversations'),
  createConversation: (title = '新会话') => request<Conversation>('/conversations', { method: 'POST', body: JSON.stringify({ title }) }),
  renameConversation: (id: string, title: string) => request<Conversation>(`/conversations/${id}`, { method: 'PATCH', body: JSON.stringify({ title }) }),
  deleteConversation: (id: string) => request<void>(`/conversations/${id}`, { method: 'DELETE' }),
  messages: (id: string, beforeId?: string) => request<Message[]>(`/conversations/${id}/messages?limit=50${beforeId ? `&before_id=${encodeURIComponent(beforeId)}` : ''}`),
  files: (conversationId: string) => request<UploadedFile[]>(`/files?conversation_id=${encodeURIComponent(conversationId)}`),
  knowledgeFiles: (category: string) => request<KnowledgeFile[]>(`/knowledge/${encodeURIComponent(category)}/files`),
  knowledgeFileUrl: (category: string, path: string) => `/api/v1/knowledge/${encodeURIComponent(category)}/open?path=${encodeURIComponent(path)}`,
  uploadedFileUrl: (id: string) => `/api/v1/files/${encodeURIComponent(id)}/content`,
  upload: (conversationId: string, file: File) => {
    const data = new FormData(); data.append('conversation_id', conversationId); data.append('upload', file)
    return request<UploadedFile>('/files', { method: 'POST', body: data })
  },
  deleteFile: (id: string) => request<void>(`/files/${id}`, { method: 'DELETE' }),
}
