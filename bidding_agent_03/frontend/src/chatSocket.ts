import type { ServerEvent } from './types'

type Listener = (event: ServerEvent) => void

export class ChatSocket {
  private socket?: WebSocket
  private listeners = new Set<Listener>()
  private attempts = 0
  private manualClose = false
  private pingTimer?: number
  private active = new Map<string, { conversationId: string; lastSeq: number }>()
  onState?: (state: 'connecting' | 'connected' | 'reconnecting' | 'closed') => void

  connect() {
    this.manualClose = false
    this.onState?.(this.attempts ? 'reconnecting' : 'connecting')
    const scheme = location.protocol === 'https:' ? 'wss' : 'ws'
    this.socket = new WebSocket(`${scheme}://${location.host}/api/v1/ws/chat`)
    this.socket.onopen = () => {
      this.attempts = 0; this.onState?.('connected')
      for (const [requestId, item] of this.active) this.resume(requestId, item.conversationId, item.lastSeq)
      this.pingTimer = window.setInterval(() => {
        const first = this.active.entries().next().value as [string, { conversationId: string }] | undefined
        if (first) this.send({ type: 'ping', request_id: first[0], conversation_id: first[1].conversationId })
      }, 25_000)
    }
    this.socket.onmessage = (message) => {
      const event = JSON.parse(message.data) as ServerEvent
      if (event.seq !== null) {
        const current = this.active.get(event.request_id)
        if (current && event.seq <= current.lastSeq) return
        if (current) current.lastSeq = event.seq
      }
      if (['done', 'cancelled', 'error'].includes(event.type) && event.seq !== null) {
        window.setTimeout(() => this.active.delete(event.request_id), 120_000)
      }
      for (const listener of this.listeners) listener(event)
    }
    this.socket.onclose = () => {
      if (this.pingTimer) clearInterval(this.pingTimer)
      if (this.manualClose) { this.onState?.('closed'); return }
      this.onState?.('reconnecting')
      const wait = Math.min(10_000, 500 * 2 ** this.attempts++)
      window.setTimeout(() => this.connect(), wait)
    }
  }

  close() {
    this.manualClose = true
    this.active.clear()
    if (this.pingTimer) clearInterval(this.pingTimer)
    this.socket?.close()
  }
  listen(listener: Listener) { this.listeners.add(listener); return () => this.listeners.delete(listener) }
  private send(body: object) {
    if (this.socket?.readyState !== WebSocket.OPEN) throw new Error('连接尚未就绪')
    this.socket.send(JSON.stringify(body))
  }
  ask(requestId: string, conversationId: string, question: string, fileIds: string[]) {
    this.active.set(requestId, { conversationId, lastSeq: 0 })
    this.send({ type: 'ask', request_id: requestId, conversation_id: conversationId, client_message_id: crypto.randomUUID(), question, file_ids: fileIds })
  }
  stop(requestId: string, conversationId: string) { this.send({ type: 'stop', request_id: requestId, conversation_id: conversationId }) }
  resume(requestId: string, conversationId: string, lastSeq: number) { this.send({ type: 'resume', request_id: requestId, conversation_id: conversationId, last_seq: lastSeq }) }
}
