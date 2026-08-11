<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from 'vue'
import { api, ApiError } from './api'
import { ChatSocket } from './chatSocket'
import { renderMarkdown } from './markdown'
import type { Citation, Conversation, Message, ServerEvent, UploadedFile, User } from './types'

const user = ref<User | null>(null)
const username = ref('')
const password = ref('')
const error = ref('')
const conversations = ref<Conversation[]>([])
const selectedId = ref('')
const messages = ref<Message[]>([])
const files = ref<UploadedFile[]>([])
const selectedFiles = ref<string[]>([])
const question = ref('')
const streamingAnswer = ref('')
const streamingCitations = ref<Citation[]>([])
const currentRequestId = ref('')
const generating = ref(false)
const stage = ref('')
const connectionState = ref<'connecting' | 'connected' | 'reconnecting' | 'closed'>('closed')
const uploading = ref(false)
const messagePane = ref<HTMLElement>()
let pollTimer: number | undefined
let unlisten: (() => void) | undefined
const socket = new ChatSocket()
socket.onState = state => { connectionState.value = state }

const selectedConversation = computed(() => conversations.value.find(item => item.id === selectedId.value))
const statusText = computed(() => ({
  planning: '正在理解问题', retrieving: '正在检索证据', reranking: '正在核验证据', generating: '正在生成回答',
}[stage.value] ?? ''))

async function scrollBottom() {
  await nextTick()
  if (messagePane.value) messagePane.value.scrollTop = messagePane.value.scrollHeight
}

async function login() {
  error.value = ''
  try {
    user.value = await api.login(username.value, password.value)
    password.value = ''
    await initialize()
  } catch (cause) { error.value = cause instanceof Error ? cause.message : '登录失败' }
}

async function initialize() {
  conversations.value = await api.conversations()
  socket.connect()
  unlisten = socket.listen(handleEvent)
  if (conversations.value.length) await selectConversation(conversations.value[0].id)
}

async function logout() {
  await api.logout().catch(() => undefined)
  socket.close(); user.value = null; conversations.value = []; selectedId.value = ''
}

async function newConversation() {
  const item = await api.createConversation()
  conversations.value.unshift(item)
  await selectConversation(item.id)
}

async function renameConversation(item: Conversation) {
  const title = window.prompt('请输入新的会话名称', item.title)?.trim()
  if (!title) return
  const updated = await api.renameConversation(item.id, title)
  conversations.value = conversations.value.map(value => value.id === updated.id ? updated : value)
}

async function deleteConversation(item: Conversation) {
  if (!window.confirm(`确认删除“${item.title}”？`)) return
  await api.deleteConversation(item.id)
  conversations.value = conversations.value.filter(value => value.id !== item.id)
  if (selectedId.value === item.id) {
    selectedId.value = ''
    messages.value = []
    if (conversations.value.length) await selectConversation(conversations.value[0].id)
  }
}

async function selectConversation(id: string) {
  selectedId.value = id
  error.value = ''; streamingAnswer.value = ''; streamingCitations.value = []
  ;[messages.value, files.value] = await Promise.all([api.messages(id), api.files(id)])
  selectedFiles.value = files.value.filter(item => item.status === 'ready').map(item => item.id)
  startFilePolling()
  await scrollBottom()
}

function startFilePolling() {
  if (pollTimer) clearInterval(pollTimer)
  pollTimer = window.setInterval(async () => {
    if (!selectedId.value || !files.value.some(item => ['queued', 'processing'].includes(item.status))) return
    files.value = await api.files(selectedId.value)
  }, 2500)
}

async function upload(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file || !selectedId.value) return
  uploading.value = true; error.value = ''
  try {
    const item = await api.upload(selectedId.value, file)
    files.value = [...files.value.filter(value => value.id !== item.id), item]
  } catch (cause) { error.value = cause instanceof Error ? cause.message : '上传失败' }
  finally { uploading.value = false; input.value = '' }
}

async function removeFile(item: UploadedFile) {
  await api.deleteFile(item.id)
  files.value = files.value.filter(value => value.id !== item.id)
  selectedFiles.value = selectedFiles.value.filter(id => id !== item.id)
}

function ask() {
  const value = question.value.trim()
  if (!value || !selectedId.value || generating.value) return
  error.value = ''; streamingAnswer.value = ''; streamingCitations.value = []
  currentRequestId.value = crypto.randomUUID()
  generating.value = true; stage.value = 'planning'
  messages.value.push({
    id: crypto.randomUUID(), role: 'user', content: value, request_id: currentRequestId.value,
    created_at: new Date().toISOString(), citations: [],
  })
  try {
    socket.ask(currentRequestId.value, selectedId.value, value, selectedFiles.value)
    question.value = ''
    void scrollBottom()
  } catch (cause) {
    generating.value = false
    error.value = cause instanceof Error ? cause.message : '发送失败'
  }
}

function stop() {
  if (currentRequestId.value && selectedId.value) socket.stop(currentRequestId.value, selectedId.value)
}

async function handleEvent(event: ServerEvent) {
  if (event.conversation_id !== selectedId.value || (currentRequestId.value && event.request_id !== currentRequestId.value)) return
  if (event.type === 'status') stage.value = String(event.payload.stage ?? '')
  if (event.type === 'token') { streamingAnswer.value += String(event.payload.content ?? ''); await scrollBottom() }
  if (event.type === 'citations') streamingCitations.value = event.payload.items ?? []
  if (event.type === 'done') {
    generating.value = false; stage.value = ''; streamingAnswer.value = ''; streamingCitations.value = []
    messages.value = await api.messages(selectedId.value)
    await scrollBottom()
  }
  if (event.type === 'cancelled') { generating.value = false; stage.value = ''; error.value = '生成已停止' }
  if (event.type === 'error') {
    generating.value = false; stage.value = ''
    error.value = String(event.payload.message ?? '请求失败')
  }
}

onMounted(async () => {
  try { user.value = await api.me(); await initialize() }
  catch (cause) { if (cause instanceof ApiError && cause.status !== 401) error.value = cause.message }
})

onBeforeUnmount(() => {
  if (pollTimer) clearInterval(pollTimer)
  unlisten?.(); socket.close()
})
</script>

<template>
  <main v-if="!user" class="login-shell">
    <form class="login-card" @submit.prevent="login">
      <div class="brand-mark">标</div>
      <h1>智能招投标问答</h1>
      <p>登录后继续你的证据式问答</p>
      <label>用户名<input v-model="username" autocomplete="username" required minlength="3" /></label>
      <label>密码<input v-model="password" type="password" autocomplete="current-password" required minlength="8" /></label>
      <button class="primary" type="submit">登录</button>
      <div v-if="error" class="error-banner">{{ error }}</div>
    </form>
  </main>

  <main v-else class="app-shell">
    <aside class="sidebar">
      <div class="brand"><span class="brand-mark small">标</span><strong>招投标问答</strong></div>
      <button class="new-chat" @click="newConversation">＋ 新建会话</button>
      <nav class="conversation-list">
        <article v-for="item in conversations" :key="item.id" :class="{ active: item.id === selectedId }" @click="selectConversation(item.id)">
          <span>{{ item.title }}</span>
          <div class="row-actions">
            <button title="重命名" @click.stop="renameConversation(item)">✎</button>
            <button title="删除" @click.stop="deleteConversation(item)">×</button>
          </div>
        </article>
      </nav>
      <div class="account"><span>{{ user.username }}</span><button @click="logout">退出</button></div>
    </aside>

    <section class="chat-panel">
      <header>
        <div><h2>{{ selectedConversation?.title ?? '请选择会话' }}</h2><span class="connection" :class="connectionState">● {{ connectionState === 'connected' ? '已连接' : connectionState === 'reconnecting' ? '正在重连' : '连接中' }}</span></div>
        <label class="upload-button">{{ uploading ? '上传中…' : '上传文件' }}<input type="file" accept=".pdf,.txt,.md,.docx" :disabled="uploading || !selectedId" @change="upload" /></label>
      </header>

      <div v-if="files.length" class="file-strip">
        <label v-for="item in files" :key="item.id" :class="['file-chip', item.status]">
          <input v-if="item.status === 'ready'" v-model="selectedFiles" type="checkbox" :value="item.id" />
          <span>{{ item.original_name }}</span><small>{{ item.status === 'ready' ? `${item.chunk_count} 段` : item.status }}</small>
          <button title="删除文件" @click.prevent="removeFile(item)">×</button>
        </label>
      </div>

      <div ref="messagePane" class="messages">
        <div v-if="!messages.length && !streamingAnswer" class="empty-state"><b>从一个招投标问题开始</b><span>回答将依据本地、私有文件和必要的联网证据，并标注 [E1] 引用。</span></div>
        <article v-for="message in messages" :key="message.id" :class="['message', message.role]">
          <div class="avatar">{{ message.role === 'user' ? '你' : 'AI' }}</div>
          <div class="bubble">
            <div v-if="message.role === 'assistant'" class="markdown" v-html="renderMarkdown(message.content)" />
            <p v-else>{{ message.content }}</p>
            <details v-if="message.citations?.length" class="citations"><summary>查看 {{ message.citations.length }} 条引用</summary>
              <a v-for="citation in message.citations" :key="citation.evidence_id" :href="citation.source_url || undefined" target="_blank" rel="noopener noreferrer nofollow">
                <b>[{{ citation.evidence_id }}] {{ citation.title || citation.source_id }}</b><span>{{ citation.category }} · {{ citation.source_type }}</span>
              </a>
            </details>
          </div>
        </article>
        <article v-if="streamingAnswer || generating" class="message assistant">
          <div class="avatar">AI</div><div class="bubble">
            <div v-if="streamingAnswer" class="markdown" v-html="renderMarkdown(streamingAnswer)" />
            <div v-else class="thinking"><i></i><i></i><i></i>{{ statusText }}</div>
            <details v-if="streamingCitations.length" class="citations" open><summary>引用</summary>
              <a v-for="citation in streamingCitations" :key="citation.evidence_id" :href="citation.url || undefined" target="_blank" rel="noopener noreferrer nofollow"><b>[{{ citation.evidence_id }}] {{ citation.title }}</b><span>{{ citation.category }}</span></a>
            </details>
          </div>
        </article>
      </div>

      <div v-if="error" class="error-banner inline">{{ error }} <button @click="error = ''">×</button></div>
      <form class="composer" @submit.prevent="ask">
        <textarea v-model="question" :disabled="!selectedId" maxlength="6000" rows="2" placeholder="输入招投标问题；Enter 发送，Shift+Enter 换行" @keydown.enter.exact.prevent="ask" />
        <button v-if="generating" class="stop" type="button" @click="stop">■ 停止</button>
        <button v-else class="send" type="submit" :disabled="!question.trim() || connectionState !== 'connected'">发送</button>
      </form>
      <footer>回答仅供辅助判断，请以原始公告、合同与现行法律政策为准。</footer>
    </section>
  </main>
</template>
