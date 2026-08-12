<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { api, ApiError } from './api'
import { ChatSocket } from './chatSocket'
import { renderMarkdown } from './markdown'
import type { Citation, Conversation, KnowledgeFile, Message, ServerEvent, UploadedFile, User } from './types'

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
const connectionState = ref<'connecting' | 'connected' | 'reconnecting' | 'closed'>('closed')
const uploading = ref(false)
const sidebarCollapsed = ref(false)
const searchQuery = ref('')
const knowledgePanelOpen = ref(false)
const activeKnowledgeKey = ref<KnowledgeKey>('enterprise')
const knowledgeFiles = ref<KnowledgeFile[]>([])
const loadedKnowledgeKey = ref<KnowledgeKey | null>(null)
const knowledgeFileQuery = ref('')
const knowledgeLoading = ref(false)
const knowledgeError = ref('')
const messagePane = ref<HTMLElement>()
const composerInput = ref<HTMLTextAreaElement>()
let pollTimer: number | undefined
let unlisten: (() => void) | undefined
const socket = new ChatSocket()
socket.onState = state => { connectionState.value = state }

interface ConversationGeneration {
  requestId: string
  streamingAnswer: string
  streamingCitations: Citation[]
  generating: boolean
  stage: string
  error: string
}

type KnowledgeKey = 'enterprise' | 'tender' | 'product' | 'laws' | 'policy' | 'uploads'

interface KnowledgeLibrary {
  key: KnowledgeKey
  label: string
  shortLabel: string
  description: string
  icon: string
  fields: string[]
}

const knowledgeLibraries: KnowledgeLibrary[] = [
  { key: 'enterprise', label: 'enterprise 企业信息库', shortLabel: 'enterprise', description: '企业主体、统一社会信用代码、法人、资质与经营状态', icon: '企', fields: ['企业名称', '信用代码', '法定代表人', '企业状态'] },
  { key: 'tender', label: 'tender 招投标库', shortLabel: 'tender', description: '招标、采购、中标、项目公告与成交信息', icon: '标', fields: ['项目名称', '采购单位', '中标单位', '中标金额'] },
  { key: 'product', label: 'product 产品库', shortLabel: 'product', description: '产品参数、型号、供应商与价格信息', icon: '品', fields: ['产品名称', '分类', '供应商', '价格'] },
  { key: 'laws', label: 'laws 法律法规库', shortLabel: 'laws', description: '招投标与政府采购相关法律法规及条款', icon: '法', fields: ['法规名称', '条款内容', '效力层级', '更新时间'] },
  { key: 'policy', label: 'policy 政策文件库', shortLabel: 'policy', description: '部门规章、规范性文件、地方政策与办事规则', icon: '策', fields: ['政策名称', '发布部门', '适用地区', '更新时间'] },
  { key: 'uploads', label: '已上传文件', shortLabel: '已上传文件', description: '当前会话上传并完成解析的私有文件', icon: '文', fields: [] },
]

const starterCards = [
  { icon: '⌕', title: '解读招标文件与质疑点', description: '分解招标条款、全维度查寻关键内容与潜在风险', prompt: '请帮我解读这份招标文件的关键要求和潜在质疑点。' },
  { icon: '□', title: '分析评分办法', description: '拆解评分指标权重，提供得分逻辑与优化建议', prompt: '请分析招标文件中的评分办法，并给出投标优化建议。' },
  { icon: '↗', title: '根据公告条款关键信息', description: '梳理招标范围、时间节点，保证信息核心准确', prompt: '请梳理公告和条款中的招标范围、时间节点及关键要求。' },
]

const generations = reactive<Record<string, ConversationGeneration>>({})
const idleGeneration: ConversationGeneration = {
  requestId: '', streamingAnswer: '', streamingCitations: [],
  generating: false, stage: '', error: '',
}

function generationFor(conversationId: string): ConversationGeneration {
  if (!generations[conversationId]) {
    generations[conversationId] = {
      requestId: '', streamingAnswer: '', streamingCitations: [],
      generating: false, stage: '', error: '',
    }
  }
  return generations[conversationId]
}

function conversationTime(value: string) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  return new Intl.DateTimeFormat('zh-CN', { hour: '2-digit', minute: '2-digit', hour12: false }).format(date)
}

function openKnowledge(key: KnowledgeKey) {
  knowledgePanelOpen.value = true
  void selectKnowledge(key)
}

async function selectKnowledge(key: KnowledgeKey) {
  activeKnowledgeKey.value = key
  knowledgeFileQuery.value = ''
  knowledgeError.value = ''
  if (key === 'uploads') {
    knowledgeLoading.value = false
    knowledgeFiles.value = []
    loadedKnowledgeKey.value = key
    return
  }
  if (loadedKnowledgeKey.value === key) return
  knowledgeLoading.value = true
  try {
    const items = await api.knowledgeFiles(key)
    if (activeKnowledgeKey.value === key) {
      knowledgeFiles.value = items
      loadedKnowledgeKey.value = key
    }
  } catch (cause) {
    if (activeKnowledgeKey.value === key) {
      knowledgeFiles.value = []
      knowledgeError.value = cause instanceof Error ? cause.message : '读取知识库文件失败'
    }
  } finally {
    if (activeKnowledgeKey.value === key) knowledgeLoading.value = false
  }
}

async function useStarter(prompt: string) {
  question.value = prompt
  await nextTick()
  composerInput.value?.focus()
}

const selectedConversation = computed(() => conversations.value.find(item => item.id === selectedId.value))
const selectedGeneration = computed(() => generations[selectedId.value] ?? idleGeneration)
const filteredConversations = computed(() => {
  const keyword = searchQuery.value.trim().toLocaleLowerCase()
  return keyword
    ? conversations.value.filter(item => item.title.toLocaleLowerCase().includes(keyword))
    : conversations.value
})
const activeKnowledge = computed(() => knowledgeLibraries.find(item => item.key === activeKnowledgeKey.value) ?? knowledgeLibraries[0])
const readyFileCount = computed(() => files.value.filter(item => item.status === 'ready').length)
const visibleKnowledgeFiles = computed(() => {
  const keyword = knowledgeFileQuery.value.trim().toLocaleLowerCase()
  const items = keyword
    ? knowledgeFiles.value.filter(item => item.name.toLocaleLowerCase().includes(keyword) || item.relative_path.toLocaleLowerCase().includes(keyword))
    : knowledgeFiles.value
  return items.slice(0, 200)
})
const statusText = computed(() => ({
  planning: '正在理解问题', retrieving: '正在检索证据', reranking: '正在核验证据', generating: '正在生成回答',
}[selectedGeneration.value.stage] ?? ''))

function fileSize(value: number) {
  if (value < 1024) return `${value} B`
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`
  return `${(value / 1024 / 1024).toFixed(1)} MB`
}

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
  for (const conversationId of Object.keys(generations)) delete generations[conversationId]
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
  error.value = ''
  const [nextMessages, nextFiles] = await Promise.all([api.messages(id), api.files(id)])
  if (selectedId.value !== id) return
  messages.value = nextMessages
  files.value = nextFiles
  const state = generations[id]
  if (state && !state.generating) {
    state.streamingAnswer = ''
    state.streamingCitations = []
  }
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
  const conversationId = selectedId.value
  if (!value || !conversationId) return
  const state = generationFor(conversationId)
  if (state.generating) return
  error.value = ''
  state.error = ''
  state.streamingAnswer = ''
  state.streamingCitations = []
  state.requestId = crypto.randomUUID()
  state.generating = true
  state.stage = 'planning'
  messages.value.push({
    id: crypto.randomUUID(), role: 'user', content: value, request_id: state.requestId,
    created_at: new Date().toISOString(), citations: [],
  })
  try {
    socket.ask(state.requestId, conversationId, value, [...selectedFiles.value])
    question.value = ''
    void scrollBottom()
  } catch (cause) {
    state.generating = false
    state.error = cause instanceof Error ? cause.message : '发送失败'
  }
}

function stop() {
  const state = generations[selectedId.value]
  if (state?.requestId && state.generating) socket.stop(state.requestId, selectedId.value)
}

async function handleEvent(event: ServerEvent) {
  const state = generationFor(event.conversation_id)
  if (state.requestId && event.request_id !== state.requestId) return
  if (!state.requestId) state.requestId = event.request_id
  if (event.type === 'ack' && event.payload.conversation_title) {
    const title = String(event.payload.conversation_title)
    conversations.value = conversations.value.map(item => item.id === event.conversation_id ? { ...item, title } : item)
  }
  if (event.type === 'status') state.stage = String(event.payload.stage ?? '')
  if (event.type === 'token') {
    state.streamingAnswer += String(event.payload.content ?? '')
    if (event.conversation_id === selectedId.value) await scrollBottom()
  }
  if (event.type === 'citations') state.streamingCitations = event.payload.items ?? []
  if (event.type === 'done') {
    state.generating = false
    state.stage = ''
    try {
      const refreshed = await api.messages(event.conversation_id)
      if (event.conversation_id === selectedId.value) {
        messages.value = refreshed
        await scrollBottom()
      }
      state.streamingAnswer = ''
      state.streamingCitations = []
    } catch (cause) {
      state.error = cause instanceof Error ? cause.message : '刷新回答失败'
    }
  }
  if (event.type === 'cancelled') {
    state.generating = false
    state.stage = ''
    state.error = '生成已停止'
  }
  if (event.type === 'error') {
    state.generating = false
    state.stage = ''
    state.error = String(event.payload.message ?? '请求失败')
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

  <main v-else :class="['app-shell', { 'sidebar-collapsed': sidebarCollapsed }]">
    <aside class="sidebar">
      <button
        class="sidebar-toggle"
        type="button"
        :title="sidebarCollapsed ? '展开侧栏' : '收起侧栏'"
        :aria-label="sidebarCollapsed ? '展开侧栏' : '收起侧栏'"
        :aria-expanded="!sidebarCollapsed"
        @click="sidebarCollapsed = !sidebarCollapsed"
      >{{ sidebarCollapsed ? '›' : '‹' }}</button>

      <div class="brand">
        <span class="brand-mark small">标</span>
        <span class="brand-copy"><strong>招投标问答</strong><small>BidMind</small></span>
      </div>

      <button class="new-chat" title="新建会话" @click="newConversation">
        <span class="new-chat-icon">＋</span><span class="new-chat-label">新建会话</span>
      </button>

      <label class="conversation-search">
        <span>⌕</span>
        <input v-model="searchQuery" type="search" placeholder="搜索会话" aria-label="搜索会话" />
      </label>

      <div class="sidebar-section-title"><span>会话</span><small>{{ filteredConversations.length }}</small></div>
      <nav class="conversation-list">
        <article v-for="item in filteredConversations" :key="item.id" :title="item.title" :class="{ active: item.id === selectedId }" @click="selectConversation(item.id)">
          <span class="conversation-short">{{ item.title.slice(0, 1) }}</span>
          <span class="conversation-title">{{ item.title }} <i v-if="generations[item.id]?.generating" class="active-generation" title="正在生成">●</i></span>
          <time>{{ conversationTime(item.updated_at) }}</time>
          <div class="row-actions">
            <button title="重命名" @click.stop="renameConversation(item)">✎</button>
            <button title="删除" @click.stop="deleteConversation(item)">×</button>
          </div>
        </article>
        <p v-if="!filteredConversations.length" class="no-conversations">没有匹配的会话</p>
      </nav>

      <section class="sidebar-knowledge">
        <div class="sidebar-section-title"><span>知识库与文件</span><small>＋</small></div>
        <button v-for="library in knowledgeLibraries" :key="library.key" :title="library.label" @click="openKnowledge(library.key)">
          <span class="knowledge-icon">{{ library.icon }}</span>
          <span class="knowledge-label">{{ library.shortLabel }}</span>
          <small v-if="library.key === 'uploads'">{{ readyFileCount }}</small>
          <span class="knowledge-arrow">›</span>
        </button>
      </section>

      <div class="account">
        <span class="account-avatar">{{ user.username.slice(0, 1).toUpperCase() }}</span>
        <span class="account-name"><strong>{{ user.username }}</strong><small>管理员</small></span>
        <button title="退出登录" @click="logout">退出</button>
      </div>
    </aside>

    <section class="chat-panel">
      <header class="topbar">
        <div class="model-name"><span class="model-icon">◇</span><strong>BidMind 1.0</strong><span>⌄</span><b>专业版</b></div>
        <div class="topbar-status">
          <span class="strict-mode">♢ 引用模式：严格</span>
          <span class="connection" :class="connectionState"><i class="connection-dot"></i>{{ connectionState === 'connected' ? '已连接本地知识库' : connectionState === 'reconnecting' ? '正在重新连接' : '正在连接' }}</span>
        </div>
        <div class="topbar-actions">
          <button type="button" title="查看知识库" @click="openKnowledge('enterprise')">▤ 知识库</button>
          <label class="upload-button">⇧ {{ uploading ? '上传中…' : '上传文件' }}<input type="file" accept=".pdf,.txt,.md,.docx" :disabled="uploading || !selectedId" @change="upload" /></label>
        </div>
      </header>

      <div v-if="files.length" class="file-strip">
        <label v-for="item in files" :key="item.id" :class="['file-chip', item.status]">
          <input v-if="item.status === 'ready'" v-model="selectedFiles" type="checkbox" :value="item.id" />
          <span>{{ item.original_name }}</span><small>{{ item.status === 'ready' ? `${item.chunk_count} 段` : item.status }}</small>
          <button title="删除文件" @click.prevent="removeFile(item)">×</button>
        </label>
      </div>

      <div ref="messagePane" class="messages">
        <div v-if="!messages.length && !selectedGeneration.streamingAnswer" class="welcome-state">
          <div class="welcome-visual" aria-hidden="true"><i></i><span>•••</span><b></b></div>
          <h1>从一个招投标问题开始</h1>
          <p>基于本地知识库与权威网络信息，提供<span>专业、可靠</span>的招投标解答</p>
          <div class="starter-grid">
            <button v-for="card in starterCards" :key="card.title" type="button" @click="useStarter(card.prompt)">
              <span class="starter-icon">{{ card.icon }}</span>
              <span><strong>{{ card.title }}</strong><small>{{ card.description }}</small></span>
              <b>→</b>
            </button>
          </div>
        </div>

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
        <article v-if="selectedGeneration.streamingAnswer || selectedGeneration.generating" class="message assistant">
          <div class="avatar">AI</div><div class="bubble">
            <div v-if="selectedGeneration.streamingAnswer" class="markdown" v-html="renderMarkdown(selectedGeneration.streamingAnswer)" />
            <div v-else class="thinking"><i></i><i></i><i></i>{{ statusText }}</div>
            <details v-if="selectedGeneration.streamingCitations.length" class="citations" open><summary>引用</summary>
              <a v-for="citation in selectedGeneration.streamingCitations" :key="citation.evidence_id" :href="citation.url || undefined" target="_blank" rel="noopener noreferrer nofollow"><b>[{{ citation.evidence_id }}] {{ citation.title }}</b><span>{{ citation.category }}</span></a>
            </details>
          </div>
        </article>
      </div>

      <div v-if="error || selectedGeneration.error" class="error-banner inline">{{ selectedGeneration.error || error }} <button @click="error = ''; selectedGeneration.error = ''">×</button></div>
      <form class="composer" @submit.prevent="ask">
        <textarea ref="composerInput" v-model="question" :disabled="!selectedId" maxlength="6000" rows="2" placeholder="输入招投标问题，获得专业解答…" @keydown.enter.exact.prevent="ask" />
        <div class="composer-toolbar">
          <div>
            <label class="composer-action">⌕ 附件<input type="file" accept=".pdf,.txt,.md,.docx" :disabled="uploading || !selectedId" @change="upload" /></label>
            <button class="composer-action active" type="button" @click="openKnowledge('enterprise')">▤ 本地知识库已启用</button>
          </div>
          <div><span>Enter 发送 / Shift+Enter 换行</span>
            <button v-if="selectedGeneration.generating" class="stop" type="button" @click="stop">■ 停止</button>
            <button v-else class="send" type="submit" :disabled="!question.trim() || connectionState !== 'connected'">➤ 发送</button>
          </div>
        </div>
      </form>
      <footer>回答仅供辅助判断，请以原始公告、合同与现行法律政策为准。</footer>

      <div v-if="knowledgePanelOpen" class="knowledge-overlay" @click.self="knowledgePanelOpen = false">
        <section class="knowledge-drawer" aria-label="知识库">
          <header><div><small>知识库与文件</small><h2>随时查看检索来源</h2></div><button title="关闭知识库" @click="knowledgePanelOpen = false">×</button></header>
          <div class="knowledge-body">
            <nav>
              <button v-for="library in knowledgeLibraries" :key="library.key" :class="{ active: activeKnowledgeKey === library.key }" @click="selectKnowledge(library.key)">
                <span>{{ library.icon }}</span><b>{{ library.shortLabel }}</b><small v-if="library.key === 'uploads'">{{ files.length }}</small>
              </button>
            </nav>
            <article class="knowledge-content">
              <div class="knowledge-heading"><span>{{ activeKnowledge.icon }}</span><div><h3>{{ activeKnowledge.label }}</h3><p>{{ activeKnowledge.description }}</p></div></div>
              <template v-if="activeKnowledge.key !== 'uploads'">
                <div class="knowledge-connected"><i></i><span><b>本地知识库已连接</b><small>问答时将自动检索并在答案中标注引用</small></span></div>
                <div class="knowledge-fields"><small>主要内容</small><div><span v-for="field in activeKnowledge.fields" :key="field">{{ field }}</span></div></div>
                <section class="knowledge-file-browser">
                  <div class="knowledge-file-title"><span><b>知识库文件</b><small v-if="!knowledgeLoading">共 {{ knowledgeFiles.length }} 个</small></span><label>⌕<input v-model="knowledgeFileQuery" type="search" placeholder="搜索文件" /></label></div>
                  <p v-if="knowledgeLoading" class="knowledge-file-message">正在读取文件列表…</p>
                  <p v-else-if="knowledgeError" class="knowledge-file-message error">{{ knowledgeError }}</p>
                  <p v-else-if="!visibleKnowledgeFiles.length" class="knowledge-file-message">该知识库暂无可显示文件</p>
                  <div v-else class="knowledge-file-list">
                    <a v-for="item in visibleKnowledgeFiles" :key="item.relative_path" :href="api.knowledgeFileUrl(activeKnowledge.key, item.relative_path)" target="_blank" rel="noopener">
                      <span class="uploaded-icon">文</span>
                      <span><b>{{ item.name }}</b><small>{{ item.relative_path }} · {{ fileSize(item.size_bytes) }}</small></span>
                      <strong>打开 ↗</strong>
                    </a>
                  </div>
                  <small v-if="visibleKnowledgeFiles.length < knowledgeFiles.length" class="knowledge-list-limit">当前显示前 {{ visibleKnowledgeFiles.length }} 个文件，可使用搜索缩小范围</small>
                </section>
                <p class="knowledge-note">文件通过只读方式打开；检索和生成仍使用现有统一 RAG 管线。</p>
              </template>
              <template v-else>
                <div v-if="files.length" class="uploaded-list">
                  <div v-for="item in files" :key="item.id">
                    <input v-if="item.status === 'ready'" v-model="selectedFiles" type="checkbox" :value="item.id" />
                    <span class="uploaded-icon">文</span>
                    <span><b>{{ item.original_name }}</b><small>{{ item.status === 'ready' ? `${item.chunk_count} 段 · 已启用` : item.status }}</small></span>
                    <a :href="api.uploadedFileUrl(item.id)" target="_blank" rel="noopener">打开 ↗</a>
                    <button title="删除文件" @click="removeFile(item)">×</button>
                  </div>
                </div>
                <div v-else class="knowledge-empty"><span>⇧</span><b>当前会话还没有上传文件</b><small>可通过顶部“上传文件”或输入框“附件”添加</small></div>
              </template>
            </article>
          </div>
        </section>
      </div>
    </section>
  </main>
</template>
