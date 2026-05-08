import { RGBA, type ScrollBoxRenderable } from '@opentui/core'
import { useKeyboard, useRenderer, useTerminalDimensions } from '@opentui/solid'
import { Match, Switch, createEffect, createMemo, createSignal, onCleanup, onMount, Show, For } from 'solid-js'
import { spawn } from 'node:child_process'
import { readFileSync } from 'node:fs'
import { BackendClient, type BackendEvent } from './backend/client'
import { copyToClipboard } from './clipboard'
import {
  appendEntryWithMerge,
  buildLogEntry,
  classifyTextLog,
  formatPayloadLines,
  type LogEntry,
  type LogKind,
  normalizeLogLines,
} from './logging'
import { HomeRoute } from './routes/HomeRoute'
import { RoutingRoute } from './routes/RoutingRoute'
import { RequirementsRoute } from './routes/RequirementsRoute'
import { ReviewRoute } from './routes/ReviewRoute'
import { DesignRoute } from './routes/DesignRoute'
import { TaskSplitRoute } from './routes/TaskSplitRoute'
import { DevelopmentRoute } from './routes/DevelopmentRoute'
import { OverallReviewRoute } from './routes/OverallReviewRoute'
import { ControlRoute } from './routes/ControlRoute'
import { resolveFooterProgressLine } from './footerProgress'
import { buildHomeAgents } from './homeAgents'
import { promptAllowsBack, resolvePromptBackValue, withPromptBackOption } from './promptBack'
import { resolvePromptResponseTransition } from './promptTransition'
import {
  applyStageChanged,
  EMPTY_STAGE_CURSOR,
  inferBootstrapStatus,
  markTerminalStage,
  shouldAcceptProgressEvent,
  shouldRecoverRunningFromStageSnapshot,
} from './stageStatus'
import { DialogSelect } from './ui/DialogSelect'
import { DialogConfirm } from './ui/DialogConfirm'
import { PromptInputPanel } from './ui/PromptInputPanel'
import type {
  AppSnapshot,
  ArtifactsSnapshot,
  ControlSnapshot,
  DesignSnapshot,
  DevelopmentMilestone,
  FileSnapshot,
  HitlSnapshot,
  HomeAgentItem,
  OverallReviewSnapshot,
  RequirementsSnapshot,
  ReviewSnapshot,
  RoutingSnapshot,
  RunOption,
  DevelopmentSnapshot,
  TaskSplitSnapshot,
  WorkerSnapshot,
} from './types'

type RouteName = 'home' | 'routing' | 'requirements' | 'review' | 'design' | 'task-split' | 'development' | 'overall-review' | 'control'
type ShellFocus = 'content' | 'prompt' | 'log' | 'dialog'

type PromptState = {
  id: string
  promptType: string
  payload: Record<string, unknown>
  draftKey: string
}

type LocalDialogState = {
  kind: 'select'
  title: string
  options: Array<{ value: string; label: string }>
  defaultValue?: string
  onSubmit: (value: string) => void
}

type DocumentPreviewState = {
  path: string
  title: string
}

type StartupOptions = {
  initialRoute?: RouteName
  initialAction?: string
  initialArgv?: string[]
}

type FooterPromptHostProps = {
  active: PromptState
  focused: boolean
  focusToken: string
  height: number
  onSubmit: (value: unknown) => void
}

type FooterStatusHostProps = {
  status: string
  progressLine: string
  pendingHitl: boolean
  height: number
}

type ShellHeights = {
  top: number
  log: number
  footer: number
}

type DialogOverlayProps = {
  helperText?: string
  children: any
}

const PRESENCE_REPORT_DEBOUNCE_MS = 1500

const EMPTY_APP_SNAPSHOT: AppSnapshot = {
  projectDir: '',
  requirementName: '',
  currentAction: '',
  activeRunId: '',
  activeStage: 'idle',
  activeStageLabel: '等待中',
  pendingHitl: false,
  pendingAttention: false,
  pendingAttentionReason: '',
  pendingAttentionSince: '',
  recentArtifacts: [],
  availableRuns: [],
  capabilities: {},
}

const EMPTY_ROUTING_SNAPSHOT: RoutingSnapshot = {
  projectDir: '',
  files: [],
  workers: [],
  statusText: '',
  done: false,
}

const EMPTY_REQUIREMENTS_SNAPSHOT: RequirementsSnapshot = {
  projectDir: '',
  requirementName: '',
  files: [],
  workers: [],
}

const EMPTY_REVIEW_SNAPSHOT: ReviewSnapshot = {
  projectDir: '',
  requirementName: '',
  files: [],
  workers: [],
  blockers: [],
}

const EMPTY_DESIGN_SNAPSHOT: DesignSnapshot = {
  projectDir: '',
  requirementName: '',
  files: [],
  workers: [],
  blockers: [],
}

const EMPTY_TASK_SPLIT_SNAPSHOT: TaskSplitSnapshot = {
  projectDir: '',
  requirementName: '',
  files: [],
  workers: [],
  blockers: [],
}

const EMPTY_DEVELOPMENT_SNAPSHOT: DevelopmentSnapshot = {
  projectDir: '',
  requirementName: '',
  files: [],
  workers: [],
  blockers: [],
  milestones: [],
  currentMilestoneKey: '',
  allTasksCompleted: false,
}

const EMPTY_OVERALL_REVIEW_SNAPSHOT: OverallReviewSnapshot = {
  projectDir: '',
  requirementName: '',
  files: [],
  workers: [],
  blockers: [],
}

const EMPTY_HITL_SNAPSHOT: HitlSnapshot = {
  pending: false,
  questionPath: '',
  answerPath: '',
  summary: '',
}

const EMPTY_ARTIFACTS_SNAPSHOT: ArtifactsSnapshot = {
  items: [],
}

function logEntryPalette(kind: LogKind, hitlRound?: number) {
  if (kind === 'hitl') {
    const hitlPalettes = [
      { border: '#ff9f1c', title: '#ffbf69', body: '#ffe7c2', background: '#261607' },
      { border: '#2ec4b6', title: '#7be7dc', body: '#d9fffb', background: '#06211d' },
      { border: '#e76f51', title: '#ffb4a2', body: '#ffe1da', background: '#2a120d' },
      { border: '#9b5de5', title: '#d0b3ff', body: '#efe5ff', background: '#1d112d' },
    ] as const
    const index = Math.max(0, ((hitlRound ?? 1) - 1) % hitlPalettes.length)
    return hitlPalettes[index]!
  }
  if (kind === 'stage') return { border: '#00d2ff', title: '#00d2ff', body: '#d8f6ff', background: '#07141c' }
  if (kind === 'summary') return { border: '#5cb3ff', title: '#5cb3ff', body: '#dcefff', background: '#0b1726' }
  if (kind === 'runtime') return { border: '#7d8fa1', title: '#b8c2cc', body: '#d0d7de', background: '#11161c' }
  if (kind === 'warning') return { border: '#f7c948', title: '#f7c948', body: '#ffe8a3', background: '#241b00' }
  if (kind === 'error') return { border: '#ff6b6b', title: '#ff6b6b', body: '#ffd0d0', background: '#2a1010' }
  return { border: '#30363d', title: '#c9d1d9', body: '#f5f5f5', background: undefined }
}

function LogEntryCard(props: { entry: LogEntry }) {
  const palette = createMemo(() => logEntryPalette(props.entry.kind, props.entry.hitlRound))
  const framed = createMemo(() => props.entry.kind !== 'plain')
  const bodyLines = createMemo(() => (props.entry.lines.length > 0 ? props.entry.lines : ['']))

  return (
    <box
      flexDirection="column"
      marginTop={props.entry.kind === 'plain' ? 0 : 1}
      marginBottom={1}
      paddingLeft={1}
      paddingRight={1}
      paddingTop={framed() ? 1 : 0}
      paddingBottom={framed() ? 1 : 0}
      borderStyle={framed() ? 'single' : undefined}
      borderColor={framed() ? palette().border : undefined}
      backgroundColor={framed() ? palette().background : undefined}
    >
      <Show when={framed()}>
        <text fg={palette().title}>{props.entry.title}</text>
      </Show>
      <For each={bodyLines()}>{(line) => <text fg={palette().body}>{line || ' '}</text>}</For>
    </box>
  )
}

function buildPromptDraftKey(promptType: string, payload: Record<string, unknown>) {
  const title = String(payload.title ?? payload.prompt_text ?? 'prompt').trim()
  const hitlFingerprint = [
    String(payload.id ?? '').trim(),
    resolveHitlQuestionPath(payload),
    resolveHitlAnswerPath(payload),
  ].find(Boolean)
  const looksLikeHitl = resolvePromptIsHitl(payload) || `${promptType} ${title}`.toLowerCase().includes('hitl')
  if (looksLikeHitl && hitlFingerprint) return `${promptType}:hitl:${hitlFingerprint}`
  return `${promptType}:${title}`
}

function resolveHitlQuestionPath(payload: Record<string, unknown> | null | undefined) {
  return String(payload?.question_path ?? payload?.questionPath ?? '').trim()
}

function resolveHitlAnswerPath(payload: Record<string, unknown> | null | undefined) {
  return String(payload?.answer_path ?? payload?.answerPath ?? '').trim()
}

function resolvePromptIsHitl(payload: Record<string, unknown> | null | undefined) {
  const explicit = payload?.is_hitl ?? payload?.isHitl
  if (explicit !== undefined && explicit !== null) return Boolean(explicit)
  return false
}

function resolvePromptRequiresAttention(promptType: string, payload: Record<string, unknown> | null | undefined) {
  const explicit = payload?.requires_attention ?? payload?.requiresAttention
  if (explicit !== undefined && explicit !== null) return Boolean(explicit)
  if (resolvePromptIsHitl(payload)) return true
  return new Set(['select', 'confirm', 'text', 'multiline']).has(String(promptType || '').trim())
}

function resolvePromptAttentionReason(promptType: string, payload: Record<string, unknown> | null | undefined) {
  if (resolvePromptIsHitl(payload)) return 'hitl'
  const explicit = payload?.attention_reason ?? payload?.attentionReason
  if (explicit !== undefined && explicit !== null && String(explicit).trim()) return String(explicit).trim()
  return String(promptType || 'prompt').trim() || 'prompt'
}

function resolvePreviewPath(payload: Record<string, unknown> | null | undefined) {
  return String(payload?.preview_path ?? payload?.previewPath ?? '').trim()
}

function resolvePreviewTitle(payload: Record<string, unknown> | null | undefined) {
  return String(payload?.preview_title ?? payload?.previewTitle ?? '').trim()
}

function resolvePromptDocumentPath(payload: Record<string, unknown> | null | undefined) {
  return resolvePreviewPath(payload) || resolveHitlQuestionPath(payload)
}

function resolvePromptDocumentTitle(payload: Record<string, unknown> | null | undefined) {
  return resolvePreviewTitle(payload) || String(payload?.title ?? payload?.prompt_text ?? '文档预览').trim() || '文档预览'
}

function isHitlPrompt(active: PromptState | null): boolean {
  if (!active) return false
  if (resolvePromptIsHitl(active.payload)) return true
  const title = String(active.payload.title ?? '')
  const promptText = String(active.payload.prompt_text ?? '')
  return `${active.promptType} ${title} ${promptText}`.toLowerCase().includes('hitl')
}

function buildPromptBackedAttentionSnapshot(active: PromptState | null, fallback: AppSnapshot): Pick<AppSnapshot, 'pendingAttention' | 'pendingAttentionReason' | 'pendingAttentionSince'> {
  if (!active || !resolvePromptRequiresAttention(active.promptType, active.payload)) {
    return {
      pendingAttention: fallback.pendingAttention,
      pendingAttentionReason: fallback.pendingAttentionReason,
      pendingAttentionSince: fallback.pendingAttentionSince,
    }
  }
  return {
    pendingAttention: true,
    pendingAttentionReason: resolvePromptAttentionReason(active.promptType, active.payload),
    pendingAttentionSince: fallback.pendingAttentionSince,
  }
}

function buildPromptBackedHitlSnapshot(active: PromptState | null, fallback: HitlSnapshot): HitlSnapshot {
  if (!isHitlPrompt(active)) return fallback
  const title = String(active?.payload.title ?? active?.payload.prompt_text ?? '').trim()
  return {
    pending: true,
    questionPath: resolveHitlQuestionPath(active?.payload) || fallback.questionPath,
    answerPath: resolveHitlAnswerPath(active?.payload) || fallback.answerPath,
    summary: title || fallback.summary || '存在待处理 HITL',
  }
}

function buildVisibleHitlSnapshot(active: PromptState | null, fallback: HitlSnapshot, appStatus: string): HitlSnapshot {
  const promptBacked = buildPromptBackedHitlSnapshot(active, fallback)
  if (isHitlPrompt(active)) return promptBacked
  if (!fallback.pending) return fallback
  if (String(appStatus || '').trim().toLowerCase() === 'awaiting-input') return fallback
  return EMPTY_HITL_SNAPSHOT
}

function allocateShellHeights(totalHeight: number, footerIsPrompt: boolean, logOpen: boolean): ShellHeights {
  const shellGapRows = logOpen ? 2 : 1
  const available = Math.max(1, totalHeight - shellGapRows)
  const topMin = 8
  const logMin = 8
  const footerMin = footerIsPrompt ? 13 : 6

  if (!logOpen) {
    const preferredTop = Math.floor(available * 0.6)
    let footer = available - Math.max(topMin, preferredTop)
    if (available >= topMin + footerMin && footer < footerMin) footer = footerMin
    footer = Math.max(Math.min(available, footer), Math.min(footerMin, available))
    const top = Math.max(0, available - footer)
    return { top, log: 0, footer }
  }

  if (available < footerMin + logMin + topMin) {
    const footer = Math.min(available, footerMin)
    const remainingAfterFooter = Math.max(0, available - footer)
    const log = Math.min(remainingAfterFooter, logMin)
    const top = Math.max(0, remainingAfterFooter - log)
    return { top, log, footer }
  }

  const preferredTop = Math.floor(available * 0.3)
  const preferredLog = Math.floor(available * 0.5)
  let footer = available - preferredTop - preferredLog
  footer = Math.max(footerMin, footer)
  let remaining = available - footer
  let log = Math.max(logMin, Math.min(preferredLog, remaining - topMin))
  let top = remaining - log
  if (top < topMin) {
    top = topMin
    log = remaining - top
  }
  if (log < logMin) {
    log = logMin
    top = remaining - log
  }
  return { top, log, footer }
}

function isOverlayPromptType(promptType: string): boolean {
  return promptType === 'select' || promptType === 'confirm'
}

function FooterPromptHost(props: FooterPromptHostProps) {
  const title = createMemo(() => String(props.active.payload.title ?? props.active.payload.prompt_text ?? '请输入'))
  const allowBack = createMemo(() => promptAllowsBack(props.active.payload))
  const backValue = createMemo(() => resolvePromptBackValue(props.active.payload))
  const hitlHints = createMemo(() => {
    if (!isHitlPrompt(props.active)) return []
    const questionPath = resolveHitlQuestionPath(props.active.payload)
    const lines: string[] = []
    if (questionPath) lines.push(`问题文件: ${questionPath.split('/').pop() || questionPath}`)
    lines.push('Ctrl+K 查看完整问题')
    if (!questionPath) return lines
    try {
      const previewLines = normalizeLogLines(readFileSync(questionPath, 'utf8'))
        .map((line) => line.trim())
        .filter(Boolean)
        .slice(0, 2)
      lines.push(...previewLines.map((line) => `> ${line.slice(0, 88)}`))
    } catch {
      lines.push('> 问题文件读取失败，请查看日志')
    }
    return lines
  })
  return (
    <box
      borderStyle="single"
      marginLeft={1}
      marginRight={1}
      marginTop={1}
      height={props.height}
      minHeight={props.height}
    >
      <PromptInputPanel
        title={title()}
        defaultValue={String(props.active.payload.default ?? '')}
        draftKey={props.active.draftKey}
        focusToken={props.focusToken}
        focused={props.focused}
        mode={props.active.promptType === 'multiline' ? 'multiline' : 'singleline'}
        hintLines={hitlHints()}
        textareaHeight={isHitlPrompt(props.active) ? 4 : undefined}
        rememberHistory={!isHitlPrompt(props.active)}
        showSubmitHelper={false}
        onBack={allowBack() ? () => void props.onSubmit(backValue()) : undefined}
        onSubmit={(value) => void props.onSubmit(value)}
      />
    </box>
  )
}

function DialogOverlay(props: DialogOverlayProps) {
  const dimensions = useTerminalDimensions()
  const dialogWidth = createMemo(() => Math.max(48, Math.min(dimensions().width - 4, 108)))
  const topPadding = createMemo(() => Math.max(1, Math.floor(dimensions().height / 6)))

  return (
    <box
      position="absolute"
      zIndex={2000}
      top={0}
      left={0}
      width={dimensions().width}
      height={dimensions().height}
      alignItems="center"
      paddingTop={topPadding()}
      backgroundColor={RGBA.fromInts(0, 0, 0, 255)}
    >
      <box width={dialogWidth()} maxWidth={dimensions().width - 2} borderStyle="single" flexDirection="column" paddingTop={1} paddingBottom={1}>
        <Show when={props.helperText}>
          <box paddingLeft={1} paddingRight={1} paddingBottom={1}>
            <text fg="#888888">{props.helperText}</text>
          </box>
        </Show>
        {props.children}
      </box>
    </box>
  )
}

function DialogPromptLayer(props: { active: PromptState; dialogActive: boolean; onSubmit: (value: unknown) => void }) {
  const hasPreview = createMemo(() => Boolean(resolvePreviewPath(props.active.payload)))
  const selectOptions = createMemo(() => withPromptBackOption(
    Array.isArray(props.active.payload.options) ? (props.active.payload.options as { value: string; label: string }[]) : [],
    props.active.payload,
  ))
  return (
    <DialogOverlay helperText={hasPreview() ? '↑/↓ 或 j/k 选择，Enter 提交，Ctrl+K 查看文档' : '↑/↓ 或 j/k 选择，Enter 提交'}>
      <Switch>
        <Match when={props.active.promptType === 'confirm'}>
          <DialogConfirm
            title={String(props.active.payload.prompt_text ?? '请确认')}
            defaultValue={Boolean(props.active.payload.default)}
            active={props.dialogActive}
            allowBack={promptAllowsBack(props.active.payload)}
            backValue={resolvePromptBackValue(props.active.payload)}
            onSubmit={(value) => void props.onSubmit(value)}
          />
        </Match>
        <Match when={true}>
          <DialogSelect
            title={String(props.active.payload.title ?? props.active.payload.prompt_text ?? '请选择')}
            defaultValue={String(props.active.payload.default_value ?? '')}
            options={selectOptions()}
            active={props.dialogActive}
            onSubmit={(value) => void props.onSubmit(value)}
          />
        </Match>
      </Switch>
    </DialogOverlay>
  )
}

function LocalDialogLayer(props: { dialog: LocalDialogState; onClose: () => void }) {
  return (
    <DialogOverlay helperText="↑/↓ 或 j/k 选择，Enter 提交">
      <DialogSelect
        title={props.dialog.title}
        defaultValue={props.dialog.defaultValue}
        options={props.dialog.options}
        active
        onSubmit={(value) => {
          props.dialog.onSubmit(value)
          props.onClose()
        }}
      />
    </DialogOverlay>
  )
}

function DocumentPreviewLayer(props: { preview: DocumentPreviewState; onScrollboxReady?: (value: ScrollBoxRenderable) => void }) {
  const dimensions = useTerminalDimensions()
  const contentLines = createMemo(() => {
    try {
      return normalizeLogLines(readFileSync(props.preview.path, 'utf8'))
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      return [`读取文档失败: ${message}`]
    }
  })
  const viewportHeight = createMemo(() => Math.max(8, Math.floor(dimensions().height * 0.6)))

  return (
    <DialogOverlay helperText="Ctrl+K 关闭，Esc 关闭，↑/↓/PgUp/PgDn/Home/End 滚动">
      <box flexDirection="column" paddingLeft={1} paddingRight={1} width="100%">
        <text>{props.preview.title || '文档预览'}</text>
        <text fg="#888888">{props.preview.path}</text>
        <scrollbox
          ref={(value: ScrollBoxRenderable) => props.onScrollboxReady?.(value)}
          height={viewportHeight()}
          minHeight={viewportHeight()}
          maxHeight={viewportHeight()}
          scrollY
        >
          <box flexDirection="column" paddingTop={1} paddingBottom={1}>
            <For each={contentLines()}>{(line) => <text>{line || ' '}</text>}</For>
          </box>
        </scrollbox>
      </box>
    </DialogOverlay>
  )
}

function FooterStatusHost(props: FooterStatusHostProps) {
  const normalizedStatus = createMemo(() => String(props.status || '').trim().toLowerCase())
  const isError = createMemo(() => props.status === 'error' || props.status === 'failed')
  const isCompleted = createMemo(() => ['completed', 'succeeded', 'done'].includes(normalizedStatus()))
  const isWaitingForHitl = createMemo(() => props.pendingHitl && !isError())
  const isRunning = createMemo(() => !isError() && !isCompleted() && (props.status === 'running' || Boolean(props.progressLine.trim())))
  const isBooting = createMemo(() => !isError() && !isCompleted() && props.status === 'booting')
  const title = createMemo(() => {
    if (isWaitingForHitl()) return '等待人工输入'
    if (isError()) return '运行失败'
    if (isCompleted()) return '已完成'
    return isRunning() ? '运行中' : '等待中'
  })
  const primaryLine = createMemo(() => {
    if (isWaitingForHitl()) return '存在待处理 HITL，请回复问题。'
    if (isError()) return '当前阶段发生错误，请查看上方日志。'
    if (isCompleted()) return '流程已完成，可退出界面。'
    if (props.progressLine.trim()) return props.progressLine
    if (isBooting()) return '⠦ 智能体启动中...'
    if (isRunning()) return '阶段调度中，等待下一个智能体任务。'
    return '当前没有待处理的人类输入，系统空闲。'
  })
  const secondaryLine = createMemo(() => {
    if (isWaitingForHitl()) return '请根据右侧问题文件或日志内容继续回复。'
    if (isError()) return '请查看失败日志，修正配置后重新发起当前阶段。'
    if (isCompleted()) return '所有阶段已收尾，结果文件已写入项目目录。'
    return isRunning() ? '系统执行智能体任务中，输入框会在需要人类交互时自动恢复。' : '等待下一次人类输入或阶段调度。'
  })

  return (
    <box
      borderStyle="single"
      marginLeft={1}
      marginRight={1}
      marginTop={1}
      height={props.height}
      minHeight={props.height}
    >
      <scrollbox flexGrow={1} scrollY>
        <box flexDirection="column" gap={1} paddingLeft={1} paddingRight={1} paddingTop={1} paddingBottom={1}>
          <text>{title()}</text>
          <text fg={isRunning() ? '#00d2ff' : '#888888'}>{primaryLine()}</text>
          <text fg="#888888">{secondaryLine()}</text>
        </box>
      </scrollbox>
    </box>
  )
}

function normalizePayload<T>(value: unknown): T {
  if (value && typeof value === 'object' && 'result' in (value as Record<string, unknown>)) {
    return ((value as Record<string, unknown>).result ?? {}) as T
  }
  return ((value as Record<string, unknown>) ?? {}) as T
}

function normalizeFileSnapshot(value: Record<string, unknown>): FileSnapshot {
  return {
    label: String(value.label ?? ''),
    path: String(value.path ?? ''),
    exists: Boolean(value.exists),
    updatedAt: String(value.updated_at ?? value.updatedAt ?? ''),
    summary: String(value.summary ?? ''),
  }
}

function normalizeWorkerSnapshot(value: Record<string, unknown>): WorkerSnapshot {
  const rawArtifactPaths = (value.artifact_paths ?? value.artifactPaths) as unknown
  return {
    index: Number(value.index ?? 0) || undefined,
    workerId: String(value.worker_id ?? value.workerId ?? ''),
    workDir: String(value.work_dir ?? value.workDir ?? ''),
    sessionName: String(value.session_name ?? value.sessionName ?? ''),
    status: String(value.status ?? ''),
    workflowStage: String(value.workflow_stage ?? value.workflowStage ?? ''),
    agentState: String(value.agent_state ?? value.agentState ?? ''),
    healthStatus: String(value.health_status ?? value.healthStatus ?? ''),
    currentTaskRuntimeStatus: String(value.current_task_runtime_status ?? value.currentTaskRuntimeStatus ?? ''),
    retryCount: Number(value.retry_count ?? value.retryCount ?? 0),
    note: String(value.note ?? ''),
    transcriptPath: String(value.transcript_path ?? value.transcriptPath ?? ''),
    turnStatusPath: String(value.turn_status_path ?? value.turnStatusPath ?? ''),
    questionPath: String(value.question_path ?? value.questionPath ?? ''),
    answerPath: String(value.answer_path ?? value.answerPath ?? ''),
    artifactPaths: Array.isArray(rawArtifactPaths)
      ? rawArtifactPaths.map((item) => String(item))
      : [],
    sessionExists: value.session_exists === undefined ? undefined : Boolean(value.session_exists),
    lastHeartbeatAt: String(value.last_heartbeat_at ?? value.lastHeartbeatAt ?? ''),
    updatedAt: String(value.updated_at ?? value.updatedAt ?? ''),
  }
}

function normalizeControlSnapshot(payload: Record<string, unknown>): ControlSnapshot {
  return {
    supported: Boolean(payload.supported),
    controlId: String(payload.control_id ?? payload.controlId ?? ''),
    runId: String(payload.run_id ?? payload.runId ?? ''),
    runtimeDir: String(payload.runtime_dir ?? payload.runtimeDir ?? ''),
    statusText: String(payload.status_text ?? payload.statusText ?? ''),
    helpText: String(payload.help_text ?? payload.helpText ?? ''),
    workers: Array.isArray(payload.workers)
      ? (payload.workers as Record<string, unknown>[]).map(normalizeWorkerSnapshot)
      : Array.isArray(payload.rows)
        ? (payload.rows as Record<string, unknown>[]).map(normalizeWorkerSnapshot)
        : [],
    done: Boolean(payload.done),
    canSwitchRuns: Boolean(payload.can_switch_runs ?? payload.canSwitchRuns),
    finalSummary: String(payload.final_summary ?? payload.finalSummary ?? ''),
    transitionText: String(payload.transition_text ?? payload.transitionText ?? ''),
  }
}

function normalizeRoutingSnapshot(payload: Record<string, unknown>): RoutingSnapshot {
  return {
    projectDir: String(payload.project_dir ?? payload.projectDir ?? ''),
    files: Array.isArray(payload.files) ? (payload.files as Record<string, unknown>[]).map(normalizeFileSnapshot) : [],
    workers: Array.isArray(payload.workers) ? (payload.workers as Record<string, unknown>[]).map(normalizeWorkerSnapshot) : [],
    statusText: String(payload.status_text ?? payload.statusText ?? ''),
    done: Boolean(payload.done),
  }
}

function normalizeRequirementsSnapshot(payload: Record<string, unknown>): RequirementsSnapshot {
  return {
    projectDir: String(payload.project_dir ?? payload.projectDir ?? ''),
    requirementName: String(payload.requirement_name ?? payload.requirementName ?? ''),
    files: Array.isArray(payload.files) ? (payload.files as Record<string, unknown>[]).map(normalizeFileSnapshot) : [],
    workers: Array.isArray(payload.workers) ? (payload.workers as Record<string, unknown>[]).map(normalizeWorkerSnapshot) : [],
  }
}

function normalizeReviewSnapshot(payload: Record<string, unknown>): ReviewSnapshot {
  return {
    projectDir: String(payload.project_dir ?? payload.projectDir ?? ''),
    requirementName: String(payload.requirement_name ?? payload.requirementName ?? ''),
    files: Array.isArray(payload.files) ? (payload.files as Record<string, unknown>[]).map(normalizeFileSnapshot) : [],
    workers: Array.isArray(payload.workers) ? (payload.workers as Record<string, unknown>[]).map(normalizeWorkerSnapshot) : [],
    blockers: Array.isArray(payload.blockers) ? payload.blockers.map((item) => String(item)) : [],
  }
}

function normalizeDesignSnapshot(payload: Record<string, unknown>): DesignSnapshot {
  return {
    projectDir: String(payload.project_dir ?? payload.projectDir ?? ''),
    requirementName: String(payload.requirement_name ?? payload.requirementName ?? ''),
    files: Array.isArray(payload.files) ? (payload.files as Record<string, unknown>[]).map(normalizeFileSnapshot) : [],
    workers: Array.isArray(payload.workers) ? (payload.workers as Record<string, unknown>[]).map(normalizeWorkerSnapshot) : [],
    blockers: Array.isArray(payload.blockers) ? payload.blockers.map((item) => String(item)) : [],
  }
}

function normalizeTaskSplitSnapshot(payload: Record<string, unknown>): TaskSplitSnapshot {
  return {
    projectDir: String(payload.project_dir ?? payload.projectDir ?? ''),
    requirementName: String(payload.requirement_name ?? payload.requirementName ?? ''),
    files: Array.isArray(payload.files) ? (payload.files as Record<string, unknown>[]).map(normalizeFileSnapshot) : [],
    workers: Array.isArray(payload.workers) ? (payload.workers as Record<string, unknown>[]).map(normalizeWorkerSnapshot) : [],
    blockers: Array.isArray(payload.blockers) ? payload.blockers.map((item) => String(item)) : [],
  }
}

function normalizeDevelopmentSnapshot(payload: Record<string, unknown>): DevelopmentSnapshot {
  return {
    projectDir: String(payload.project_dir ?? payload.projectDir ?? ''),
    requirementName: String(payload.requirement_name ?? payload.requirementName ?? ''),
    files: Array.isArray(payload.files) ? (payload.files as Record<string, unknown>[]).map(normalizeFileSnapshot) : [],
    workers: Array.isArray(payload.workers) ? (payload.workers as Record<string, unknown>[]).map(normalizeWorkerSnapshot) : [],
    blockers: Array.isArray(payload.blockers) ? payload.blockers.map((item) => String(item)) : [],
    milestones: Array.isArray(payload.milestones)
      ? payload.milestones.map((item) => ({
        key: String((item as Record<string, unknown>).key ?? ''),
        completed: Boolean((item as Record<string, unknown>).completed),
        tasks: Array.isArray((item as Record<string, unknown>).tasks)
          ? ((item as Record<string, unknown>).tasks as Record<string, unknown>[]).map((task) => ({
            key: String(task.key ?? ''),
            completed: Boolean(task.completed),
          }))
          : [],
      } satisfies DevelopmentMilestone))
      : [],
    currentMilestoneKey: String(payload.current_milestone_key ?? payload.currentMilestoneKey ?? ''),
    allTasksCompleted: Boolean(payload.all_tasks_completed ?? payload.allTasksCompleted),
  }
}

function normalizeOverallReviewSnapshot(payload: Record<string, unknown>): OverallReviewSnapshot {
  return {
    projectDir: String(payload.project_dir ?? payload.projectDir ?? ''),
    requirementName: String(payload.requirement_name ?? payload.requirementName ?? ''),
    files: Array.isArray(payload.files) ? (payload.files as Record<string, unknown>[]).map(normalizeFileSnapshot) : [],
    workers: Array.isArray(payload.workers) ? (payload.workers as Record<string, unknown>[]).map(normalizeWorkerSnapshot) : [],
    blockers: Array.isArray(payload.blockers) ? payload.blockers.map((item) => String(item)) : [],
  }
}

function buildDevelopmentStatusLines(snapshot: DevelopmentSnapshot): string[] {
  if (snapshot.milestones.length === 0) return []
  const lines: string[] = [`current_milestone: ${snapshot.currentMilestoneKey || '(none)'}`]
  for (const milestone of snapshot.milestones) {
    const isCurrent = milestone.key === snapshot.currentMilestoneKey && Boolean(snapshot.currentMilestoneKey)
    lines.push(`${isCurrent ? '> ' : '  '}${milestone.completed ? '☑' : '☐'} ${milestone.key}${isCurrent ? ' (当前)' : ''}`)
    for (const task of milestone.tasks) {
      lines.push(`    ${task.completed ? '☑' : '☐'} ${task.key}`)
    }
  }
  return lines
}

function normalizeHitlSnapshot(payload: Record<string, unknown>): HitlSnapshot {
  return {
    pending: Boolean(payload.pending),
    questionPath: String(payload.question_path ?? payload.questionPath ?? ''),
    answerPath: String(payload.answer_path ?? payload.answerPath ?? ''),
    summary: String(payload.summary ?? ''),
  }
}

function normalizeArtifactsSnapshot(payload: Record<string, unknown>): ArtifactsSnapshot {
  return {
    items: Array.isArray(payload.items)
      ? payload.items.map((item) => ({
        path: String((item as Record<string, unknown>).path ?? ''),
        updatedAt: String((item as Record<string, unknown>).updated_at ?? ''),
        summary: String((item as Record<string, unknown>).summary ?? ''),
      }))
      : [],
  }
}

function normalizeAppSnapshot(payload: Record<string, unknown>): AppSnapshot {
  return {
    projectDir: String(payload.project_dir ?? payload.projectDir ?? ''),
    requirementName: String(payload.requirement_name ?? payload.requirementName ?? ''),
    currentAction: String(payload.current_action ?? payload.currentAction ?? ''),
    activeRunId: String(payload.active_run_id ?? payload.activeRunId ?? ''),
    activeStage: String(payload.active_stage ?? payload.activeStage ?? 'idle'),
    activeStageLabel: String(payload.active_stage_label ?? payload.activeStageLabel ?? '等待中'),
    pendingHitl: Boolean(payload.pending_hitl ?? payload.pendingHitl),
    pendingAttention: Boolean(payload.pending_attention ?? payload.pendingAttention),
    pendingAttentionReason: String(payload.pending_attention_reason ?? payload.pendingAttentionReason ?? ''),
    pendingAttentionSince: String(payload.pending_attention_since ?? payload.pendingAttentionSince ?? ''),
    recentArtifacts: Array.isArray(payload.recent_artifacts)
      ? payload.recent_artifacts.map((item) => ({
        path: String((item as Record<string, unknown>).path ?? ''),
        updatedAt: String((item as Record<string, unknown>).updated_at ?? ''),
        summary: String((item as Record<string, unknown>).summary ?? ''),
      }))
      : [],
    availableRuns: Array.isArray(payload.available_runs)
      ? (payload.available_runs as Record<string, unknown>[]).map((item) => ({
        runId: String(item.run_id ?? item.runId ?? ''),
        runtimeDir: String(item.runtime_dir ?? item.runtimeDir ?? ''),
        projectDir: String(item.project_dir ?? item.projectDir ?? ''),
        status: String(item.status ?? ''),
        updatedAt: String(item.updated_at ?? item.updatedAt ?? ''),
        workerCount: Number(item.worker_count ?? item.workerCount ?? 0),
        failedCount: Number(item.failed_count ?? item.failedCount ?? 0),
      } satisfies RunOption))
      : [],
    capabilities: (payload.capabilities as Record<string, unknown>) ?? {},
  }
}

async function copyCurrentSelection(renderer: ReturnType<typeof useRenderer>): Promise<boolean> {
  const text = renderer.getSelection?.()?.getSelectedText?.()
  if (!text) return false
  try {
    const copied = await copyToClipboard(text)
    if (copied) renderer.clearSelection?.()
    return copied
  } catch {
    return false
  }
}

async function runInheritedCommand(renderer: ReturnType<typeof useRenderer>, command: string[], cwd?: string) {
  renderer.suspend()
  try {
    const exitCode = await new Promise<number>((resolve, reject) => {
      const child = spawn(command[0]!, command.slice(1), {
        cwd,
        stdio: 'inherit',
      })
      child.on('exit', (code) => resolve(code ?? 0))
      child.on('error', reject)
    })
    if (exitCode !== 0) {
      throw new Error(`${command.join(' ')} 退出码异常: ${exitCode}`)
    }
  } finally {
    renderer.resume()
  }
}

const client = new BackendClient()

export function App(props: StartupOptions) {
  const renderer = useRenderer()
  const dimensions = useTerminalDimensions()
  const [route, setRoute] = createSignal<RouteName>('home')
  const [logs, setLogs] = createSignal<LogEntry[]>([])
  const [progress, setProgress] = createSignal<Record<string, string>>({})
  const [footerSpinnerTick, setFooterSpinnerTick] = createSignal(0)
  const [status, setStatus] = createSignal('booting')
  const [stageCursor, setStageCursor] = createSignal(EMPTY_STAGE_CURSOR)
  const [prompt, setPrompt] = createSignal<PromptState | null>(null)
  const [localDialog, setLocalDialog] = createSignal<LocalDialogState | null>(null)
  const [bootstrap, setBootstrap] = createSignal<Record<string, unknown>>({})
  const [appSnapshot, setAppSnapshot] = createSignal<AppSnapshot>(EMPTY_APP_SNAPSHOT)
  const [routingSnapshot, setRoutingSnapshot] = createSignal<RoutingSnapshot>(EMPTY_ROUTING_SNAPSHOT)
  const [requirementsSnapshot, setRequirementsSnapshot] = createSignal<RequirementsSnapshot>(EMPTY_REQUIREMENTS_SNAPSHOT)
  const [reviewSnapshot, setReviewSnapshot] = createSignal<ReviewSnapshot>(EMPTY_REVIEW_SNAPSHOT)
  const [designSnapshot, setDesignSnapshot] = createSignal<DesignSnapshot>(EMPTY_DESIGN_SNAPSHOT)
  const [taskSplitSnapshot, setTaskSplitSnapshot] = createSignal<TaskSplitSnapshot>(EMPTY_TASK_SPLIT_SNAPSHOT)
  const [developmentSnapshot, setDevelopmentSnapshot] = createSignal<DevelopmentSnapshot>(EMPTY_DEVELOPMENT_SNAPSHOT)
  const [overallReviewSnapshot, setOverallReviewSnapshot] = createSignal<OverallReviewSnapshot>(EMPTY_OVERALL_REVIEW_SNAPSHOT)
  const [controlSnapshot, setControlSnapshot] = createSignal<ControlSnapshot | null>(null)
  const [controlSelectedIndex, setControlSelectedIndex] = createSignal(0)
  const [hitlSnapshot, setHitlSnapshot] = createSignal<HitlSnapshot>(EMPTY_HITL_SNAPSHOT)
  const [artifactsSnapshot, setArtifactsSnapshot] = createSignal<ArtifactsSnapshot>(EMPTY_ARTIFACTS_SNAPSHOT)
  const displayHitlSnapshot = createMemo(() => buildVisibleHitlSnapshot(prompt(), hitlSnapshot(), status()))
  const displayAppSnapshot = createMemo<AppSnapshot>(() => {
    const base = appSnapshot()
    const visibleHitl = displayHitlSnapshot()
    return {
      ...base,
      pendingHitl: visibleHitl.pending || (base.pendingHitl && status() === 'awaiting-input'),
    }
  })
  const homeAgents = createMemo<HomeAgentItem[]>(() =>
    buildHomeAgents(
      [
        { source: 'control', workers: controlSnapshot()?.workers ?? [] },
        { source: 'routing', workers: routingSnapshot().workers },
        { source: 'requirements', workers: requirementsSnapshot().workers },
        { source: 'review', workers: reviewSnapshot().workers },
        { source: 'design', workers: designSnapshot().workers },
        { source: 'task-split', workers: taskSplitSnapshot().workers },
        { source: 'development', workers: developmentSnapshot().workers },
        { source: 'overall-review', workers: overallReviewSnapshot().workers },
      ],
      displayAppSnapshot().activeStage,
    ),
  )
  const footerPrompt = createMemo<PromptState | null>(() => {
    const active = prompt()
    if (!active || isOverlayPromptType(active.promptType)) return null
    return active
  })
  const dialogPrompt = createMemo<PromptState | null>(() => {
    const active = prompt()
    if (!active || !isOverlayPromptType(active.promptType)) return null
    return active
  })
  const [showLogs, setShowLogs] = createSignal(false)
  const [logAutoFollow, setLogAutoFollow] = createSignal(true)
  const [logOpenGeneration, setLogOpenGeneration] = createSignal(0)
  const [shellFocus, setShellFocus] = createSignal<ShellFocus>('content')
  const [focusBeforeLog, setFocusBeforeLog] = createSignal<ShellFocus>('content')
  const [documentPreviewOpen, setDocumentPreviewOpen] = createSignal(false)
  const lastPresenceAtByReason = new Map<string, number>()
  const footerPromptFocusToken = createMemo(() => `${footerPrompt()?.id ?? 'no-prompt'}:${shellFocus()}`)
  const shellHeights = createMemo(() => allocateShellHeights(dimensions().height, Boolean(footerPrompt()), showLogs()))
  let logScrollbox: ScrollBoxRenderable | undefined
  let documentPreviewScrollbox: ScrollBoxRenderable | undefined
  let controlPollInFlight = false
  const promptPreview = createMemo<DocumentPreviewState | null>(() => {
    const active = dialogPrompt() ?? footerPrompt()
    if (!active) return null
    const path = resolvePromptDocumentPath(active.payload)
    if (!path) return null
    return {
      path,
      title: resolvePromptDocumentTitle(active.payload),
    }
  })
  const activeDocumentPreview = createMemo<DocumentPreviewState | null>(() => (documentPreviewOpen() ? promptPreview() : null))

  const currentProgress = createMemo(() => Object.values(progress()).join(' | '))
  const footerProgressLine = createMemo(() =>
    resolveFooterProgressLine(
      {
        status: status(),
        route: route(),
        activeStage: displayAppSnapshot().activeStage,
        activeStageLabel: displayAppSnapshot().activeStageLabel,
        routingWorkers: routingSnapshot().workers,
        requirementsWorkers: requirementsSnapshot().workers,
        reviewWorkers: reviewSnapshot().workers,
        designWorkers: designSnapshot().workers,
        taskSplitWorkers: taskSplitSnapshot().workers,
        developmentWorkers: developmentSnapshot().workers,
        overallReviewWorkers: overallReviewSnapshot().workers,
      },
      currentProgress(),
      footerSpinnerTick(),
    ),
  )
  const developmentStatusLines = createMemo(() => {
    const activeStage = displayAppSnapshot().activeStage
    const isDevelopmentStage = route() === 'development' || activeStage === 'stage.a07.start'
    if (!isDevelopmentStage) return []
    return buildDevelopmentStatusLines(developmentSnapshot())
  })
  const statusLines = createMemo(() => {
    const lines = ['运行状态', `status: ${status()}`]
    const snapshot = displayAppSnapshot()
    if (footerProgressLine()) lines.push(footerProgressLine())
    if (Object.keys(bootstrap()).length > 0) lines.push(`python: ${String(bootstrap().python_path ?? '')}`)
    if (snapshot.currentAction) lines.push(`action: ${snapshot.currentAction}`)
    if (snapshot.activeStage || snapshot.activeStageLabel) {
      lines.push(`stage: ${snapshot.activeStageLabel || snapshot.activeStage}${snapshot.activeStage ? ` (${snapshot.activeStage})` : ''}`)
    }
    lines.push(`active_run: ${snapshot.activeRunId || '(none)'}`)
    lines.push(`pending_hitl: ${snapshot.pendingHitl ? 'yes' : 'no'}`)
    lines.push(`pending_attention: ${snapshot.pendingAttention ? 'yes' : 'no'}`)
    lines.push(...developmentStatusLines())
    if (showLogs()) lines.push('log_open: yes')
    return lines
  })

  const appendLog = (text: string, sourceEventType = 'log.append', payload: Record<string, unknown> = {}) => {
    const entry = classifyTextLog(text, sourceEventType, payload)
    setLogs((prev) => appendEntryWithMerge(prev, entry))
  }

  const appendStructuredLog = (entry: LogEntry) => {
    setLogs((prev) => appendEntryWithMerge(prev, entry))
  }

  const appendRuntimeLog = (action: string, payload: Record<string, unknown>) => {
    appendStructuredLog(
      buildLogEntry({
        kind: 'runtime',
        sourceEventType: action,
        title: `运行时事件 · ${action}`,
        lines: formatPayloadLines(payload),
      }),
    )
  }

  const scrollLogBy = (delta: number) => {
    if (!logScrollbox || logScrollbox.isDestroyed) return
    logScrollbox.scrollTop += delta
    setLogAutoFollow(false)
  }

  const scrollLogTo = (position: 'top' | 'bottom') => {
    if (!logScrollbox || logScrollbox.isDestroyed) return false
    if (position === 'top') {
      logScrollbox.scrollTop = 0
      setLogAutoFollow(false)
      return true
    }
    logScrollbox.scrollTop = logScrollbox.scrollHeight
    setLogAutoFollow(true)
    return true
  }

  const scrollDocumentPreviewBy = (delta: number) => {
    if (!documentPreviewScrollbox || documentPreviewScrollbox.isDestroyed) return
    documentPreviewScrollbox.scrollTop += delta
  }

  const scrollDocumentPreviewTo = (position: 'top' | 'bottom') => {
    if (!documentPreviewScrollbox || documentPreviewScrollbox.isDestroyed) return
    if (position === 'top') {
      documentPreviewScrollbox.scrollTop = 0
      return
    }
    documentPreviewScrollbox.scrollTop = documentPreviewScrollbox.scrollHeight
  }

  const resolveFocusAfterLogClose = (): ShellFocus => {
    const previous = focusBeforeLog()
    if (previous === 'prompt' && footerPrompt()) return 'prompt'
    return 'content'
  }

  const openLogs = () => {
    if (localDialog() || dialogPrompt()) return
    setFocusBeforeLog(shellFocus())
    setLogOpenGeneration((prev) => prev + 1)
    setShowLogs(true)
    setLogAutoFollow(true)
  }

  const closeLogs = () => {
    setShowLogs(false)
    setShellFocus(footerPrompt() ? 'prompt' : resolveFocusAfterLogClose())
  }

  const toggleLogs = () => {
    if (showLogs()) {
      closeLogs()
      return
    }
    openLogs()
  }

  const handleLogNavigation = (event: { name: string; ctrl?: boolean; preventDefault: () => void }) => {
    if (!showLogs() || !event.ctrl) return false
    if (event.name === 'up') {
      event.preventDefault()
      scrollLogBy(-1)
      return true
    }
    if (event.name === 'down') {
      event.preventDefault()
      scrollLogBy(1)
      return true
    }
    if (event.name === 'pageup') {
      event.preventDefault()
      scrollLogBy(-12)
      return true
    }
    if (event.name === 'pagedown') {
      event.preventDefault()
      scrollLogBy(12)
      return true
    }
    if (event.name === 'home') {
      event.preventDefault()
      scrollLogTo('top')
      return true
    }
    if (event.name === 'end') {
      event.preventDefault()
      scrollLogTo('bottom')
      return true
    }
    return false
  }

  const clampSelectedWorker = (nextSnapshot: ControlSnapshot | null) => {
    if (!nextSnapshot || nextSnapshot.workers.length === 0) {
      setControlSelectedIndex(0)
      return
    }
    setControlSelectedIndex((prev) => Math.min(prev, nextSnapshot.workers.length - 1))
  }

  const applyBootstrapSnapshots = (payload: Record<string, unknown>) => {
    const snapshots = (payload.snapshots as Record<string, unknown>) ?? {}
    if (snapshots.app && typeof snapshots.app === 'object') setAppSnapshot(normalizeAppSnapshot(snapshots.app as Record<string, unknown>))
    if (snapshots.stages && typeof snapshots.stages === 'object') {
      const stageSnapshots = snapshots.stages as Record<string, unknown>
      if (stageSnapshots.routing && typeof stageSnapshots.routing === 'object') {
        setRoutingSnapshot(normalizeRoutingSnapshot(stageSnapshots.routing as Record<string, unknown>))
      }
      if (stageSnapshots.requirements && typeof stageSnapshots.requirements === 'object') {
        setRequirementsSnapshot(normalizeRequirementsSnapshot(stageSnapshots.requirements as Record<string, unknown>))
      }
      if (stageSnapshots.review && typeof stageSnapshots.review === 'object') {
        setReviewSnapshot(normalizeReviewSnapshot(stageSnapshots.review as Record<string, unknown>))
      }
      if (stageSnapshots.design && typeof stageSnapshots.design === 'object') {
        setDesignSnapshot(normalizeDesignSnapshot(stageSnapshots.design as Record<string, unknown>))
      }
      if (stageSnapshots['task-split'] && typeof stageSnapshots['task-split'] === 'object') {
        setTaskSplitSnapshot(normalizeTaskSplitSnapshot(stageSnapshots['task-split'] as Record<string, unknown>))
      }
      if (stageSnapshots.development && typeof stageSnapshots.development === 'object') {
        setDevelopmentSnapshot(normalizeDevelopmentSnapshot(stageSnapshots.development as Record<string, unknown>))
      }
      if (stageSnapshots['overall-review'] && typeof stageSnapshots['overall-review'] === 'object') {
        setOverallReviewSnapshot(normalizeOverallReviewSnapshot(stageSnapshots['overall-review'] as Record<string, unknown>))
      }
    }
    if (snapshots.control && typeof snapshots.control === 'object') {
      const nextControl = normalizeControlSnapshot(snapshots.control as Record<string, unknown>)
      setControlSnapshot(nextControl)
      clampSelectedWorker(nextControl)
    }
    if (snapshots.hitl && typeof snapshots.hitl === 'object') setHitlSnapshot(normalizeHitlSnapshot(snapshots.hitl as Record<string, unknown>))
    if (snapshots.artifacts && typeof snapshots.artifacts === 'object') {
      setArtifactsSnapshot(normalizeArtifactsSnapshot(snapshots.artifacts as Record<string, unknown>))
    }
  }

  const reportPresence = (reason: string, focus: ShellFocus = shellFocus()) => {
    const normalizedReason = String(reason || '').trim()
    if (!normalizedReason) return
    const now = Date.now()
    const lastAt = lastPresenceAtByReason.get(normalizedReason) ?? 0
    if (now - lastAt < PRESENCE_REPORT_DEBOUNCE_MS) return
    lastPresenceAtByReason.set(normalizedReason, now)
    client.sendPresence(normalizedReason, focus)
  }

  const handleEvent = (event: BackendEvent) => {
    if (event.type === 'log.append') {
      appendLog(String(event.payload.text ?? ''), event.type, event.payload)
      return
    }
    if (event.type === 'progress.start') {
      if (!shouldAcceptProgressEvent(stageCursor(), event.payload)) return
      return
    }
    if (event.type === 'progress.update') {
      if (!shouldAcceptProgressEvent(stageCursor(), event.payload)) return
      const id = String(event.payload.id ?? '')
      const line = String(event.payload.line ?? '')
      setProgress((prev) => ({ ...prev, [id]: line }))
      return
    }
    if (event.type === 'progress.stop') {
      if (!shouldAcceptProgressEvent(stageCursor(), event.payload)) return
      const id = String(event.payload.id ?? '')
      setProgress((prev) => {
        const next = { ...prev }
        delete next[id]
        return next
      })
      return
    }
    if (event.type === 'prompt.request') {
      const promptType = String(event.payload.prompt_type ?? 'text')
      const nextFocus = isOverlayPromptType(promptType) ? 'dialog' : 'prompt'
      setPrompt({
        id: String(event.payload.id ?? ''),
        promptType,
        payload: event.payload,
        draftKey: buildPromptDraftKey(promptType, event.payload),
      })
      setStatus('awaiting-input')
      setShellFocus(nextFocus)
      reportPresence('prompt-open', nextFocus)
      return
    }
    if (event.type === 'stage.changed') {
      const transition = applyStageChanged(stageCursor(), event.payload)
      if (!transition.accepted) return
      setStageCursor(transition.cursor)
      if (transition.status !== 'running') setProgress({})
      setStatus(transition.status)
      appendStructuredLog(
        buildLogEntry({
          kind: 'stage',
          sourceEventType: event.type,
          title: String(event.payload.action ?? '阶段切换'),
          lines: [`status: ${transition.status}`],
        }),
      )
      return
    }
    if (event.type === 'snapshot.app') {
      setAppSnapshot(normalizeAppSnapshot(event.payload))
      return
    }
    if (event.type === 'snapshot.stage') {
      const stageRoute = String(event.payload.route ?? '')
      const stageSnapshot = (event.payload.snapshot as Record<string, unknown>) ?? {}
      if (stageRoute === 'routing') setRoutingSnapshot(normalizeRoutingSnapshot(stageSnapshot))
      if (stageRoute === 'requirements') setRequirementsSnapshot(normalizeRequirementsSnapshot(stageSnapshot))
      if (stageRoute === 'review') setReviewSnapshot(normalizeReviewSnapshot(stageSnapshot))
      if (stageRoute === 'design') setDesignSnapshot(normalizeDesignSnapshot(stageSnapshot))
      if (stageRoute === 'task-split') setTaskSplitSnapshot(normalizeTaskSplitSnapshot(stageSnapshot))
      if (stageRoute === 'development') setDevelopmentSnapshot(normalizeDevelopmentSnapshot(stageSnapshot))
      if (stageRoute === 'overall-review') setOverallReviewSnapshot(normalizeOverallReviewSnapshot(stageSnapshot))
      const activeStage = displayAppSnapshot().activeStage !== 'idle' ? displayAppSnapshot().activeStage : stageCursor().activeAction
      const hasPendingInput = Boolean(prompt()) || Boolean(hitlSnapshot().pending)
      if (shouldRecoverRunningFromStageSnapshot(status(), activeStage, stageRoute, stageSnapshot, hasPendingInput)) {
        setStatus('running')
      }
      return
    }
    if (event.type === 'snapshot.control') {
      const nextControl = normalizeControlSnapshot(event.payload)
      setControlSnapshot(nextControl)
      clampSelectedWorker(nextControl)
      return
    }
    if (event.type === 'snapshot.hitl') {
      setHitlSnapshot(normalizeHitlSnapshot(event.payload))
      return
    }
    if (event.type === 'snapshot.artifacts') {
      setArtifactsSnapshot(normalizeArtifactsSnapshot(event.payload))
      return
    }
    if (event.type === 'error') {
      setProgress({})
      setStageCursor((prev) => markTerminalStage(prev))
      appendStructuredLog(
        buildLogEntry({
          kind: 'error',
          sourceEventType: event.type,
          title: '错误',
          lines: normalizeLogLines(String(event.payload.message ?? 'unknown error')),
        }),
      )
      setStatus('error')
      return
    }
    appendRuntimeLog(event.type, event.payload)
  }

  const requestAction = async (action: string, payload: Record<string, unknown> = {}, quiet = false) => {
    try {
      const raw = await client.request(action, payload)
      const result = normalizePayload<Record<string, unknown>>(raw)
      if (action === 'control.b01.open' || action.startsWith('worker.') || action === 'run.resume') {
        const nextControl = normalizeControlSnapshot(result)
        setControlSnapshot(nextControl)
        clampSelectedWorker(nextControl)
      }
      if (!quiet && Object.keys(result).length > 0) {
        appendRuntimeLog(action, result)
      }
      return result
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      setProgress({})
      setStageCursor((prev) => markTerminalStage(prev))
      appendStructuredLog(
        buildLogEntry({
          kind: 'error',
          sourceEventType: action,
          title: `请求失败 · ${action}`,
          lines: [message],
        }),
      )
      setStatus('error')
      throw error
    }
  }

  onMount(async () => {
    const spinnerTimer = setInterval(() => {
      setFooterSpinnerTick((prev) => prev + 1)
    }, 500)
    let unsubscribeBackend: (() => void) | undefined
    onCleanup(() => clearInterval(spinnerTimer))
    onCleanup(() => {
      unsubscribeBackend?.()
      client.stop()
    })
    try {
      await client.start()
      unsubscribeBackend = client.subscribe(handleEvent)
      const result = (await client.bootstrap()) as Record<string, unknown>
      setBootstrap(result)
      applyBootstrapSnapshots(result)
      setStatus(inferBootstrapStatus(result))
      if (props.initialRoute) {
        setRoute(props.initialRoute)
      }
      if (props.initialAction) {
        try {
          await requestAction(props.initialAction, { argv: props.initialArgv ?? [] }, true)
        } catch {
          return
        }
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      appendLog(`ERROR: ${message}`)
      setProgress({})
      setStageCursor((prev) => markTerminalStage(prev))
      setStatus('error')
    }
  })

  onMount(() => {
    const timer = setInterval(() => {
      if (route() !== 'control') return
      const snapshot = controlSnapshot()
      const controlId = String(snapshot?.controlId ?? '').trim()
      if (!controlId || controlPollInFlight) return
      controlPollInFlight = true
      void requestAction('control.b01.open', { control_id: controlId }, true).catch(() => undefined).finally(() => {
        controlPollInFlight = false
      })
    }, 2000)
    onCleanup(() => clearInterval(timer))
  })

  createEffect(() => {
    if (!showLogs() || !logAutoFollow()) return
    logs()
    queueMicrotask(() => scrollLogTo('bottom'))
  })

  createEffect(() => {
    const preview = activeDocumentPreview()
    if (!preview) return
    queueMicrotask(() => scrollDocumentPreviewTo('top'))
  })

  createEffect(() => {
    if (!promptPreview()) setDocumentPreviewOpen(false)
  })

  createEffect(() => {
    const hasDialog = Boolean(localDialog() || dialogPrompt())
    const hasFooterPrompt = Boolean(footerPrompt())
    const currentFocus = shellFocus()
    showLogs()
    if (hasDialog) {
      if (currentFocus !== 'dialog') setShellFocus('dialog')
      return
    }
    if (hasFooterPrompt) {
      if (currentFocus === 'content' || currentFocus === 'prompt') setShellFocus('prompt')
      return
    }
    if (currentFocus !== 'content') setShellFocus('content')
  })

  const sendPromptValue = async (value: unknown) => {
    const current = prompt()
    if (!current) return
    await client.submitPrompt(current.id, value)
    const transition = resolvePromptResponseTransition(current.id, prompt())
    if (transition.clearPrompt) setPrompt(null)
    setStatus(transition.nextStatus)
    setShellFocus(transition.nextShellFocus)
  }

  const openResumeDialog = async () => {
    const result = await requestAction('run.list', {}, true)
    const runs = Array.isArray(result.runs) ? result.runs as Record<string, unknown>[] : []
    if (runs.length === 0) {
      appendLog('当前没有可恢复的 run。')
      return
    }
    setLocalDialog({
      kind: 'select',
      title: '选择要恢复的 run',
      defaultValue: String(runs[0]?.run_id ?? ''),
      options: runs.map((item) => ({
        value: String(item.run_id ?? ''),
        label: `${String(item.run_id ?? '')} | ${String(item.status ?? '')} | ${String(item.project_dir ?? '')}`,
      })),
      onSubmit: (value) => {
        const currentControlId = String(controlSnapshot()?.controlId ?? '')
        void requestAction('run.resume', { control_id: currentControlId, run_id: value })
      },
    })
    setShellFocus('dialog')
  }

  const performControlAction = async (action: 'attach' | 'detach' | 'restart' | 'retry' | 'kill') => {
    const snapshot = controlSnapshot()
    const worker = snapshot?.workers[controlSelectedIndex()]
    if (!snapshot?.controlId || !worker) return
    const argument = String(worker.index ?? controlSelectedIndex() + 1)
    if (action === 'attach') {
      const result = await requestAction('worker.attach', { control_id: snapshot.controlId, argument }, true)
      const command = Array.isArray(result.attach_command) ? result.attach_command.map(String) : []
      if (command.length === 0) {
        appendLog('attach 失败: 后端未返回 attach_command')
        return
      }
      appendLog(`attach: ${command.join(' ')}`)
      await runInheritedCommand(renderer, command, String(result.work_dir ?? ''))
      await requestAction('control.b01.open', { control_id: snapshot.controlId }, true)
      return
    }
    const mapping = {
      detach: 'worker.detach',
      restart: 'worker.restart',
      retry: 'worker.retry',
      kill: 'worker.kill',
    } as const
    await requestAction(mapping[action], { control_id: snapshot.controlId, argument })
  }

  useKeyboard(async (event) => {
    if (event.name) reportPresence('keyboard')
    if (activeDocumentPreview()) {
      if (event.name === 'k' && event.ctrl) {
        event.preventDefault()
        setDocumentPreviewOpen(false)
        return
      }
      if (event.name === 'escape') {
        event.preventDefault()
        setDocumentPreviewOpen(false)
        return
      }
      if (event.name === 'up' || event.name === 'k') {
        event.preventDefault()
        scrollDocumentPreviewBy(-1)
        return
      }
      if (event.name === 'down' || event.name === 'j') {
        event.preventDefault()
        scrollDocumentPreviewBy(1)
        return
      }
      if (event.name === 'pageup') {
        event.preventDefault()
        scrollDocumentPreviewBy(-12)
        return
      }
      if (event.name === 'pagedown') {
        event.preventDefault()
        scrollDocumentPreviewBy(12)
        return
      }
      if (event.name === 'home') {
        event.preventDefault()
        scrollDocumentPreviewTo('top')
        return
      }
      if (event.name === 'end') {
        event.preventDefault()
        scrollDocumentPreviewTo('bottom')
        return
      }
      return
    }
    if (event.name === 'k' && event.ctrl) {
      const preview = promptPreview()
      if (preview) {
        event.preventDefault()
        setDocumentPreviewOpen(true)
        queueMicrotask(() => scrollDocumentPreviewTo('top'))
        return
      }
    }
    if (event.name === 'l' && event.ctrl) {
      event.preventDefault()
      toggleLogs()
      return
    }
    if (localDialog() || dialogPrompt()) return
    if (handleLogNavigation(event)) return
    if (footerPrompt() && shellFocus() === 'prompt') return
    if (route() === 'control') {
      const snapshot = controlSnapshot()
      const workerCount = snapshot?.workers.length ?? 0
      if ((event.name === 'up' || event.name === 'k') && workerCount > 0) {
        event.preventDefault()
        setControlSelectedIndex((prev) => (prev <= 0 ? workerCount - 1 : prev - 1))
        return
      }
      if ((event.name === 'down' || event.name === 'j') && workerCount > 0) {
        event.preventDefault()
        setControlSelectedIndex((prev) => (prev >= workerCount - 1 ? 0 : prev + 1))
        return
      }
      if (event.name === 'return') {
        event.preventDefault()
        await performControlAction('attach')
        return
      }
      if (event.name === 'r') {
        event.preventDefault()
        await performControlAction('restart')
        return
      }
      if (event.name === 't') {
        event.preventDefault()
        await performControlAction('retry')
        return
      }
      if (event.name === 'k') {
        event.preventDefault()
        await performControlAction('kill')
        return
      }
      if (event.name === 'd') {
        event.preventDefault()
        await performControlAction('detach')
        return
      }
      if (event.name === 'u') {
        event.preventDefault()
        await openResumeDialog()
        return
      }
      if (event.name === 'g') {
        event.preventDefault()
        setRoute('home')
        return
      }
    }
  })

  return (
    <box
      flexDirection="column"
      width="100%"
      height="100%"
      onMouseUp={(event) => {
        reportPresence('mouse')
        const selectedText = renderer.getSelection?.()?.getSelectedText?.()
        if (!selectedText) return
        event.preventDefault()
        event.stopPropagation()
        void copyCurrentSelection(renderer)
      }}
    >
      <box
        flexDirection="row"
        gap={1}
        paddingLeft={1}
        paddingRight={1}
        height={shellHeights().top}
        minHeight={shellHeights().top}
      >
        <box width="40%" borderStyle="single" flexDirection="column" height="100%" minHeight="100%">
          <scrollbox flexGrow={1} scrollY>
            <box flexDirection="column" paddingLeft={1} paddingRight={1} paddingTop={1} paddingBottom={1}>
              <For each={statusLines()}>
                {(line, index) => (
                  <text
                    fg={
                      index() === 0
                        ? '#ffffff'
                        : line === currentProgress() || line.startsWith('current_milestone:')
                          ? '#00d2ff'
                          : line.startsWith('> ')
                            ? '#7be495'
                            : '#888888'
                    }
                  >{line}</text>
                )}
              </For>
            </box>
          </scrollbox>
        </box>
        <box width="60%" borderStyle="single" flexDirection="column" flexGrow={1} height="100%" minHeight="100%">
          <scrollbox flexGrow={1} scrollY>
            <box flexDirection="column" paddingTop={1} paddingBottom={1}>
              <Switch>
                <Match when={route() === 'home'}>
                  <HomeRoute snapshot={displayAppSnapshot()} hitl={displayHitlSnapshot()} agents={homeAgents()} />
                </Match>
                <Match when={route() === 'routing'}>
                  <RoutingRoute snapshot={routingSnapshot()} />
                </Match>
                <Match when={route() === 'requirements'}>
                  <RequirementsRoute snapshot={requirementsSnapshot()} />
                </Match>
                <Match when={route() === 'review'}>
                  <ReviewRoute snapshot={reviewSnapshot()} />
                </Match>
                <Match when={route() === 'design'}>
                  <DesignRoute snapshot={designSnapshot()} />
                </Match>
                <Match when={route() === 'task-split'}>
                  <TaskSplitRoute snapshot={taskSplitSnapshot()} />
                </Match>
                <Match when={route() === 'development'}>
                  <DevelopmentRoute snapshot={developmentSnapshot()} />
                </Match>
                <Match when={route() === 'overall-review'}>
                  <OverallReviewRoute snapshot={overallReviewSnapshot()} />
                </Match>
                <Match when={route() === 'control'}>
                  <ControlRoute snapshot={controlSnapshot()} selectedWorkerIndex={controlSelectedIndex()} />
                </Match>
              </Switch>
            </box>
          </scrollbox>
        </box>
      </box>
      <Show when={showLogs() ? logOpenGeneration() : null} keyed>
        {() => (
          <box
            flexDirection="column"
            borderStyle="single"
            marginLeft={1}
            marginRight={1}
            marginTop={1}
            height={shellHeights().log}
            minHeight={shellHeights().log}
          >
            <box flexDirection="column" paddingLeft={1} paddingRight={1} paddingTop={1}>
              <text>日志</text>
              <text fg="#888888">日志已展开，可继续输入 | Ctrl+↑/↓/PgUp/PgDn/Home/End 滚动 | Ctrl+L 收起</text>
            </box>
            <scrollbox
              ref={(value: ScrollBoxRenderable) => {
                logScrollbox = value
              }}
              flexGrow={1}
              scrollY
              stickyScroll={logAutoFollow()}
              stickyStart="bottom"
            >
              <box flexDirection="column" paddingLeft={1} paddingRight={1} paddingBottom={1}>
                <For each={logs()}>{(entry) => <LogEntryCard entry={entry} />}</For>
              </box>
            </scrollbox>
          </box>
        )}
      </Show>
      <Show
        when={footerPrompt()}
        keyed
        fallback={
          <FooterStatusHost
            status={status()}
            progressLine={footerProgressLine()}
            pendingHitl={displayAppSnapshot().pendingHitl}
            height={shellHeights().footer}
          />
        }
      >
        {(active: PromptState) => (
          <FooterPromptHost
            active={active}
            focused={shellFocus() === 'prompt'}
            focusToken={footerPromptFocusToken()}
            height={shellHeights().footer}
            onSubmit={(value) => void sendPromptValue(value)}
          />
        )}
      </Show>
      <Show
        when={localDialog()}
        keyed
        fallback={
          <Show when={dialogPrompt()} keyed>
            {(active: PromptState) => (
              <DialogPromptLayer
                active={active}
                dialogActive={!Boolean(activeDocumentPreview())}
                onSubmit={(value) => void sendPromptValue(value)}
              />
            )}
          </Show>
        }
      >
        {(activeDialog: LocalDialogState) => (
          <LocalDialogLayer
            dialog={activeDialog}
            onClose={() => {
              setLocalDialog(null)
              setShellFocus(footerPrompt() ? 'prompt' : 'content')
            }}
          />
        )}
      </Show>
      <Show when={activeDocumentPreview()} keyed>
        {(preview: DocumentPreviewState) => (
          <box
            position="absolute"
            zIndex={2500}
            top={0}
            left={0}
            width={dimensions().width}
            height={dimensions().height}
          >
            <DocumentPreviewLayer
              preview={preview}
              onScrollboxReady={(value) => {
                documentPreviewScrollbox = value
              }}
            />
          </box>
        )}
      </Show>
    </box>
  )
}
