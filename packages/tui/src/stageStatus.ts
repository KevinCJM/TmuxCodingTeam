import { stageRouteForAction } from './stageRegistry'

export type StageCursor = {
  activeAction: string
  activeStageSeq: number
  terminalAction: string
  terminalStageSeq: number
}

export const EMPTY_STAGE_CURSOR: StageCursor = {
  activeAction: '',
  activeStageSeq: 0,
  terminalAction: '',
  terminalStageSeq: 0,
}

export function normalizeStageSeq(value: unknown): number {
  const candidate = Number(value)
  if (!Number.isFinite(candidate) || candidate <= 0) return 0
  return Math.floor(candidate)
}

export function isTerminalStageStatus(status: string): boolean {
  return ['failed', 'error', 'completed'].includes(String(status ?? '').trim())
}

type StageChangedPayload = {
  action?: unknown
  status?: unknown
  stage_seq?: unknown
  stageSeq?: unknown
}

const ACTIVE_RUNTIME_STATUSES = new Set(['running', 'pending'])
const ACTIVE_WORKER_STATUSES = new Set(['running', 'pending', 'submitted', 'submitting'])
const TERMINAL_RUNTIME_STATUSES = new Set(['done', 'succeeded', 'completed'])
const COMPLETED_WORKER_STATUSES = new Set(['done', 'succeeded', 'completed', 'ready', 'idle'])
const FAILED_WORKER_STATUSES = new Set(['failed', 'stale_failed', 'error'])
const LIVE_WORKER_HEALTH_STATUSES = new Set(['alive', 'observe_error', 'provider_auth_error'])
const STALE_MISSING_SESSION_LIVE_EVIDENCE_MS = 300_000

export function applyStageChanged(cursor: StageCursor, payload: StageChangedPayload): {
  cursor: StageCursor
  accepted: boolean
  status: string
} {
  const action = String(payload.action ?? '').trim()
  const status = String(payload.status ?? 'running').trim() || 'running'
  const stageSeq = normalizeStageSeq(payload.stage_seq ?? payload.stageSeq)
  const isTerminal = isTerminalStageStatus(status)
  if (
    !isTerminal &&
    stageSeq > 0 &&
    cursor.terminalStageSeq > 0 &&
    action === cursor.terminalAction &&
    stageSeq <= cursor.terminalStageSeq
  ) {
    return { cursor, accepted: false, status }
  }
  const next: StageCursor = {
    activeAction: action || cursor.activeAction,
    activeStageSeq: stageSeq || cursor.activeStageSeq,
    terminalAction: cursor.terminalAction,
    terminalStageSeq: cursor.terminalStageSeq,
  }
  if (isTerminal) {
    next.terminalAction = action || next.activeAction
    next.terminalStageSeq = stageSeq || next.activeStageSeq
    return { cursor: next, accepted: true, status }
  }
  if (
    stageSeq > 0 &&
    (
      next.terminalStageSeq === 0
      || stageSeq > next.terminalStageSeq
      || action !== next.terminalAction
    )
  ) {
    next.terminalAction = ''
    next.terminalStageSeq = 0
  }
  return { cursor: next, accepted: true, status }
}

type ProgressPayload = {
  action?: unknown
  stage_seq?: unknown
  stageSeq?: unknown
}

function getObject(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' ? value as Record<string, unknown> : {}
}

function getWorkers(value: unknown): Array<Record<string, unknown>> {
  const snapshot = getObject(value)
  const workers = snapshot.workers
  return Array.isArray(workers) ? workers.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === 'object') : []
}

function workerHasLiveWork(worker: Record<string, unknown>): boolean {
  const sessionName = String(worker.session_name ?? worker.sessionName ?? '').trim()
  if (!sessionName) return false
  if (workerIsStaleMissingSessionLiveNoise(worker)) return false
  const agentState = String(worker.agent_state ?? worker.agentState ?? '').trim().toUpperCase()
  const status = String(worker.status ?? '').trim().toLowerCase()
  const resultStatus = String(worker.result_status ?? worker.resultStatus ?? '').trim().toLowerCase()
  const runtimeStatus = String(worker.current_task_runtime_status ?? worker.currentTaskRuntimeStatus ?? '').trim().toLowerCase()
  const healthStatus = String(worker.health_status ?? worker.healthStatus ?? '').trim().toLowerCase()
  if (agentState === 'DEAD' || healthStatus === 'dead') return false
  if (
    TERMINAL_RUNTIME_STATUSES.has(runtimeStatus) ||
    COMPLETED_WORKER_STATUSES.has(resultStatus) ||
    COMPLETED_WORKER_STATUSES.has(status)
  ) return false
  if (ACTIVE_RUNTIME_STATUSES.has(runtimeStatus) && agentState !== 'READY') return true
  if (FAILED_WORKER_STATUSES.has(resultStatus) || FAILED_WORKER_STATUSES.has(status)) return false
  if (agentState === 'BUSY' || agentState === 'STARTING') return true
  if (agentState === 'READY') return false
  if (ACTIVE_WORKER_STATUSES.has(resultStatus) || ACTIVE_WORKER_STATUSES.has(status)) return true
  return false
}

function workerFreshnessTs(worker: Record<string, unknown>): number {
  const updatedAtTs = Date.parse(String(worker.updated_at ?? worker.updatedAt ?? '').trim())
  const heartbeatTs = Date.parse(String(worker.last_heartbeat_at ?? worker.lastHeartbeatAt ?? '').trim())
  const updatedAt = Number.isFinite(updatedAtTs) ? updatedAtTs : 0
  const heartbeat = Number.isFinite(heartbeatTs) ? heartbeatTs : 0
  return Math.max(updatedAt, heartbeat)
}

function workerHasActiveTurnEvidence(worker: Record<string, unknown>): boolean {
  const status = String(worker.status ?? '').trim().toLowerCase()
  const resultStatus = String(worker.result_status ?? worker.resultStatus ?? '').trim().toLowerCase()
  const runtimeStatus = String(worker.current_task_runtime_status ?? worker.currentTaskRuntimeStatus ?? '').trim().toLowerCase()
  const dispatchState = String(worker.dispatch_state ?? worker.dispatchState ?? '').trim().toLowerCase()
  if (ACTIVE_RUNTIME_STATUSES.has(runtimeStatus)) return true
  if (ACTIVE_WORKER_STATUSES.has(resultStatus) || ACTIVE_WORKER_STATUSES.has(status)) return true
  if (dispatchState === 'submitting' || dispatchState === 'submitted') return true
  return Boolean(String(worker.turn_status_path ?? worker.turnStatusPath ?? '').trim())
}

function workerIsStaleMissingSessionLiveNoise(worker: Record<string, unknown>): boolean {
  if (Boolean(worker.session_exists ?? worker.sessionExists)) return false
  const hasExplicitSessionExists = 'session_exists' in worker || 'sessionExists' in worker
  if (!hasExplicitSessionExists) return false
  const healthStatus = String(worker.health_status ?? worker.healthStatus ?? '').trim().toLowerCase()
  if (!LIVE_WORKER_HEALTH_STATUSES.has(healthStatus)) return false
  const status = String(worker.status ?? '').trim().toLowerCase()
  const resultStatus = String(worker.result_status ?? worker.resultStatus ?? '').trim().toLowerCase()
  if (FAILED_WORKER_STATUSES.has(status) || FAILED_WORKER_STATUSES.has(resultStatus)) return false
  const agentState = String(worker.agent_state ?? worker.agentState ?? '').trim().toUpperCase()
  if (agentState === 'DEAD' || agentState === 'STARTING') return false
  if (workerHasActiveTurnEvidence(worker)) return false
  const freshness = workerFreshnessTs(worker)
  if (freshness <= 0) return false
  const now = Date.now()
  return freshness <= now && now - freshness > STALE_MISSING_SESSION_LIVE_EVIDENCE_MS
}

function stageSnapshotForAction(snapshots: Record<string, unknown>, activeStage: string): unknown {
  const stages = getObject(snapshots.stages)
  const route = stageRouteForAction(activeStage)
  return route ? stages[route] : undefined
}

export function stageSnapshotHasLiveWork(snapshot: unknown): boolean {
  return getWorkers(snapshot).some(workerHasLiveWork)
}

export function inferBootstrapStatus(payload: Record<string, unknown>): string {
  const snapshots = getObject(payload.snapshots)
  const app = getObject(snapshots.app)
  const activeStageStatus = String(app.active_stage_status ?? app.activeStageStatus ?? '').trim().toLowerCase()
  if (activeStageStatus === 'failed' || activeStageStatus === 'error') return activeStageStatus
  if (Boolean(app.pending_hitl ?? app.pendingHitl)) return 'awaiting-input'
  if (Boolean(app.pending_attention ?? app.pendingAttention)) return 'awaiting-input'
  if (activeStageStatus === 'awaiting-input') return 'awaiting-input'
  if (activeStageStatus === 'running') return 'running'
  const activeStage = String(app.active_stage ?? app.activeStage ?? '').trim()
  const stageSnapshot = stageSnapshotForAction(snapshots, activeStage)
  if (getWorkers(stageSnapshot).some(workerHasLiveWork)) return 'running'
  return 'ready'
}

export function shouldRecoverRunningFromStageSnapshot(
  currentStatus: string,
  activeStage: string,
  route: string,
  snapshot: unknown,
  hasPendingInput = false,
): boolean {
  const normalizedStatus = String(currentStatus ?? '').trim().toLowerCase()
  if (
    normalizedStatus !== 'failed' &&
    normalizedStatus !== 'error' &&
    normalizedStatus !== 'awaiting-input'
  ) return false
  if (normalizedStatus === 'awaiting-input' && hasPendingInput) return false
  const activeRoute = stageRouteForAction(activeStage)
  if (!activeRoute || String(route || '').trim() !== activeRoute) return false
  return stageSnapshotHasLiveWork(snapshot)
}

export function shouldAcceptProgressEvent(cursor: StageCursor, payload: ProgressPayload): boolean {
  const action = String(payload.action ?? '').trim()
  const stageSeq = normalizeStageSeq(payload.stage_seq ?? payload.stageSeq)
  if (!action || stageSeq === 0) return false
  if (
    cursor.terminalStageSeq > 0 &&
    action === cursor.terminalAction &&
    stageSeq <= cursor.terminalStageSeq
  ) {
    return false
  }
  if (!cursor.activeAction || cursor.activeStageSeq === 0) return false
  return action === cursor.activeAction && stageSeq === cursor.activeStageSeq
}

export function markTerminalStage(cursor: StageCursor): StageCursor {
  if (!cursor.activeAction || cursor.activeStageSeq === 0) return cursor
  return {
    ...cursor,
    terminalAction: cursor.activeAction,
    terminalStageSeq: cursor.activeStageSeq,
  }
}
