import { stageRouteForAction } from './stageRegistry'
import type { HomeAgentItem, WorkerSnapshot } from './types'

const LIVE_WORKER_HEALTH_STATUSES = new Set(['alive', 'observe_error', 'provider_auth_error'])
const SOURCE_RANK: Record<HomeAgentItem['source'], number> = {
  control: 7,
  routing: 6,
  requirements: 5,
  review: 4,
  design: 3,
  'task-split': 2,
  development: 1,
  'overall-review': 1,
}
const DISPLAY_SOURCE_ORDER: HomeAgentItem['source'][] = [
  'routing',
  'requirements',
  'review',
  'design',
  'task-split',
  'development',
  'overall-review',
  'control',
]
const DESIGN_REVIEW_ROLE_ORDER = ['开发工程师', '测试工程师', '架构师', '审核员']
const DEVELOPMENT_REVIEW_ROLE_ORDER = ['需求分析师', '测试工程师', '审核员', '架构师']

type HomeAgentSortEntry = {
  item: HomeAgentItem
  workerId: string
}

export function isRunningWorker(worker: WorkerSnapshot): boolean {
  if (!worker.sessionName.trim()) return false
  const agentState = String(worker.agentState || '').trim().toUpperCase()
  if (agentState === 'DEAD' || agentState === 'STARTING') return true
  if (worker.sessionExists !== undefined) return worker.sessionExists
  return LIVE_WORKER_HEALTH_STATUSES.has(worker.healthStatus)
}

export function resolveHomeAgentState(worker: WorkerSnapshot): string {
  const agentState = String(worker.agentState || '').trim().toUpperCase()
  if (agentState === 'DEAD') return 'DEAD'
  if (agentState === 'STARTING') return 'STARTING'
  if (agentState === 'BUSY') return 'BUSY'
  if (agentState === 'READY') return 'READY'
  return 'UNKNOWN'
}

function workerFreshnessTs(worker: WorkerSnapshot): number {
  const updatedAtTs = Date.parse(String(worker.updatedAt || '').trim())
  const heartbeatTs = Date.parse(String(worker.lastHeartbeatAt || '').trim())
  const updatedAt = Number.isFinite(updatedAtTs) ? updatedAtTs : 0
  const heartbeat = Number.isFinite(heartbeatTs) ? heartbeatTs : 0
  return Math.max(updatedAt, heartbeat)
}

function allowedHomeSources(activeStage: string): ReadonlySet<HomeAgentItem['source']> | null {
  const stageRoute = stageRouteForAction(activeStage)
  if (!stageRoute) return null
  return new Set<HomeAgentItem['source']>(['control', stageRoute as HomeAgentItem['source']])
}

function compareText(left: string, right: string): number {
  if (left === right) return 0
  return left < right ? -1 : 1
}

function workerRoleFromSessionName(sessionName: string): string {
  const normalized = String(sessionName || '').trim()
  const separatorIndex = normalized.indexOf('-')
  return (separatorIndex >= 0 ? normalized.slice(0, separatorIndex) : normalized).trim()
}

function roleOrderRank(roleName: string, roleOrder: string[]): number {
  const index = roleOrder.indexOf(roleName)
  return index >= 0 ? index : roleOrder.length
}

function numberedReviewerRank(workerId: string): number | null {
  const match = String(workerId || '').trim().toLowerCase().match(/^requirements-review-r(\d+)$/)
  if (!match) return null
  return Number.parseInt(match[1] || '0', 10) || 0
}

function workerRoleRank(source: HomeAgentItem['source'], workerId: string, sessionName: string): number {
  const normalizedWorkerId = String(workerId || '').trim().toLowerCase()
  const sessionRole = workerRoleFromSessionName(sessionName)
  if (source === 'routing') {
    return normalizedWorkerId === 'routing-initializer' || sessionRole === '路由器' ? 0 : 100
  }
  if (source === 'requirements') {
    if (normalizedWorkerId === 'requirements-notion-reader' || normalizedWorkerId === 'requirements-analyst') return 0
    if (sessionRole === '需求录入员' || sessionRole === '需求分析师' || sessionRole === '分析师') return 0
    return 100
  }
  if (source === 'review') {
    if (normalizedWorkerId === 'requirements-review-analyst') return 0
    if (sessionRole === '评审分析师' || sessionRole === '需求分析师') return 0
    const numberedRank = numberedReviewerRank(normalizedWorkerId)
    if (numberedRank !== null) return 10 + numberedRank
    if (sessionRole === '审核器' || sessionRole === '审核员') return 50
    return 100
  }
  if (source === 'design' || source === 'task-split') {
    if (normalizedWorkerId === 'detailed-design-analyst' || normalizedWorkerId === 'task-split-analyst') return 0
    if (sessionRole === '需求分析师' || sessionRole === '分析师') return 0
    const prefix = source === 'design' ? 'detailed-design-review-' : 'task-split-review-'
    const roleName = normalizedWorkerId.startsWith(prefix) ? workerId.slice(prefix.length).trim() : sessionRole
    return 10 + roleOrderRank(roleName, DESIGN_REVIEW_ROLE_ORDER)
  }
  if (source === 'development' || source === 'overall-review') {
    if (normalizedWorkerId === 'development-developer') return 0
    if (sessionRole === '开发工程师') return 0
    const roleName = normalizedWorkerId.startsWith('development-review-')
      ? workerId.slice('development-review-'.length).trim()
      : sessionRole
    return 10 + roleOrderRank(roleName, DEVELOPMENT_REVIEW_ROLE_ORDER)
  }
  return 100
}

function sourceDisplayRank(source: HomeAgentItem['source'], activeStage: string): number {
  const activeSource = stageRouteForAction(activeStage) as HomeAgentItem['source'] | ''
  if (activeSource) {
    if (source === activeSource) return 0
    if (source === 'control') return 1
    return 2 + DISPLAY_SOURCE_ORDER.indexOf(source)
  }
  const index = DISPLAY_SOURCE_ORDER.indexOf(source)
  return index >= 0 ? index : DISPLAY_SOURCE_ORDER.length
}

function compareHomeAgentEntries(left: HomeAgentSortEntry, right: HomeAgentSortEntry, activeStage: string): number {
  const leftSourceRank = sourceDisplayRank(left.item.source, activeStage)
  const rightSourceRank = sourceDisplayRank(right.item.source, activeStage)
  if (leftSourceRank !== rightSourceRank) return leftSourceRank - rightSourceRank

  const leftRoleRank = workerRoleRank(left.item.source, left.workerId, left.item.sessionName)
  const rightRoleRank = workerRoleRank(right.item.source, right.workerId, right.item.sessionName)
  if (leftRoleRank !== rightRoleRank) return leftRoleRank - rightRoleRank

  return compareText(left.item.sessionName, right.item.sessionName) || compareText(left.item.source, right.item.source)
}

export function buildHomeAgents(
  sources: Array<{ source: HomeAgentItem['source']; workers: WorkerSnapshot[] }>,
  activeStage = '',
): HomeAgentItem[] {
  const scopedSources = allowedHomeSources(activeStage)
  const deduped = new Map<string, HomeAgentSortEntry>()
  const freshnessBySession = new Map<string, number>()
  const sourceRankBySession = new Map<string, number>()
  for (const source of sources) {
    if (scopedSources && !scopedSources.has(source.source)) continue
    for (const worker of source.workers) {
      if (!isRunningWorker(worker)) continue
      const sessionName = worker.sessionName.trim()
      if (!sessionName) continue
      const nextAgentState = resolveHomeAgentState(worker)
      const freshness = workerFreshnessTs(worker)
      const previousFreshness = freshnessBySession.get(sessionName) ?? 0
      const sourceRank = SOURCE_RANK[source.source] || 0
      const previousSourceRank = sourceRankBySession.get(sessionName) || 0
      const previousAgentState = String(deduped.get(sessionName)?.item.agentState || '').trim().toUpperCase()
      if (deduped.has(sessionName)) {
        if (previousFreshness > freshness) continue
        if (previousFreshness === freshness) {
          if (previousAgentState === 'DEAD' && nextAgentState !== 'DEAD') {
            // Prefer a live backend state over a same-timestamp stale DEAD projection.
          } else if (previousAgentState !== 'DEAD' && nextAgentState === 'DEAD') {
            continue
          } else if (previousSourceRank >= sourceRank) {
            continue
          }
        }
      }
      deduped.set(sessionName, {
        item: {
          source: source.source,
          sessionName,
          healthStatus: worker.healthStatus || 'unknown',
          agentState: nextAgentState,
          attachCommand: `tmux attach -t ${sessionName}`,
          workDir: worker.workDir,
        },
        workerId: String(worker.workerId || '').trim(),
      })
      freshnessBySession.set(sessionName, freshness)
      sourceRankBySession.set(sessionName, sourceRank)
    }
  }
  return [...deduped.values()]
    .sort((left, right) => compareHomeAgentEntries(left, right, activeStage))
    .map((entry) => entry.item)
}
