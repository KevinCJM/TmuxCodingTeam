import { isBusyWorker } from './homeAgents'
import type { WorkerSnapshot } from './types'
import { stageBusyLabel as resolveStageBusyLabel, stageProgressKey } from './stageRegistry'

export const FOOTER_SPINNER_FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'] as const

type FooterProgressContext = {
  status: string
  route: string
  activeStage: string
  activeStageLabel: string
  routingWorkers: WorkerSnapshot[]
  requirementsWorkers: WorkerSnapshot[]
  reviewWorkers: WorkerSnapshot[]
  designWorkers: WorkerSnapshot[]
  taskSplitWorkers: WorkerSnapshot[]
  developmentWorkers: WorkerSnapshot[]
  overallReviewWorkers: WorkerSnapshot[]
}

const STARTUP_PROGRESS_PATTERNS = [
  '智能体启动中',
  '启动审核器',
  '初始化审核器',
  '配置审核器',
  '配置审核器模型',
] as const

function stageWorkers(context: FooterProgressContext): WorkerSnapshot[] {
  switch (stageProgressKey(context.route, context.activeStage)) {
    case 'routing':
      return context.routingWorkers
    case 'requirements-intake':
    case 'requirements-clarification':
    case 'requirements':
      return context.requirementsWorkers
    case 'review':
      return context.reviewWorkers
    case 'design':
      return context.designWorkers
    case 'task-split':
      return context.taskSplitWorkers
    case 'development':
      return context.developmentWorkers
    case 'overall-review':
      return context.overallReviewWorkers
    default:
      return []
  }
}

function stageBusyLabel(context: FooterProgressContext): string {
  return resolveStageBusyLabel(context.route, context.activeStage, context.activeStageLabel)
}

export function resolveFooterProgressLine(
  context: FooterProgressContext,
  explicitProgressLine: string,
  frameIndex: number,
): string {
  const explicit = String(explicitProgressLine || '').trim()
  const normalizedStatus = String(context.status || '').trim().toLowerCase()
  if (normalizedStatus !== 'running') return explicit
  const busyWorkers = stageWorkers(context).filter(isBusyWorker)
  const busyCount = busyWorkers.length
  if (explicit) {
    const looksLikeStartup = STARTUP_PROGRESS_PATTERNS.some((pattern) => explicit.includes(pattern))
    if (!looksLikeStartup || busyCount === 0) return explicit
  }
  if (busyCount === 0) return explicit
  const spinner = FOOTER_SPINNER_FRAMES[Math.abs(frameIndex) % FOOTER_SPINNER_FRAMES.length] || FOOTER_SPINNER_FRAMES[0]
  const suffix = busyCount > 1 ? ` | ${busyCount} 个智能体执行中` : ''
  return `${spinner} ${stageBusyLabel(context)}${suffix}`
}
