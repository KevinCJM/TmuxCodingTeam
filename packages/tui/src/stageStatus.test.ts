import { expect, test } from 'bun:test'
import {
  applyStageChanged,
  EMPTY_STAGE_CURSOR,
  inferBootstrapStatus,
  markTerminalStage,
  shouldAcceptProgressEvent,
  shouldRecoverRunningFromStageSnapshot,
} from './stageStatus'

test('failed stage keeps later progress from the same stage sequence from reviving running state', () => {
  const failed = applyStageChanged(EMPTY_STAGE_CURSOR, {
    action: 'stage.a05.start',
    status: 'failed',
    stage_seq: 7,
  })
  expect(failed.accepted).toBe(true)
  expect(failed.status).toBe('failed')
  expect(
    shouldAcceptProgressEvent(failed.cursor, {
      action: 'stage.a05.start',
      stage_seq: 7,
    }),
  ).toBe(false)
})

test('newer stage sequence clears terminal lock and accepts fresh progress', () => {
  const failed = applyStageChanged(EMPTY_STAGE_CURSOR, {
    action: 'stage.a05.start',
    status: 'failed',
    stage_seq: 7,
  })
  const restarted = applyStageChanged(failed.cursor, {
    action: 'stage.a05.start',
    status: 'running',
    stage_seq: 8,
  })
  expect(restarted.accepted).toBe(true)
  expect(restarted.status).toBe('running')
  expect(
    shouldAcceptProgressEvent(restarted.cursor, {
      action: 'stage.a05.start',
      stage_seq: 7,
    }),
  ).toBe(false)
  expect(
    shouldAcceptProgressEvent(restarted.cursor, {
      action: 'stage.a05.start',
      stage_seq: 8,
    }),
  ).toBe(true)
})

test('newer stage sequence also revives awaiting-input after failure', () => {
  const failed = applyStageChanged(EMPTY_STAGE_CURSOR, {
    action: 'stage.a07.start',
    status: 'failed',
    stage_seq: 11,
  })
  const awaiting = applyStageChanged(failed.cursor, {
    action: 'stage.a07.start',
    status: 'awaiting-input',
    stage_seq: 12,
  })
  expect(awaiting.accepted).toBe(true)
  expect(awaiting.status).toBe('awaiting-input')
  expect(
    shouldAcceptProgressEvent(awaiting.cursor, {
      action: 'stage.a07.start',
      stage_seq: 12,
    }),
  ).toBe(true)
})

test('error path can mark the current stage as terminal before failed stage.changed arrives', () => {
  const running = applyStageChanged(EMPTY_STAGE_CURSOR, {
    action: 'stage.a05.start',
    status: 'running',
    stage_seq: 3,
  })
  const cursor = markTerminalStage(running.cursor)
  expect(
    shouldAcceptProgressEvent(cursor, {
      action: 'stage.a05.start',
      stage_seq: 3,
    }),
  ).toBe(false)
})

test('bootstrap infers running when active stage has a busy worker', () => {
  expect(
    inferBootstrapStatus({
      snapshots: {
        app: {
          active_stage: 'stage.a07.start',
          pending_hitl: false,
        },
        stages: {
          development: {
            workers: [
              {
                session_name: '开发工程师-天猛星',
                status: 'running',
                agent_state: 'BUSY',
                health_status: 'alive',
                current_task_runtime_status: 'running',
                session_exists: true,
              },
            ],
          },
        },
      },
    }),
  ).toBe('running')
})

test('bootstrap keeps runner failure status ahead of live worker inference', () => {
  expect(
    inferBootstrapStatus({
      snapshots: {
        app: {
          active_stage: 'stage.a07.start',
          active_stage_status: 'failed',
          pending_hitl: false,
        },
        stages: {
          development: {
            workers: [
              {
                session_name: '开发工程师-昴日鸡',
                status: 'running',
                agent_state: 'BUSY',
                health_status: 'alive',
                current_task_runtime_status: 'running',
                session_exists: true,
              },
            ],
          },
        },
      },
    }),
  ).toBe('failed')
})

test('bootstrap maps control session action to routing stage workers', () => {
  expect(
    inferBootstrapStatus({
      snapshots: {
        app: {
          active_stage: 'control.b01.open',
          pending_hitl: false,
        },
        stages: {
          routing: {
            workers: [
              {
                session_name: '路由初始化-角木蛟',
                status: 'running',
                agent_state: 'BUSY',
                health_status: 'alive',
                current_task_runtime_status: 'running',
                session_exists: true,
              },
            ],
          },
        },
      },
    }),
  ).toBe('running')
})

test('bootstrap keeps awaiting-input ahead of worker activity', () => {
  expect(
    inferBootstrapStatus({
      snapshots: {
        app: {
          active_stage: 'stage.a07.start',
          pending_hitl: true,
        },
        stages: {
          development: {
            workers: [
              {
                session_name: '开发工程师-天猛星',
                status: 'running',
                agent_state: 'BUSY',
                health_status: 'alive',
                current_task_runtime_status: 'running',
                session_exists: true,
              },
            ],
          },
        },
      },
    }),
  ).toBe('awaiting-input')
})

test('bootstrap treats pending attention as awaiting-input without HITL', () => {
  expect(
    inferBootstrapStatus({
      snapshots: {
        app: {
          active_stage: 'stage.a08.start',
          pending_hitl: false,
          pending_attention: true,
        },
        stages: {
          'overall-review': {
            workers: [],
          },
        },
      },
    }),
  ).toBe('awaiting-input')
})

test('failed status recovers when current routing snapshot still has live workers', () => {
  expect(
    shouldRecoverRunningFromStageSnapshot(
      'failed',
      'stage.a01.start',
      'routing',
      {
        workers: [
          {
            session_name: '路由器-地微星',
            status: 'running',
            agent_state: 'BUSY',
            health_status: 'alive',
            current_task_runtime_status: 'running',
            session_exists: true,
          },
        ],
      },
    ),
  ).toBe(true)
})

test('failed status does not recover from idle READY sessions after a stage runner failed', () => {
  expect(
    shouldRecoverRunningFromStageSnapshot(
      'failed',
      'stage.a05.start',
      'design',
      {
        workers: [
          {
            session_name: '架构师-奎木狼',
            status: 'failed',
            agent_state: 'READY',
            health_status: 'alive',
            current_task_runtime_status: '',
            session_exists: true,
          },
          {
            session_name: '开发工程师-地魁星',
            status: 'succeeded',
            agent_state: 'READY',
            health_status: 'alive',
            current_task_runtime_status: 'done',
            session_exists: true,
          },
        ],
      },
    ),
  ).toBe(false)
})

test('failed status recovers from live BUSY worker even when stale failed status remains', () => {
  expect(
    shouldRecoverRunningFromStageSnapshot(
      'failed',
      'stage.a07.start',
      'development',
      {
        workers: [
          {
            session_name: '开发工程师-地威星',
            status: 'failed',
            agent_state: 'BUSY',
            health_status: 'alive',
            current_task_runtime_status: 'running',
            session_exists: true,
          },
        ],
      },
    ),
  ).toBe(true)
})

test('bootstrap does not infer running from completed READY sessions', () => {
  expect(
    inferBootstrapStatus({
      snapshots: {
        app: {
          active_stage: 'stage.a05.start',
          pending_hitl: false,
        },
        stages: {
          design: {
            workers: [
              {
                session_name: '开发工程师-地魁星',
                status: 'succeeded',
                agent_state: 'READY',
                health_status: 'alive',
                current_task_runtime_status: 'done',
                session_exists: true,
              },
            ],
          },
        },
      },
    }),
  ).toBe('ready')
})

test('bootstrap ignores stale running runtime status on READY workers', () => {
  expect(
    inferBootstrapStatus({
      snapshots: {
        app: {
          active_stage: 'stage.a07.start',
          pending_hitl: false,
        },
        stages: {
          development: {
            workers: [
              {
                session_name: '测试工程师-天寿星',
                status: 'ready',
                agent_state: 'READY',
                health_status: 'alive',
                current_task_runtime_status: 'running',
                session_exists: true,
              },
            ],
          },
        },
      },
    }),
  ).toBe('ready')
})

test('bootstrap ignores terminal stale BUSY workers in overall review', () => {
  const staleTerminalWorker = {
    session_name: '需求分析师-地英星',
    status: 'succeeded',
    result_status: 'succeeded',
    agent_state: 'BUSY',
    health_status: 'alive',
    current_task_runtime_status: 'done',
    session_exists: true,
  }
  expect(
    inferBootstrapStatus({
      snapshots: {
        app: {
          active_stage: 'stage.a08.start',
          pending_hitl: false,
        },
        stages: {
          'overall-review': {
            workers: [staleTerminalWorker],
          },
        },
      },
    }),
  ).toBe('ready')
  expect(
    shouldRecoverRunningFromStageSnapshot(
      'awaiting-input',
      'stage.a08.start',
      'overall-review',
      { workers: [staleTerminalWorker] },
      false,
    ),
  ).toBe(false)
})

test('bootstrap ignores old missing-session alive workers in overall review', () => {
  const staleMissingSessionWorker = {
    session_name: '需求分析师-地英星',
    status: '',
    agent_state: 'BUSY',
    health_status: 'alive',
    current_task_runtime_status: '',
    session_exists: false,
    updated_at: '2000-01-01T00:00:00Z',
    last_heartbeat_at: '2000-01-01T00:00:00Z',
  }
  expect(
    inferBootstrapStatus({
      snapshots: {
        app: {
          active_stage: 'stage.a08.start',
          pending_hitl: false,
        },
        stages: {
          'overall-review': {
            workers: [staleMissingSessionWorker],
          },
        },
      },
    }),
  ).toBe('ready')
  expect(
    shouldRecoverRunningFromStageSnapshot(
      'awaiting-input',
      'stage.a08.start',
      'overall-review',
      { workers: [staleMissingSessionWorker] },
      false,
    ),
  ).toBe(false)
})

test('bootstrap ignores terminal stale BUSY workers in requirements review', () => {
  const staleTerminalWorker = {
    session_name: '审核器-地进星',
    status: 'succeeded',
    result_status: 'succeeded',
    agent_state: 'BUSY',
    health_status: 'alive',
    current_task_runtime_status: 'done',
    session_exists: true,
  }
  expect(
    inferBootstrapStatus({
      snapshots: {
        app: {
          active_stage: 'stage.a04.start',
          pending_hitl: false,
        },
        stages: {
          review: {
            workers: [staleTerminalWorker],
          },
        },
      },
    }),
  ).toBe('ready')
  expect(
    shouldRecoverRunningFromStageSnapshot(
      'awaiting-input',
      'stage.a04.start',
      'review',
      { workers: [staleTerminalWorker] },
      false,
    ),
  ).toBe(false)
})

test('bootstrap ignores terminal stale BUSY workers in detailed design', () => {
  const staleTerminalWorker = {
    session_name: '分析师-心月狐',
    status: 'succeeded',
    result_status: 'succeeded',
    agent_state: 'BUSY',
    health_status: 'alive',
    current_task_runtime_status: 'done',
    session_exists: true,
  }
  expect(
    inferBootstrapStatus({
      snapshots: {
        app: {
          active_stage: 'stage.a05.start',
          pending_hitl: false,
        },
        stages: {
          design: {
            workers: [staleTerminalWorker],
          },
        },
      },
    }),
  ).toBe('ready')
  expect(
    shouldRecoverRunningFromStageSnapshot(
      'awaiting-input',
      'stage.a05.start',
      'design',
      { workers: [staleTerminalWorker] },
      false,
    ),
  ).toBe(false)
})

test('awaiting-input recovers when no prompt is pending and current snapshot has live workers', () => {
  expect(
    shouldRecoverRunningFromStageSnapshot(
      'awaiting-input',
      'stage.a04.start',
      'review',
      {
        workers: [
          {
            session_name: '审核器-地奇星',
            status: 'running',
            agent_state: 'BUSY',
            health_status: 'alive',
            current_task_runtime_status: 'running',
            session_exists: true,
          },
        ],
      },
      false,
    ),
  ).toBe(true)
})

test('awaiting-input stays sticky while a real prompt is pending', () => {
  expect(
    shouldRecoverRunningFromStageSnapshot(
      'awaiting-input',
      'stage.a04.start',
      'review',
      {
        workers: [
          {
            session_name: '审核器-地奇星',
            status: 'running',
            agent_state: 'BUSY',
            health_status: 'alive',
            current_task_runtime_status: 'running',
            session_exists: true,
          },
        ],
      },
      true,
    ),
  ).toBe(false)
})

test('failed status is not recovered from unrelated stage snapshots', () => {
  expect(
    shouldRecoverRunningFromStageSnapshot(
      'failed',
      'stage.a01.start',
      'development',
      {
        workers: [
          {
            session_name: '开发工程师-天速星',
            status: 'running',
            agent_state: 'BUSY',
            health_status: 'alive',
            session_exists: true,
          },
        ],
      },
    ),
  ).toBe(false)
})
