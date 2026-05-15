import { expect, test } from 'bun:test'
import { buildAgentConfigLabel, buildHomeAgents, isBusyWorker, isRunningWorker, resolveHomeAgentState } from './homeAgents'
import type { WorkerSnapshot } from './types'

function worker(overrides: Partial<WorkerSnapshot> = {}): WorkerSnapshot {
  return {
    workDir: '/tmp/project',
    sessionName: 'sess-1',
    status: 'running',
    workflowStage: 'create_running',
    agentState: 'READY',
    healthStatus: 'alive',
    currentTaskRuntimeStatus: '',
    retryCount: 0,
    note: '',
    transcriptPath: '',
    turnStatusPath: '',
    questionPath: '',
    answerPath: '',
    artifactPaths: [],
    ...overrides,
  }
}

test('isRunningWorker keeps live health evidence when sessionExists is false', () => {
  expect(isRunningWorker(worker({ sessionExists: true, healthStatus: 'unknown' }))).toBe(true)
  expect(isRunningWorker(worker({ sessionExists: false, healthStatus: 'alive' }))).toBe(true)
  expect(isRunningWorker(worker({ sessionExists: false, healthStatus: 'observe_error' }))).toBe(true)
  expect(isRunningWorker(worker({ sessionExists: false, healthStatus: 'provider_auth_error' }))).toBe(true)
})

test('isRunningWorker hides stale missing-session alive snapshots without active turn evidence', () => {
  expect(
    isRunningWorker(
      worker({
        sessionExists: false,
        healthStatus: 'alive',
        agentState: 'READY',
        status: 'succeeded',
        resultStatus: 'succeeded',
        currentTaskRuntimeStatus: 'done',
        updatedAt: '2000-01-01T00:00:00Z',
        lastHeartbeatAt: '2000-01-01T00:00:00Z',
      }),
    ),
  ).toBe(false)
  expect(
    isRunningWorker(
      worker({
        sessionExists: false,
        healthStatus: 'alive',
        agentState: 'READY',
        status: 'succeeded',
        resultStatus: 'succeeded',
        currentTaskRuntimeStatus: 'done',
        updatedAt: new Date().toISOString(),
      }),
    ),
  ).toBe(true)
})

test('isRunningWorker hides stale non-live snapshots without visible state evidence', () => {
  expect(
    isRunningWorker(
      worker({
        sessionExists: false,
        healthStatus: 'unknown',
        agentState: 'READY',
        status: 'ready',
        resultStatus: '',
        currentTaskRuntimeStatus: '',
      }),
    ),
  ).toBe(false)
  expect(isRunningWorker(worker({ sessionExists: false, healthStatus: 'unknown', agentState: 'BUSY' }))).toBe(true)
})

test('isRunningWorker keeps workers already marked DEAD so home can show failed agents', () => {
  expect(isRunningWorker(worker({ sessionExists: true, agentState: 'DEAD' }))).toBe(true)
})

test('isRunningWorker keeps prelaunch STARTING workers before tmux session exists', () => {
  expect(isRunningWorker(worker({ sessionExists: false, agentState: 'STARTING', healthStatus: 'unknown' }))).toBe(true)
})

test('resolveHomeAgentState prioritizes live BUSY over stale failed status', () => {
  expect(
    resolveHomeAgentState(
      worker({
        agentState: 'BUSY',
        status: 'failed',
        resultStatus: 'failed',
        currentTaskRuntimeStatus: 'running',
      }),
    ),
  ).toBe('BUSY')
  expect(
    isBusyWorker(
      worker({
        agentState: 'BUSY',
        status: 'failed',
        resultStatus: 'failed',
        currentTaskRuntimeStatus: 'running',
      }),
    ),
  ).toBe(true)
})

test('resolveHomeAgentState lets terminal result override stale BUSY', () => {
  const terminalBusy = worker({
    agentState: 'BUSY',
    status: 'succeeded',
    resultStatus: 'succeeded',
    currentTaskRuntimeStatus: 'done',
  })
  expect(resolveHomeAgentState(terminalBusy)).toBe('READY')
  expect(isBusyWorker(terminalBusy)).toBe(false)
})

test('buildHomeAgents shows completed A04 reviewers as READY and active BA feedback as BUSY', () => {
  const agents = buildHomeAgents(
    [
      {
        source: 'review',
        workers: [
          worker({
            workerId: 'requirements-review-r1',
            sessionName: '审核器-地会星',
            sessionExists: true,
            agentState: 'BUSY',
            status: 'succeeded',
            resultStatus: 'succeeded',
            currentTaskRuntimeStatus: 'done',
          }),
          worker({
            workerId: 'requirements-review-r2',
            sessionName: '审核器-地走星',
            sessionExists: true,
            agentState: 'BUSY',
            status: 'succeeded',
            resultStatus: 'succeeded',
            currentTaskRuntimeStatus: 'done',
          }),
          worker({
            workerId: 'requirements-analyst',
            sessionName: '分析师-心月狐',
            sessionExists: true,
            agentState: 'BUSY',
            status: 'running',
            resultStatus: 'running',
            currentTaskRuntimeStatus: 'running',
            note: 'turn:requirements_review_feedback_round_2_round_1',
          }),
        ],
      },
    ],
    'stage.a04.start',
  )

  expect(agents.map((agent) => [agent.sessionName, agent.agentState])).toEqual([
    ['审核器-地会星', 'READY'],
    ['审核器-地走星', 'READY'],
    ['分析师-心月狐', 'BUSY'],
  ])
  expect(agents.filter((agent) => agent.agentState === 'BUSY')).toHaveLength(1)
})

test('buildHomeAgents shows completed A05 stale busy workers as READY and active reviewer as BUSY', () => {
  const agents = buildHomeAgents(
    [
      {
        source: 'design',
        workers: [
          worker({
            workerId: 'requirements-analyst',
            sessionName: '分析师-心月狐',
            sessionExists: true,
            agentState: 'BUSY',
            status: 'succeeded',
            resultStatus: 'succeeded',
            currentTaskRuntimeStatus: 'done',
          }),
          worker({
            workerId: 'detailed-design-review-开发工程师',
            sessionName: '开发工程师-地遂星',
            sessionExists: true,
            agentState: 'BUSY',
            status: 'succeeded',
            resultStatus: 'succeeded',
            currentTaskRuntimeStatus: 'done',
          }),
          worker({
            workerId: 'detailed-design-review-审核员',
            sessionName: '审核员-天退星',
            sessionExists: true,
            agentState: 'BUSY',
            status: 'running',
            resultStatus: 'running',
            currentTaskRuntimeStatus: 'running',
          }),
        ],
      },
    ],
    'stage.a05.start',
  )

  expect(agents.map((agent) => [agent.sessionName, agent.agentState])).toEqual([
    ['分析师-心月狐', 'READY'],
    ['开发工程师-地遂星', 'READY'],
    ['审核员-天退星', 'BUSY'],
  ])
  expect(agents.filter((agent) => agent.agentState === 'BUSY')).toHaveLength(1)
})

test('buildAgentConfigLabel formats vendor model and effort for home display', () => {
  expect(buildAgentConfigLabel(worker({
    vendor: 'codex',
    model: 'gpt-5.5',
    reasoningEffort: 'high',
  }))).toBe('Codex | GPT-5.5, High')
})

test('buildHomeAgents omits config label when worker config is missing', () => {
  const agents = buildHomeAgents([
    {
      source: 'development',
      workers: [worker({ sessionName: '开发工程师-地雄星', sessionExists: true })],
    },
  ])
  expect(agents[0]?.agentConfigLabel).toBe('')
})

test('buildHomeAgents prefers newer DEAD snapshot over older READY snapshot for the same session', () => {
  const agents = buildHomeAgents([
    {
      source: 'control',
      workers: [worker({ sessionName: 'sess-1', sessionExists: true, agentState: 'READY', updatedAt: '2026-04-22T10:00:00+08:00' })],
    },
    {
      source: 'development',
      workers: [worker({ sessionName: 'sess-1', sessionExists: false, agentState: 'DEAD', updatedAt: '2026-04-22T10:00:01+08:00' })],
    },
  ])
  expect(agents).toHaveLength(1)
  expect(agents[0]).toMatchObject({
    source: 'development',
    sessionName: 'sess-1',
    agentState: 'DEAD',
    agentConfigLabel: '',
  })
})

test('buildHomeAgents prefers live state over DEAD when timestamps collide for the same session', () => {
  const agents = buildHomeAgents([
    {
      source: 'control',
      workers: [worker({ sessionName: 'sess-1', sessionExists: true, agentState: 'DEAD', updatedAt: '2026-04-22T10:00:00+08:00' })],
    },
    {
      source: 'development',
      workers: [worker({ sessionName: 'sess-1', sessionExists: true, agentState: 'READY', updatedAt: '2026-04-22T10:00:00+08:00' })],
    },
  ])
  expect(agents).toHaveLength(1)
  expect(agents[0]?.agentState).toBe('READY')
})

test('buildHomeAgents limits home overview to control and current stage sources when active stage is known', () => {
  const agents = buildHomeAgents(
    [
      {
        source: 'requirements',
        workers: [worker({ sessionName: '分析师-天慧星', sessionExists: false, agentState: 'DEAD' })],
      },
      {
        source: 'development',
        workers: [worker({ sessionName: '开发工程师-天猛星', sessionExists: true, agentState: 'BUSY' })],
      },
      {
        source: 'control',
        workers: [worker({ sessionName: '控制台-当前运行', sessionExists: true, agentState: 'READY' })],
      },
    ],
    'stage.a07.start',
  )
  expect(agents).toHaveLength(2)
  expect(agents.map((agent) => agent.sessionName)).toEqual(['开发工程师-天猛星', '控制台-当前运行'])
})

test('buildHomeAgents keeps live overall review workers when session probe is temporarily false', () => {
  const expectedSessionNames = [
    '需求分析师-地奇星',
    '测试工程师-天慧星',
    '测试工程师-天暴星',
    '审核员-翼火蛇',
    '架构师-亢金龙',
    '需求分析师-地英星',
  ]
  const agents = buildHomeAgents(
    [
      {
        source: 'overall-review',
        workers: [
          worker({
            sessionName: '需求分析师-地奇星',
            sessionExists: false,
            healthStatus: 'alive',
            agentState: 'READY',
            status: 'succeeded',
            resultStatus: 'succeeded',
            currentTaskRuntimeStatus: 'done',
          }),
          worker({
            sessionName: '测试工程师-天慧星',
            sessionExists: false,
            healthStatus: 'alive',
            agentState: 'READY',
            status: 'succeeded',
            resultStatus: 'succeeded',
            currentTaskRuntimeStatus: 'done',
          }),
          worker({
            sessionName: '测试工程师-天暴星',
            sessionExists: false,
            healthStatus: 'alive',
            agentState: 'READY',
            status: 'succeeded',
            resultStatus: 'succeeded',
            currentTaskRuntimeStatus: 'done',
          }),
          worker({
            sessionName: '审核员-翼火蛇',
            sessionExists: false,
            healthStatus: 'alive',
            agentState: 'READY',
            status: 'succeeded',
            resultStatus: 'succeeded',
            currentTaskRuntimeStatus: 'done',
          }),
          worker({
            sessionName: '架构师-亢金龙',
            sessionExists: false,
            healthStatus: 'alive',
            agentState: 'BUSY',
            status: 'running',
            currentTaskRuntimeStatus: 'running',
          }),
          worker({
            sessionName: '需求分析师-地英星',
            sessionExists: false,
            healthStatus: 'alive',
            agentState: 'BUSY',
            status: 'running',
            currentTaskRuntimeStatus: 'running',
          }),
        ],
      },
    ],
    'stage.a08.start',
  )

  expect(agents).toHaveLength(6)
  expect([...agents.map((agent) => agent.sessionName)].sort()).toEqual([...expectedSessionNames].sort())
  expect(agents.filter((agent) => agent.agentState === 'BUSY').map((agent) => agent.sessionName).sort()).toEqual([
    '架构师-亢金龙',
    '需求分析师-地英星',
  ].sort())
})

test('buildHomeAgents keeps current-stage DEAD worker visible for requirements stage', () => {
  const agents = buildHomeAgents(
    [
      {
        source: 'requirements',
        workers: [worker({ sessionName: '分析师-天慧星', sessionExists: false, agentState: 'DEAD' })],
      },
      {
        source: 'development',
        workers: [worker({ sessionName: '开发工程师-天猛星', sessionExists: true, agentState: 'BUSY' })],
      },
    ],
    'stage.a03.start',
  )
  expect(agents).toHaveLength(1)
  expect(agents[0]).toMatchObject({
    source: 'requirements',
    sessionName: '分析师-天慧星',
    agentState: 'DEAD',
  })
})

test('buildHomeAgents keeps cross-stage aggregation when active stage is unknown', () => {
  const agents = buildHomeAgents([
    {
      source: 'requirements',
      workers: [worker({ sessionName: '分析师-天慧星', sessionExists: false, agentState: 'DEAD' })],
    },
    {
      source: 'development',
      workers: [worker({ sessionName: '开发工程师-天猛星', sessionExists: true, agentState: 'BUSY' })],
    },
  ])
  expect(agents).toHaveLength(2)
  expect(agents.map((agent) => agent.sessionName)).toEqual(['分析师-天慧星', '开发工程师-天猛星'])
})

test('buildHomeAgents prefers fresher live heartbeat over stale dead snapshot for the same session', () => {
  const agents = buildHomeAgents([
    {
      source: 'control',
      workers: [worker({
        sessionName: 'sess-1',
        sessionExists: false,
        agentState: 'DEAD',
        updatedAt: '2026-04-22T10:00:01+08:00',
      })],
    },
    {
      source: 'design',
      workers: [worker({
        sessionName: 'sess-1',
        sessionExists: true,
        agentState: 'BUSY',
        updatedAt: '2026-04-22T10:00:00+08:00',
        lastHeartbeatAt: '2026-04-22T10:00:02+08:00',
      })],
    },
  ])
  expect(agents).toHaveLength(1)
  expect(agents[0]).toMatchObject({
    source: 'design',
    sessionName: 'sess-1',
    agentState: 'BUSY',
  })
})

test('buildHomeAgents keeps control workers when duplicate sessions exist across sources', () => {
  const agents = buildHomeAgents([
    { source: 'control', workers: [worker({ sessionName: 'sess-1', sessionExists: true })] },
    { source: 'routing', workers: [worker({ sessionName: 'sess-1', sessionExists: true, workDir: '/tmp/other' })] },
    { source: 'requirements', workers: [worker({ sessionName: 'sess-2', sessionExists: true })] },
  ])
  expect(agents).toHaveLength(2)
  expect(agents.map((agent) => agent.sessionName)).toEqual(['sess-2', 'sess-1'])
  expect(agents.find((agent) => agent.sessionName === 'sess-1')).toMatchObject({
    source: 'control',
    sessionName: 'sess-1',
    attachCommand: 'tmux attach -t sess-1',
  })
  expect(agents.find((agent) => agent.sessionName === 'sess-2')).toMatchObject({
    source: 'requirements',
    sessionName: 'sess-2',
  })
})

test('resolveHomeAgentState preserves READY when backend already reports ready', () => {
  expect(resolveHomeAgentState(worker({ agentState: 'READY', status: 'running' }))).toBe('READY')
  expect(resolveHomeAgentState(worker({ agentState: 'READY', status: 'ready', currentTaskRuntimeStatus: 'running' }))).toBe('READY')
  expect(resolveHomeAgentState(worker({ agentState: 'STARTING', status: 'running' }))).toBe('STARTING')
  expect(resolveHomeAgentState(worker({ agentState: '', status: 'running' }))).toBe('BUSY')
})

test('resolveHomeAgentState promotes running snapshots to BUSY and terminal snapshots to READY', () => {
  expect(resolveHomeAgentState(worker({ agentState: '', resultStatus: 'running' }))).toBe('BUSY')
  expect(resolveHomeAgentState(worker({ agentState: '', currentTaskRuntimeStatus: 'running' }))).toBe('BUSY')
  expect(resolveHomeAgentState(worker({ agentState: '', status: 'ready' }))).toBe('READY')
  expect(resolveHomeAgentState(worker({ agentState: '', resultStatus: 'completed' }))).toBe('READY')
})

test('buildHomeAgents keeps newer READY development state over stale control DEAD', () => {
  const agents = buildHomeAgents(
    [
      {
        source: 'control',
        workers: [
          worker({
            sessionName: '开发工程师-天速星',
            sessionExists: true,
            agentState: 'DEAD',
            healthStatus: 'alive',
            updatedAt: '2026-04-22T10:00:00+08:00',
          }),
        ],
      },
      {
        source: 'development',
        workers: [
          worker({
            sessionName: '开发工程师-天速星',
            sessionExists: true,
            agentState: 'READY',
            healthStatus: 'alive',
            updatedAt: '2026-04-22T10:00:01+08:00',
          }),
        ],
      },
    ],
    'stage.a07.start',
  )
  expect(agents).toHaveLength(1)
  expect(agents[0]).toMatchObject({
    source: 'development',
    sessionName: '开发工程师-天速星',
    agentState: 'READY',
  })
})

test('buildHomeAgents keeps backend READY state for running development worker', () => {
  const agents = buildHomeAgents([
    {
      source: 'development',
      workers: [
        worker({
          sessionName: '开发工程师-亢金龙',
          sessionExists: true,
          agentState: 'READY',
          status: 'running',
          currentTaskRuntimeStatus: 'running',
        }),
      ],
    },
  ])
  expect(agents).toHaveLength(1)
  expect(agents[0]).toMatchObject({
    source: 'development',
    sessionName: '开发工程师-亢金龙',
    agentState: 'READY',
  })
})

test('buildHomeAgents prefers same-timestamp BUSY over stale READY for the same session', () => {
  const agents = buildHomeAgents([
    {
      source: 'control',
      workers: [
        worker({
          sessionName: '开发工程师-地雄星',
          sessionExists: true,
          agentState: 'READY',
          updatedAt: '2026-04-22T10:00:00+08:00',
        }),
      ],
    },
    {
      source: 'development',
      workers: [
        worker({
          sessionName: '开发工程师-地雄星',
          sessionExists: true,
          agentState: 'BUSY',
          updatedAt: '2026-04-22T10:00:00+08:00',
        }),
      ],
    },
  ])

  expect(agents).toHaveLength(1)
  expect(agents[0]).toMatchObject({
    source: 'development',
    sessionName: '开发工程师-地雄星',
    agentState: 'BUSY',
  })
})

test('buildHomeAgents shows prelaunch routing workers as STARTING', () => {
  const agents = buildHomeAgents(
    [
      {
        source: 'routing',
        workers: [
          worker({
            sessionName: '路由器-天伤星',
            sessionExists: false,
            agentState: 'STARTING',
            healthStatus: 'unknown',
            status: 'pending',
            workflowStage: 'pending',
          }),
        ],
      },
    ],
    'stage.a01.start',
  )
  expect(agents).toHaveLength(1)
  expect(agents[0]).toMatchObject({
    source: 'routing',
    sessionName: '路由器-天伤星',
    healthStatus: 'unknown',
    agentState: 'STARTING',
  })
})

test('buildHomeAgents keeps stable role order when heartbeat freshness changes', () => {
  const first = buildHomeAgents(
    [
      {
        source: 'development',
        workers: [
          worker({
            workerId: 'development-review-审核员',
            sessionName: '审核员-地奇星',
            sessionExists: true,
            agentState: 'BUSY',
            lastHeartbeatAt: '2026-04-22T10:00:03+08:00',
          }),
          worker({
            workerId: 'development-developer',
            sessionName: '开发工程师-天贵星',
            sessionExists: true,
            agentState: 'READY',
            lastHeartbeatAt: '2026-04-22T10:00:01+08:00',
          }),
          worker({
            workerId: 'development-review-测试工程师',
            sessionName: '测试工程师-亢金龙',
            sessionExists: true,
            agentState: 'STARTING',
            lastHeartbeatAt: '2026-04-22T10:00:02+08:00',
          }),
        ],
      },
    ],
    'stage.a07.start',
  )
  const second = buildHomeAgents(
    [
      {
        source: 'development',
        workers: [
          worker({
            workerId: 'development-review-审核员',
            sessionName: '审核员-地奇星',
            sessionExists: true,
            agentState: 'READY',
            lastHeartbeatAt: '2026-04-22T10:00:01+08:00',
          }),
          worker({
            workerId: 'development-developer',
            sessionName: '开发工程师-天贵星',
            sessionExists: true,
            agentState: 'BUSY',
            lastHeartbeatAt: '2026-04-22T10:00:03+08:00',
          }),
          worker({
            workerId: 'development-review-测试工程师',
            sessionName: '测试工程师-亢金龙',
            sessionExists: true,
            agentState: 'DEAD',
            lastHeartbeatAt: '2026-04-22T10:00:02+08:00',
          }),
        ],
      },
    ],
    'stage.a07.start',
  )

  const expectedOrder = ['开发工程师-天贵星', '测试工程师-亢金龙', '审核员-地奇星']
  expect(first.map((agent) => agent.sessionName)).toEqual(expectedOrder)
  expect(second.map((agent) => agent.sessionName)).toEqual(expectedOrder)
})

test('buildHomeAgents sorts design workers by fixed main and reviewer role order', () => {
  const agents = buildHomeAgents(
    [
      {
        source: 'design',
        workers: [
          worker({ workerId: 'detailed-design-review-审核员', sessionName: '审核员-天机星', sessionExists: true }),
          worker({ workerId: 'detailed-design-review-架构师', sessionName: '架构师-地速星', sessionExists: true }),
          worker({ workerId: 'detailed-design-analyst', sessionName: '需求分析师-天佑星', sessionExists: true }),
          worker({ workerId: 'detailed-design-review-测试工程师', sessionName: '测试工程师-亢金龙', sessionExists: true }),
          worker({ workerId: 'detailed-design-review-开发工程师', sessionName: '开发工程师-天贵星', sessionExists: true }),
        ],
      },
    ],
    'stage.a05.start',
  )

  expect(agents.map((agent) => agent.sessionName)).toEqual([
    '需求分析师-天佑星',
    '开发工程师-天贵星',
    '测试工程师-亢金龙',
    '架构师-地速星',
    '审核员-天机星',
  ])
})

test('buildHomeAgents sorts development reviewers by fixed role order', () => {
  const agents = buildHomeAgents(
    [
      {
        source: 'development',
        workers: [
          worker({ workerId: 'development-review-架构师', sessionName: '架构师-地速星', sessionExists: true }),
          worker({ workerId: 'development-review-审核员', sessionName: '审核员-地奇星', sessionExists: true }),
          worker({ workerId: 'development-review-需求分析师', sessionName: '需求分析师-天机星', sessionExists: true }),
          worker({ workerId: 'development-developer', sessionName: '开发工程师-天贵星', sessionExists: true }),
          worker({ workerId: 'development-review-测试工程师', sessionName: '测试工程师-亢金龙', sessionExists: true }),
        ],
      },
    ],
    'stage.a07.start',
  )

  expect(agents.map((agent) => agent.sessionName)).toEqual([
    '开发工程师-天贵星',
    '需求分析师-天机星',
    '测试工程师-亢金龙',
    '审核员-地奇星',
    '架构师-地速星',
  ])
})

test('buildHomeAgents falls back to stable session name order without worker id', () => {
  const agents = buildHomeAgents(
    [
      {
        source: 'design',
        workers: [
          worker({
            sessionName: '自定义角色-B',
            sessionExists: true,
            lastHeartbeatAt: '2026-04-22T10:00:02+08:00',
          }),
          worker({
            sessionName: '自定义角色-A',
            sessionExists: true,
            lastHeartbeatAt: '2026-04-22T10:00:01+08:00',
          }),
        ],
      },
    ],
    'stage.a05.start',
  )

  expect(agents.map((agent) => agent.sessionName)).toEqual(['自定义角色-A', '自定义角色-B'])
})
