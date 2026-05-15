import { For, Show, createMemo, createSignal, onCleanup } from 'solid-js'
import type { AppSnapshot, HitlSnapshot, HomeAgentItem } from '../types'

type Props = {
  snapshot: AppSnapshot
  hitl: HitlSnapshot
  agents: HomeAgentItem[]
}

const HOME_AGENT_SPINNER_FRAMES = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']

function AgentStatusMarker(props: { agentState: string }) {
  const [frameIndex, setFrameIndex] = createSignal(0)
  const normalizedState = createMemo(() => String(props.agentState || '').trim().toUpperCase())
  const isBusy = createMemo(() => normalizedState() === 'BUSY')
  const timer = setInterval(() => {
    if (!isBusy()) return
    setFrameIndex((prev) => (prev + 1) % HOME_AGENT_SPINNER_FRAMES.length)
  }, 400)
  onCleanup(() => clearInterval(timer))

  const marker = createMemo(() => (isBusy() ? HOME_AGENT_SPINNER_FRAMES[frameIndex()] : '•'))
  const color = createMemo(() => {
    if (normalizedState() === 'BUSY') return '#f7c948'
    if (normalizedState() === 'DEAD') return '#ff5d5d'
    if (normalizedState() === 'READY') return '#00d2ff'
    if (normalizedState() === 'STARTING') return '#888888'
    return '#888888'
  })

  const markerText = createMemo(() => {
    if (normalizedState() === 'DEAD') return '❌'
    return marker()
  })

  return <text fg={color()}>{markerText()}</text>
}

function agentSummary(agent: HomeAgentItem): string {
  const stateText = `${agent.healthStatus}/${agent.agentState}`
  return [agent.sessionName, agent.agentConfigLabel, stateText].filter(Boolean).join(' | ')
}

export function HomeRoute(props: Props) {
  return (
    <box flexDirection="column" gap={1} paddingLeft={1} paddingRight={1} flexGrow={1}>
      <text>总览</text>
      <text fg="#888888">当前阶段: {props.snapshot.activeStageLabel || '等待中'}</text>
      <text fg="#888888">{`项目目录: ${props.snapshot.projectDir || '(unset)'}`}</text>
      <Show when={props.snapshot.requirementName}>
        <text fg="#888888">{`需求名称: ${props.snapshot.requirementName}`}</text>
      </Show>
      <box borderStyle="single" paddingLeft={1} paddingRight={1} paddingTop={1} paddingBottom={1} flexDirection="column">
        <text>待处理 HITL</text>
        <text fg={props.hitl.pending ? '#f7c948' : '#888888'}>{props.hitl.pending ? '存在待处理 HITL' : '当前没有待处理 HITL'}</text>
        <Show when={props.hitl.questionPath}>
          <text fg="#888888">question: {props.hitl.questionPath}</text>
        </Show>
        <Show when={props.hitl.attachCommand}>
          <text fg="#f7c948">{props.hitl.attachCommand}</text>
        </Show>
        <Show when={props.hitl.pending}>
          <text fg="#888888">Ctrl+L 查看完整日志</text>
        </Show>
      </box>
      <Show when={props.snapshot.pendingAttention}>
        <box borderStyle="single" paddingLeft={1} paddingRight={1} paddingTop={1} paddingBottom={1} flexDirection="column">
          <text>人工提醒</text>
          <text fg="#f7c948">{`macOS 提醒中: ${props.snapshot.pendingAttentionReason || '待处理人工输入'}`}</text>
          <Show when={props.snapshot.pendingAttentionSince}>
            <text fg="#888888">{`since: ${props.snapshot.pendingAttentionSince}`}</text>
          </Show>
        </box>
      </Show>
      <box borderStyle="single" paddingLeft={1} paddingRight={1} paddingTop={1} paddingBottom={1} flexDirection="column">
        <text>智能体状态</text>
        <Show when={props.agents.length > 0} fallback={<text fg="#888888">当前没有可显示的智能体状态。</text>}>
          <For each={props.agents}>
            {(agent) => (
              <box flexDirection="column" marginTop={1}>
                <box flexDirection="row" gap={1}>
                  <AgentStatusMarker agentState={agent.agentState} />
                  <text>{agentSummary(agent)}</text>
                </box>
                <text fg="#888888">{agent.attachCommand}</text>
              </box>
            )}
          </For>
        </Show>
      </box>
    </box>
  )
}
