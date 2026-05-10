import { For, Show } from 'solid-js'
import type { DevelopmentSnapshot } from '../types'

type Props = {
  snapshot: DevelopmentSnapshot
}

export function DevelopmentRoute(props: Props) {
  return (
    <box flexDirection="column" gap={1} paddingLeft={1} paddingRight={1} flexGrow={1}>
      <text>任务开发</text>
      <text fg="#888888">{`项目目录: ${props.snapshot.projectDir || '(unset)'}`}</text>
      <text fg="#888888">{`需求名称: ${props.snapshot.requirementName || '(unset)'}`}</text>
      <box borderStyle="single" paddingLeft={1} paddingRight={1} flexDirection="column">
        <text>阶段文档</text>
        <Show when={props.snapshot.files.length > 0} fallback={<text fg="#888888">尚未发现任务开发产物。</text>}>
          <For each={props.snapshot.files}>
            {(item) => (
              <text fg={item.exists ? '#00d2ff' : '#888888'}>{`${item.label}: ${item.exists ? 'ready' : 'missing'}`}</text>
            )}
          </For>
        </Show>
      </box>
      <box borderStyle="single" paddingLeft={1} paddingRight={1} flexDirection="column">
        <text>里程碑与任务清单</text>
        <Show when={props.snapshot.milestones.length > 0} fallback={<text fg="#888888">当前没有可展示的任务进度。</text>}>
          <For each={props.snapshot.milestones}>
            {(milestone) => {
              const isCurrent = () => milestone.key === props.snapshot.currentMilestoneKey && Boolean(props.snapshot.currentMilestoneKey)
              return (
                <box flexDirection="column">
                  <text fg={isCurrent() ? '#7be495' : '#00d2ff'}>{`${milestone.completed ? '☑' : '☐'} ${milestone.key}${isCurrent() ? ' (当前)' : ''}`}</text>
                  <For each={milestone.tasks}>
                    {(task) => (
                      <text fg={task.completed ? '#7be495' : '#f7c948'}>{`  ${task.completed ? '☑' : '☐'} ${task.key}`}</text>
                    )}
                  </For>
                </box>
              )
            }}
          </For>
        </Show>
      </box>
      <box borderStyle="single" paddingLeft={1} paddingRight={1} flexDirection="column">
        <text>Workers</text>
        <Show when={props.snapshot.workers.length > 0} fallback={<text fg="#888888">当前没有任务开发 workers。</text>}>
          <For each={props.snapshot.workers}>
            {(worker) => {
              const dispatch = () => worker.dispatchState ? ` | dispatch:${worker.dispatchState}` : ''
              const reason = () => worker.dispatchReason ? ` (${worker.dispatchReason})` : ''
              return (
                <text>{`${worker.sessionName} | ${worker.workflowStage}/${worker.agentState} | ${worker.healthStatus}${dispatch()}${reason()}`}</text>
              )
            }}
          </For>
        </Show>
      </box>
      <Show when={props.snapshot.blockers.length > 0}>
        <box borderStyle="single" paddingLeft={1} paddingRight={1} flexDirection="column">
          <text>阻塞项</text>
          <For each={props.snapshot.blockers}>
            {(item) => <text fg="#f7c948">{item}</text>}
          </For>
        </box>
      </Show>
    </box>
  )
}
