import { useKeyboard } from '@opentui/solid'
import { For, Show, createEffect, createMemo } from 'solid-js'
import { clearPromptDraft, readPromptDraft, rememberPromptValue } from '../promptMemory'
import { PromptTextarea } from './PromptTextarea'

type Props = {
  title: string
  defaultValue?: string
  draftKey?: string
  focusToken?: string
  focused?: boolean
  mode?: 'singleline' | 'multiline'
  hintLines?: string[]
  textareaHeight?: number
  rememberHistory?: boolean
  showSubmitHelper?: boolean
  onBack?: () => void
  onSubmit: (value: string) => void
}

export function PromptInputPanel(props: Props) {
  const multiline = createMemo(() => props.mode !== 'singleline')
  const textareaHeight = createMemo(() => Math.max(1, props.textareaHeight ?? (multiline() ? 5 : 1)))
  const helperText = createMemo(() =>
    props.showSubmitHelper && multiline() ? 'Enter 提交，Shift+Enter / Meta+Enter / Ctrl+J 换行' : ''
  )
  const promptToken = createMemo(() => `${props.focusToken ?? ''}:${props.draftKey ?? ''}:${props.title}`)
  let submittedPromptToken = ''

  createEffect(() => {
    promptToken()
    submittedPromptToken = ''
  })

  const submitValue = (value: string) => {
    const token = promptToken()
    if (submittedPromptToken === token) return false
    submittedPromptToken = token
    props.onSubmit(value)
    return true
  }

  const submitCurrentDraft = () => {
    const value = readPromptDraft(props.draftKey) || props.defaultValue || ''
    if (!submitValue(value)) return
    if (props.rememberHistory !== false) {
      rememberPromptValue(props.draftKey, value)
    } else {
      clearPromptDraft(props.draftKey)
    }
  }

  const submitBack = () => {
    const token = promptToken()
    if (submittedPromptToken === token) return
    submittedPromptToken = token
    props.onBack?.()
  }

  useKeyboard((event) => {
    if (props.focused && props.onBack && event.name === 'escape') {
      event.preventDefault()
      event.stopPropagation()
      submitBack()
      return
    }
    const isSubmitKey =
      ['return', 'linefeed', 'enter'].includes(event.name) ||
      event.raw === '\r' ||
      event.raw === '\n' ||
      event.sequence === '\r' ||
      event.sequence === '\n'
    if (!isSubmitKey) return
    if (event.shift || event.meta || event.ctrl) return
    if (props.focused === false && !readPromptDraft(props.draftKey).trim()) return
    event.preventDefault()
    event.stopPropagation()
    submitCurrentDraft()
  })

  return (
    <box flexDirection="column" paddingLeft={1} paddingRight={1} paddingTop={1} width="100%">
      <text>{props.title}</text>
      <Show when={props.onBack}>
        <box
          flexDirection="row"
          gap={1}
          onMouseUp={(event) => {
            event.preventDefault()
            event.stopPropagation()
            submitBack()
          }}
        >
          <text fg="#00d2ff">[上一步]</text>
          <text fg="#888888">Esc</text>
        </box>
      </Show>
      <Show when={helperText()}>
        <text fg="#888888">{helperText()}</text>
      </Show>
      <For each={props.hintLines ?? []}>{(line) => <text fg="#888888">{line}</text>}</For>
      <PromptTextarea
        initialValue={props.defaultValue ?? ''}
        draftKey={props.draftKey}
        focusToken={props.focusToken}
        focused={props.focused}
        multiline={multiline()}
        height={textareaHeight()}
        minHeight={textareaHeight()}
        maxHeight={textareaHeight()}
        rememberHistory={props.rememberHistory}
        placeholder=""
        onSubmit={submitValue}
      />
    </box>
  )
}
