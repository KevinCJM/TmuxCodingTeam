import type { TextareaRenderable } from '@opentui/core'
import { For, createEffect, createMemo, createSignal, onCleanup, onMount } from 'solid-js'
import { clearPromptDraft, latestPromptHistory, readPromptDraft, rememberPromptValue, writePromptDraft } from '../promptMemory'

type ScrollableTextareaRenderable = TextareaRenderable & {
  scrollY: number
}

type Props = {
  placeholder?: string
  initialValue?: string
  draftKey?: string
  focusToken?: string
  focused?: boolean
  resetToken?: string | number
  multiline?: boolean
  height?: number
  minHeight?: number
  maxHeight?: number
  rememberHistory?: boolean
  onSubmit: (value: string) => void
}

export function PromptTextarea(props: Props) {
  let textarea: ScrollableTextareaRenderable | undefined
  const resolveInitialValue = () =>
    readPromptDraft(props.draftKey) || props.initialValue || (props.rememberHistory === false ? '' : latestPromptHistory(props.draftKey)) || ''
  const [value, setValue] = createSignal(resolveInitialValue())
  const minHeight = () => props.minHeight ?? (props.multiline ? 3 : 1)
  const maxHeight = () => props.maxHeight ?? (props.multiline ? 6 : 1)
  const textareaHeight = createMemo(() => Math.max(1, props.height ?? maxHeight()))
  const [scrollState, setScrollState] = createSignal({
    scrollY: 0,
    totalLines: textareaHeight(),
    visibleLines: textareaHeight(),
  })

  const syncScrollState = () => {
    if (!textarea || textarea.isDestroyed) return
    setScrollState({
      scrollY: textarea.scrollY,
      totalLines: Math.max(textarea.virtualLineCount, textareaHeight()),
      visibleLines: textareaHeight(),
    })
  }

  const focusTextarea = () => {
    if (!textarea || textarea.isDestroyed) return
    textarea.focus()
    textarea.gotoLineEnd()
    syncScrollState()
  }

  const scrollTextareaBy = (delta: number) => {
    if (!textarea || textarea.isDestroyed) return
    const maxScroll = Math.max(0, textarea.virtualLineCount - textareaHeight())
    if (maxScroll <= 0) {
      syncScrollState()
      return
    }
    textarea.scrollY = Math.max(0, Math.min(maxScroll, textarea.scrollY + delta))
    syncScrollState()
  }

  onMount(() => {
    setTimeout(() => {
      focusTextarea()
    }, 1)
    const timer = setInterval(() => {
      syncScrollState()
    }, 60)
    onCleanup(() => clearInterval(timer))
  })

  createEffect(() => {
    props.resetToken
    if (!textarea) return
    const nextValue = resolveInitialValue()
    textarea.setText(nextValue)
    setValue(nextValue)
    focusTextarea()
    syncScrollState()
  })

  createEffect(() => {
    props.focusToken
    if (!textarea) return
    if (props.focused === false) {
      textarea.blur()
      return
    }
    queueMicrotask(() => {
      focusTextarea()
    })
  })

  const scrollbarRows = createMemo(() => {
    const state = scrollState()
    const rows = Math.max(1, state.visibleLines)
    const total = Math.max(rows, state.totalLines)
    if (total <= rows) {
      return Array.from({ length: rows }, () => ({ active: false }))
    }
    const thumbSize = Math.max(1, Math.round((rows / total) * rows))
    const maxThumbStart = Math.max(0, rows - thumbSize)
    const maxScroll = Math.max(1, total - rows)
    const thumbStart = Math.min(maxThumbStart, Math.round((state.scrollY / maxScroll) * maxThumbStart))
    return Array.from({ length: rows }, (_, index) => ({
      active: index >= thumbStart && index < thumbStart + thumbSize,
    }))
  })

  return (
    <box flexDirection="row" width="100%">
      <box
        flexGrow={1}
        onMouseScroll={(event) => {
          const direction = event.scroll?.direction
          if (!direction) return
          event.preventDefault()
          event.stopPropagation()
          focusTextarea()
          scrollTextareaBy(direction === 'up' ? -1 : 1)
        }}
      >
        <textarea
          ref={(val: TextareaRenderable) => {
            textarea = val as ScrollableTextareaRenderable
          }}
          initialValue={props.initialValue ?? ''}
          placeholder={props.placeholder ?? '输入内容'}
          placeholderColor="#666666"
          textColor="#ffffff"
          focusedTextColor="#ffffff"
          cursorColor="#ffffff"
          height={props.height}
          minHeight={minHeight()}
          maxHeight={maxHeight()}
          onContentChange={() => {
            const nextValue = textarea?.plainText ?? ''
            setValue(nextValue)
            writePromptDraft(props.draftKey, nextValue)
            syncScrollState()
          }}
          onCursorChange={() => {
            syncScrollState()
          }}
          keyBindings={[
            { name: 'return', shift: true, action: 'newline' },
            { name: 'linefeed', shift: true, action: 'newline' },
            { name: 'return', meta: true, action: 'newline' },
            { name: 'linefeed', meta: true, action: 'newline' },
            { name: 'j', ctrl: true, action: 'newline' },
            { name: 'backspace', super: true, action: 'delete-line' },
            { name: 'delete', super: true, action: 'delete-line' },
            { name: 'return', action: 'submit' },
            { name: 'linefeed', action: 'submit' },
          ]}
          onSubmit={() => {
            const submittedValue = textarea?.plainText ?? value()
            if (props.rememberHistory !== false) rememberPromptValue(props.draftKey, submittedValue)
            clearPromptDraft(props.draftKey)
            props.onSubmit(submittedValue)
          }}
        />
      </box>
      <box flexDirection="column" width={1} marginLeft={1}>
        <For each={scrollbarRows()}>
          {(row) => <text fg={row.active ? '#00d2ff' : '#555555'}>{row.active ? '#' : '|'}</text>}
        </For>
      </box>
    </box>
  )
}
