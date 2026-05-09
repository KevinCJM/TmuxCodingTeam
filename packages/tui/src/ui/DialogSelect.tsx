import type { ScrollBoxRenderable } from '@opentui/core'
import { useKeyboard, useTerminalDimensions } from '@opentui/solid'
import { createEffect, createMemo, createSignal, For, Show } from 'solid-js'

export type SelectOption = {
  value: string
  label: string
}

type Props = {
  title: string
  options: SelectOption[]
  defaultValue?: string
  hintLines?: string[]
  active?: boolean
  onSubmit: (value: string) => void
}

export function DialogSelect(props: Props) {
  const dimensions = useTerminalDimensions()
  const initialIndex = Math.max(0, props.options.findIndex((item) => item.value === props.defaultValue))
  const [selected, setSelected] = createSignal(initialIndex)
  const current = createMemo(() => props.options[selected()] ?? props.options[0])
  const expandedRows = createMemo(() => Math.max(1, props.options.length))
  const maxVisibleRows = createMemo(() => Math.max(2, Math.floor(dimensions().height / 2) - 6))
  const shouldScroll = createMemo(() => expandedRows() > maxVisibleRows())
  const visibleRows = createMemo(() => (shouldScroll() ? maxVisibleRows() : expandedRows()))
  let optionScrollbox: ScrollBoxRenderable | undefined

  createEffect(() => {
    const scrollbox = optionScrollbox
    if (!shouldScroll() || !scrollbox || scrollbox.isDestroyed) return
    const currentIndex = selected()
    const viewportStart = scrollbox.scrollTop
    const viewportEnd = viewportStart + visibleRows() - 1
    if (currentIndex < viewportStart) {
      scrollbox.scrollTop = currentIndex
      return
    }
    if (currentIndex > viewportEnd) {
      scrollbox.scrollTop = currentIndex - visibleRows() + 1
    }
  })

  useKeyboard((event) => {
    if (props.active === false) return
    if (event.name === 'up' || event.name === 'k') {
      event.preventDefault()
      setSelected((prev) => (prev <= 0 ? props.options.length - 1 : prev - 1))
      return
    }
    if (event.name === 'down' || event.name === 'j') {
      event.preventDefault()
      setSelected((prev) => (prev >= props.options.length - 1 ? 0 : prev + 1))
      return
    }
    if (event.name === 'pageup') {
      event.preventDefault()
      setSelected((prev) => Math.max(0, prev - Math.max(1, visibleRows() - 1)))
      return
    }
    if (event.name === 'pagedown') {
      event.preventDefault()
      setSelected((prev) => Math.min(props.options.length - 1, prev + Math.max(1, visibleRows() - 1)))
      return
    }
    if (event.name === 'home') {
      event.preventDefault()
      setSelected(0)
      return
    }
    if (event.name === 'end') {
      event.preventDefault()
      setSelected(Math.max(0, props.options.length - 1))
      return
    }
    if (event.name === 'return') {
      event.preventDefault()
      if (current()) props.onSubmit(current()!.value)
    }
  })

  const renderOption = (item: SelectOption, index: () => number) => {
    const active = () => index() === selected()
    return (
      <box
        flexDirection="row"
        width="100%"
        maxWidth="100%"
        gap={1}
        height={1}
        minHeight={1}
        paddingLeft={1}
        paddingRight={1}
        backgroundColor={active() ? '#07141c' : undefined}
      >
        <text flexShrink={0} fg={active() ? '#00d2ff' : '#666666'}>{active() ? '›' : ' '}</text>
        <text fg={active() ? '#00d2ff' : '#ffffff'} flexGrow={1} overflow="hidden" wrapMode="none">{item.label}</text>
      </box>
    )
  }

  return (
    <box flexDirection="column" gap={1} paddingLeft={1} paddingRight={1} width="100%">
      <text>{props.title}</text>
      <Show when={(props.hintLines ?? []).length > 0}>
        <box flexDirection="column" width="100%">
          <For each={props.hintLines ?? []}>
            {(line) => <text fg="#f7c948" overflow="hidden" wrapMode="none">{line}</text>}
          </For>
        </box>
      </Show>
      <Show
        when={shouldScroll()}
        fallback={
          <box flexDirection="column" width="100%" height={visibleRows()} minHeight={visibleRows()}>
            <For each={props.options}>
              {(item, index) => renderOption(item, index)}
            </For>
          </box>
        }
      >
        <scrollbox
          ref={(value: ScrollBoxRenderable) => {
            optionScrollbox = value
          }}
          height={visibleRows()}
          minHeight={visibleRows()}
          maxHeight={visibleRows()}
        >
          <box flexDirection="column" width="100%">
            <For each={props.options}>
              {(item, index) => renderOption(item, index)}
            </For>
          </box>
        </scrollbox>
      </Show>
    </box>
  )
}
