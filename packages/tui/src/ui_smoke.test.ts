import { expect, test } from 'bun:test'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'

test('ui component files exist with expected exports', () => {
  const root = import.meta.dir
  const files = [
    ['ui/PromptTextarea.tsx', 'PromptTextarea'],
    ['ui/PromptInputPanel.tsx', 'PromptInputPanel'],
    ['ui/DialogSelect.tsx', 'DialogSelect'],
    ['ui/DialogPrompt.tsx', 'DialogPrompt'],
    ['ui/DialogConfirm.tsx', 'DialogConfirm'],
  ] as const
  for (const [relativePath, exportName] of files) {
    const content = readFileSync(join(root, relativePath), 'utf8')
    expect(content.includes(`export function ${exportName}`)).toBe(true)
  }
})

test('PromptInputPanel centralizes title, optional helper lines, and textarea wiring for text prompts', () => {
  const content = readFileSync(join(import.meta.dir, 'ui/PromptInputPanel.tsx'), 'utf8')
  expect(content.includes("type Props = {")).toBe(true)
  expect(content.includes("mode?: 'singleline' | 'multiline'")).toBe(true)
  expect(content.includes('hintLines?: string[]')).toBe(true)
  expect(content.includes('textareaHeight?: number')).toBe(true)
  expect(content.includes('rememberHistory?: boolean')).toBe(true)
  expect(content.includes('showSubmitHelper?: boolean')).toBe(true)
  expect(content.includes('onBack?: () => void')).toBe(true)
  expect(content.includes('readPromptDraft')).toBe(true)
  expect(content.includes('rememberPromptValue')).toBe(true)
  expect(content.includes('submittedPromptToken')).toBe(true)
  expect(content.includes("props.focused && props.onBack && event.name === 'escape'")).toBe(true)
  expect(content.includes('const submitBack = () => {')).toBe(true)
  expect(content.includes('props.onBack?.()')).toBe(true)
  expect(content.includes('<text fg="#00d2ff">[上一步]</text>')).toBe(true)
  expect(content.includes("props.showSubmitHelper && multiline() ? 'Enter 提交，Shift+Enter / Meta+Enter / Ctrl+J 换行' : ''")).toBe(true)
  expect(content.includes('<PromptTextarea')).toBe(true)
  expect(content.includes('focusToken={props.focusToken}')).toBe(true)
  expect(content.includes('focused={props.focused}')).toBe(true)
  expect(content.includes('height={textareaHeight()}')).toBe(true)
  expect(content.includes('rememberHistory={props.rememberHistory}')).toBe(true)
  expect(content.includes('onSubmit={submitValue}')).toBe(true)
  expect(content.includes('<For each={props.hintLines ?? []}>')).toBe(true)
})

test('PromptInputPanel catches Enter key events and submits the current draft once', () => {
  const content = readFileSync(join(import.meta.dir, 'ui/PromptInputPanel.tsx'), 'utf8')
  expect(content.includes("['return', 'linefeed', 'enter'].includes(event.name)")).toBe(true)
  expect(content.includes("event.raw === '\\r'")).toBe(true)
  expect(content.includes("event.sequence === '\\n'")).toBe(true)
  expect(content.includes('if (event.shift || event.meta || event.ctrl) return')).toBe(true)
  expect(content.includes("if (props.focused === false && !readPromptDraft(props.draftKey).trim()) return")).toBe(true)
  expect(content.includes('event.stopPropagation()')).toBe(true)
  expect(content.includes('submitCurrentDraft()')).toBe(true)
})

test('DialogPrompt delegates to PromptInputPanel instead of assembling its own helper and textarea layout', () => {
  const content = readFileSync(join(import.meta.dir, 'ui/DialogPrompt.tsx'), 'utf8')
  expect(content.includes("import { PromptInputPanel } from './PromptInputPanel'")).toBe(true)
  expect(content.includes('<PromptInputPanel')).toBe(true)
  expect(content.includes("mode={props.multiline ? 'multiline' : 'singleline'}")).toBe(true)
  expect(content.includes('showSubmitHelper={Boolean(props.multiline)}')).toBe(true)
  expect(content.includes('onBack?: () => void')).toBe(true)
  expect(content.includes('onBack={props.onBack}')).toBe(true)
  expect(content.includes('<PromptTextarea')).toBe(false)
})

test('TUI startup enables mouse capture, alternate-screen mode, and disables noisy console overlay', () => {
  const content = readFileSync(join(import.meta.dir, 'index.tsx'), 'utf8')
  expect(content.includes('useMouse: true')).toBe(true)
  expect(content.includes('useKittyKeyboard: { disambiguate: true, alternateKeys: true, allKeysAsEscapes: true }')).toBe(true)
  expect(content.includes("screenMode: 'alternate-screen'")).toBe(true)
  expect(content.includes("consoleMode: 'disabled'")).toBe(true)
  expect(content.includes('openConsoleOnError: false')).toBe(true)
  expect(content.includes('consoleOptions: {')).toBe(true)
  expect(content.includes("keyBindings: [{ name: 'y', ctrl: true, action: 'copy-selection' }]")).toBe(true)
  expect(content.includes('onCopySelection: (text) => {')).toBe(true)
  expect(content.includes('void copyToClipboard(text)')).toBe(true)
})

test('App document preview wiring references the active prompt preview signal', () => {
  const content = readFileSync(join(import.meta.dir, 'app.tsx'), 'utf8')
  expect(content.includes('const promptPreview = createMemo<DocumentPreviewState | null>')).toBe(true)
  expect(content.includes('if (!promptPreview()) setDocumentPreviewOpen(false)')).toBe(true)
  expect(content.includes('dialogPreview')).toBe(false)
})

test('PromptTextarea binds Shift+Enter and Ctrl+J to newline for multiline input', () => {
  const content = readFileSync(join(import.meta.dir, 'ui/PromptTextarea.tsx'), 'utf8')
  expect(content.includes("{ name: 'return', shift: true, action: 'newline' }")).toBe(true)
  expect(content.includes("{ name: 'linefeed', shift: true, action: 'newline' }")).toBe(true)
  expect(content.includes("{ name: 'j', ctrl: true, action: 'newline' }")).toBe(true)
})

test('PromptTextarea submits return and linefeed key events', () => {
  const content = readFileSync(join(import.meta.dir, 'ui/PromptTextarea.tsx'), 'utf8')
  expect(content.includes("{ name: 'return', action: 'submit' }")).toBe(true)
  expect(content.includes("{ name: 'linefeed', action: 'submit' }")).toBe(true)
})

test('PromptTextarea binds Command+Delete to delete the current line', () => {
  const content = readFileSync(join(import.meta.dir, 'ui/PromptTextarea.tsx'), 'utf8')
  expect(content.includes("{ name: 'backspace', super: true, action: 'delete-line' }")).toBe(true)
  expect(content.includes("{ name: 'delete', super: true, action: 'delete-line' }")).toBe(true)
})

test('PromptTextarea explicitly focuses the textarea on mount and reset', () => {
  const content = readFileSync(join(import.meta.dir, 'ui/PromptTextarea.tsx'), 'utf8')
  expect(content.includes('textarea.focus()')).toBe(true)
  expect(content.includes('textarea.gotoLineEnd()')).toBe(true)
  expect(content.includes('textarea.blur()')).toBe(true)
  expect(content.includes('props.focusToken')).toBe(true)
  expect(content.includes('const timer = setInterval(() => {')).toBe(true)
  expect(content.includes('scrollY: textarea.scrollY')).toBe(true)
  expect(content.includes('const scrollTextareaBy = (delta: number) => {')).toBe(true)
  expect(content.includes('textarea.scrollY = Math.max(0, Math.min(maxScroll, textarea.scrollY + delta))')).toBe(true)
})

test('PromptTextarea persists draft and remembers submitted history for prompt reuse', () => {
  const content = readFileSync(join(import.meta.dir, 'ui/PromptTextarea.tsx'), 'utf8')
  expect(content.includes('readPromptDraft')).toBe(true)
  expect(content.includes('writePromptDraft')).toBe(true)
  expect(content.includes('rememberPromptValue')).toBe(true)
  expect(content.includes('props.rememberHistory === false')).toBe(true)
  expect(content.includes('if (props.rememberHistory !== false) rememberPromptValue')).toBe(true)
  expect(content.includes('<box flexDirection="row" width="100%">')).toBe(true)
  expect(content.includes("{row.active ? '#' : '|'}")).toBe(true)
  expect(content.includes('onMouseScroll={(event) => {')).toBe(true)
  expect(content.includes('scrollTextareaBy(direction === \'up\' ? -1 : 1)')).toBe(true)
})

test('PromptTextarea submits the live textarea contents instead of a possibly stale signal', () => {
  const content = readFileSync(join(import.meta.dir, 'ui/PromptTextarea.tsx'), 'utf8')
  expect(content.includes('const submittedValue = textarea?.plainText ?? value()')).toBe(true)
  expect(content.includes('rememberPromptValue(props.draftKey, submittedValue)')).toBe(true)
  expect(content.includes('props.onSubmit(submittedValue)')).toBe(true)
})

test('DialogSelect uses terminal-aware sizing and width clipping so long option lists remain navigable inside the dialog border', () => {
  const content = readFileSync(join(import.meta.dir, 'ui/DialogSelect.tsx'), 'utf8')
  expect(content.includes('type { ScrollBoxRenderable }')).toBe(true)
  expect(content.includes('useTerminalDimensions')).toBe(true)
  expect(content.includes('active?: boolean')).toBe(true)
  expect(content.includes("if (props.active === false) return")).toBe(true)
  expect(content.includes('const maxVisibleRows = createMemo(() => Math.max(2, Math.floor(dimensions().height / 2) - 6))')).toBe(true)
  expect(content.includes('const visibleRows = createMemo(() => (shouldScroll() ? maxVisibleRows() : expandedRows()))')).toBe(true)
  expect(content.includes('width="100%"')).toBe(true)
  expect(content.includes('overflow="hidden" wrapMode="none"')).toBe(true)
  expect(content.includes('fallback={')).toBe(true)
  expect(content.includes('<box flexDirection="column" width="100%" height={visibleRows()} minHeight={visibleRows()}>')).toBe(true)
  expect(content.includes('<scrollbox')).toBe(true)
  expect(content.includes('height={visibleRows()}')).toBe(true)
  expect(content.includes('maxHeight={visibleRows()}')).toBe(true)
})

test('DialogConfirm forwards dialog active state to the shared select renderer', () => {
  const content = readFileSync(join(import.meta.dir, 'ui/DialogConfirm.tsx'), 'utf8')
  expect(content.includes('active?: boolean')).toBe(true)
  expect(content.includes('active={props.active}')).toBe(true)
  expect(content.includes('allowBack?: boolean')).toBe(true)
  expect(content.includes('backValue?: string')).toBe(true)
  expect(content.includes('withPromptBackOption')).toBe(true)
  expect(content.includes("value === backValue() ? backValue() : value === 'yes'")).toBe(true)
})

test('clipboard helper writes trimmed text to the system clipboard', () => {
  const content = readFileSync(join(import.meta.dir, 'clipboard.ts'), 'utf8')
  expect(content.includes("import clipboard from 'clipboardy'")).toBe(true)
  expect(content.includes('export async function copyToClipboard(text: string): Promise<boolean> {')).toBe(true)
  expect(content.includes('const value = text.trim()')).toBe(true)
  expect(content.includes('await clipboard.write(value)')).toBe(true)
})
