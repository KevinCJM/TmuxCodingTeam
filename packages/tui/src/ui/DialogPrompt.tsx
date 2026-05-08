import { PromptInputPanel } from './PromptInputPanel'

type Props = {
  title: string
  defaultValue?: string
  draftKey?: string
  focusToken?: string
  focused?: boolean
  multiline?: boolean
  hintLines?: string[]
  onBack?: () => void
  onSubmit: (value: string) => void
}

export function DialogPrompt(props: Props) {
  return (
    <PromptInputPanel
      title={props.title}
      defaultValue={props.defaultValue}
      draftKey={props.draftKey}
      focusToken={props.focusToken}
      focused={props.focused}
      mode={props.multiline ? 'multiline' : 'singleline'}
      hintLines={props.hintLines}
      showSubmitHelper={Boolean(props.multiline)}
      onBack={props.onBack}
      onSubmit={props.onSubmit}
    />
  )
}
