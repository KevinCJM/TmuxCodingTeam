export type BootstrapPromptState = {
  id: string
  promptType: string
  payload: Record<string, unknown>
  draftKey: string
}

export function promptStateFromSnapshot(
  snapshot: unknown,
  buildDraftKey: (promptType: string, payload: Record<string, unknown>) => string,
): BootstrapPromptState | null {
  if (!snapshot || typeof snapshot !== 'object') return null
  const value = snapshot as Record<string, unknown>
  if (!Boolean(value.pending)) return null
  const promptId = String(value.prompt_id ?? value.promptId ?? '').trim()
  const promptType = String(value.prompt_type ?? value.promptType ?? '').trim()
  const payload = value.payload && typeof value.payload === 'object'
    ? value.payload as Record<string, unknown>
    : {}
  if (!promptId || !promptType) return null
  return {
    id: promptId,
    promptType,
    payload,
    draftKey: buildDraftKey(promptType, payload),
  }
}
