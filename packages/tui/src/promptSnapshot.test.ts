import { describe, expect, it } from 'bun:test'
import { promptStateFromSnapshot } from './promptSnapshot'

describe('promptStateFromSnapshot', () => {
  it('restores a pending bootstrap prompt', () => {
    const restored = promptStateFromSnapshot(
      {
        pending: true,
        prompt_id: 'prompt_1',
        prompt_type: 'select',
        payload: {
          title: 'HITL: 开发工程师 需要人工介入',
          is_hitl: true,
        },
      },
      (promptType, payload) => `${promptType}:${String(payload.title ?? '')}`,
    )

    expect(restored).toEqual({
      id: 'prompt_1',
      promptType: 'select',
      payload: {
        title: 'HITL: 开发工程师 需要人工介入',
        is_hitl: true,
      },
      draftKey: 'select:HITL: 开发工程师 需要人工介入',
    })
  })

  it('ignores non-pending bootstrap prompt snapshots', () => {
    const restored = promptStateFromSnapshot(
      {
        pending: false,
        prompt_id: 'prompt_1',
        prompt_type: 'select',
        payload: { title: 'ignored' },
      },
      () => 'unused',
    )

    expect(restored).toBeNull()
  })
})
