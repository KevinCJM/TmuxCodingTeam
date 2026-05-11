import { expect, test } from 'bun:test'
import { existsSync } from 'node:fs'
import { join } from 'node:path'
import { BackendClient, readPythonPath, repoRoot } from './client'

test('BackendClient can be constructed', () => {
  const client = new BackendClient()
  expect(client).toBeInstanceOf(BackendClient)
})

test('backend client resolves repo root and python config from repository root', () => {
  const root = repoRoot()
  expect(existsSync(join(root, 'U01_common_config.py'))).toBe(true)
  expect(readPythonPath().length).toBeGreaterThan(0)
})

test('BackendClient treats non-JSON stdout lines as log events instead of crashing', () => {
  const events: Array<{ type: string; payload: Record<string, unknown> }> = []
  const client = new BackendClient() as any
  client.subscribe((event: { type: string; payload: Record<string, unknown> }) => {
    events.push(event)
  })
  client.handleLine('警告：文件不存在 -> /tmp/demo')
  expect(events).toHaveLength(1)
  expect(events[0]?.type).toBe('log.append')
  expect(String(events[0]?.payload.text ?? '')).toContain('警告：文件不存在')
})

test('BackendClient stop tears down child process and completes pending requests', async () => {
  const client = new BackendClient() as any
  const signals: string[] = []
  let resolved = false
  client.process = {
    kill: (signal?: string) => {
      signals.push(signal || '')
    },
  }
  client.pending.set('req_1', {
    resolve: () => {
      resolved = true
    },
    reject: () => {
      throw new Error('shutdown should not reject pending requests')
    },
  })
  client.subscribe(() => undefined)

  await client.stop()

  expect(signals).toEqual(['SIGTERM'])
  expect(resolved).toBe(true)
  expect(client.process).toBeUndefined()
  expect(client.pending.size).toBe(0)
  expect(client.listeners.size).toBe(0)
})

test('BackendClient stop waits for backend exit before completing', async () => {
  const client = new BackendClient() as any
  const signals: string[] = []
  let resolveExited!: () => void
  client.process = {
    kill: (signal?: string) => {
      signals.push(signal || '')
    },
    exited: new Promise<void>((resolve) => {
      resolveExited = resolve
    }),
  }

  let stopped = false
  const stopping = client.stop(50).then(() => {
    stopped = true
  })
  await Promise.resolve()

  expect(signals).toEqual(['SIGTERM'])
  expect(stopped).toBe(false)

  resolveExited()
  await stopping

  expect(stopped).toBe(true)
  expect(signals).toEqual(['SIGTERM'])
})

test('BackendClient stop escalates to SIGKILL when backend does not exit', async () => {
  const client = new BackendClient() as any
  const signals: string[] = []
  client.process = {
    kill: (signal?: string) => {
      signals.push(signal || '')
    },
    exited: new Promise(() => undefined),
  }

  client.stop(1)
  await new Promise((resolve) => setTimeout(resolve, 5))

  expect(signals).toEqual(['SIGTERM', 'SIGKILL'])
  client.stoppingProcess = undefined
  client.clearProcessExitHandlerIfIdle()
})

test('BackendClient process exit fallback sends SIGTERM only', () => {
  const client = new BackendClient() as any
  const signals: string[] = []
  client.process = {
    kill: (signal?: string) => {
      signals.push(signal || '')
    },
  }

  client.ensureProcessExitHandler()
  client.processExitHandler()
  client.process = undefined
  client.clearProcessExitHandlerIfIdle()

  expect(signals).toEqual(['SIGTERM'])
})
