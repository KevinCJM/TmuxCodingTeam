import { createCliRenderer } from '@opentui/core'
import { render } from '@opentui/solid'
import { App, stopBackendClient } from './app'
import { copyToClipboard } from './clipboard'

type StartupRoute = 'home' | 'routing' | 'requirements' | 'review' | 'design' | 'task-split' | 'development' | 'overall-review' | 'control'
type ShutdownSignal = 'SIGINT' | 'SIGTERM' | 'SIGHUP'

function parseStartupArgs(argv: string[]) {
  let route: StartupRoute | undefined
  let action = ''
  let initialArgv: string[] = []
  for (let index = 0; index < argv.length; index += 1) {
    const item = argv[index]
    if (item === '--route') {
      const value = argv[index + 1]
      if (
        value === 'home'
        || value === 'routing'
        || value === 'requirements'
        || value === 'review'
        || value === 'design'
        || value === 'task-split'
        || value === 'development'
        || value === 'overall-review'
        || value === 'control'
      ) {
        route = value
      }
      index += 1
      continue
    }
    if (item === '--action') {
      action = argv[index + 1] ?? ''
      index += 1
      continue
    }
    if (item === '--argv-json') {
      const raw = argv[index + 1] ?? '[]'
      try {
        const parsed = JSON.parse(raw)
        if (Array.isArray(parsed)) {
          initialArgv = parsed.map((entry) => String(entry))
        }
      } catch {
        initialArgv = []
      }
      index += 1
    }
  }
  return {
    route,
    action: action || undefined,
    initialArgv,
  }
}

const renderer = await createCliRenderer({
  targetFps: 60,
  exitOnCtrlC: false,
  useMouse: true,
  useKittyKeyboard: { disambiguate: true, alternateKeys: true, allKeysAsEscapes: true },
  autoFocus: true,
  screenMode: 'alternate-screen',
  externalOutputMode: 'passthrough',
  consoleMode: 'disabled',
  openConsoleOnError: false,
  consoleOptions: {
    keyBindings: [{ name: 'y', ctrl: true, action: 'copy-selection' }],
    onCopySelection: (text) => {
      if (!text) return
      void copyToClipboard(text)
    },
  },
})

let shutdownStarted = false

function exitCodeForSignal(signal: ShutdownSignal) {
  if (signal === 'SIGINT') return 130
  if (signal === 'SIGTERM') return 143
  if (signal === 'SIGHUP') return 129
  return 1
}

async function shutdownFromSignal(signal: ShutdownSignal) {
  if (shutdownStarted) return
  shutdownStarted = true
  try {
    renderer.destroy()
  } catch {
    // Renderer may already be shutting down.
  }
  await stopBackendClient()
  process.exit(exitCodeForSignal(signal))
}

for (const signal of ['SIGINT', 'SIGTERM', 'SIGHUP'] as const) {
  process.on(signal, () => {
    void shutdownFromSignal(signal)
  })
}

const startup = parseStartupArgs(Bun.argv.slice(2))

await render(
  () => (
    <App
      initialRoute={startup.route}
      initialAction={startup.action}
      initialArgv={startup.initialArgv}
      onExitRequest={() => shutdownFromSignal('SIGINT')}
    />
  ),
  renderer,
)
