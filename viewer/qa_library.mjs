// Headless phone-size QA for the move library (2026-11-17).
// Opens each URL in headless Chrome at a phone viewport through the Chrome
// DevTools Protocol (no npm deps; Node >= 22 has WebSocket), records console
// errors, exceptions and failed requests, and for viewer pages checks that the
// model loaded (duration > 0), the clock advances while playing and the source
// video advances too. Screenshots go to --out.
//
//   node viewer/qa_library.mjs --base http://192.168.0.157:8787/webapps/move-library/ \
//        --out qa_shots index.html "viewer.html?glb=..." ...
import { spawn } from 'node:child_process'
import { existsSync, mkdtempSync, writeFileSync, mkdirSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'

const args = process.argv.slice(2)
let base = '', outDir = 'qa_shots'
const urls = []
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--base') base = args[++i]
  else if (args[i] === '--out') outDir = args[++i]
  else urls.push(args[i])
}
mkdirSync(outDir, { recursive: true })
const chrome = [process.env.CHROME, 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
  path.join(process.env.LOCALAPPDATA || '', 'Google\\Chrome\\Application\\chrome.exe'),
  '/usr/bin/google-chrome', '/usr/bin/chromium'].filter(Boolean).find((p) => existsSync(p))
const port = 9300 + Math.floor(Math.random() * 500)
const profile = mkdtempSync(path.join(os.tmpdir(), 'mlqa-'))
const proc = spawn(chrome, ['--headless=new', '--remote-debugging-port=' + port, '--user-data-dir=' + profile,
  '--no-first-run', '--autoplay-policy=no-user-gesture-required', '--enable-unsafe-swiftshader', '--use-angle=swiftshader',
  'about:blank'], { stdio: 'ignore' })
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
let tab
for (let i = 0; i < 50 && !tab; i++) {
  await sleep(200)
  try { tab = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).find((t) => t.type === 'page') } catch {}
}
const ws = new WebSocket(tab.webSocketDebuggerUrl)
await new Promise((r) => ws.addEventListener('open', r))
let id = 0
const pending = new Map()
let errors = [], failed = []
ws.addEventListener('message', (ev) => {
  const m = JSON.parse(ev.data)
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id) }
  if (m.method === 'Runtime.exceptionThrown') errors.push(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text)
  if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') errors.push(m.params.args.map((a) => a.value || a.description).join(' '))
  if (m.method === 'Log.entryAdded' && m.params.entry.level === 'error') errors.push(m.params.entry.text + ' ' + (m.params.entry.url || ''))
  if (m.method === 'Network.loadingFailed' && !m.params.canceled) failed.push(m.params.errorText)
  if (m.method === 'Network.responseReceived' && m.params.response.status >= 400) failed.push(m.params.response.status + ' ' + m.params.response.url)
})
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })) })
const evaluate = async (expr) => (await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true })).result?.result?.value
await send('Runtime.enable'); await send('Page.enable'); await send('Log.enable'); await send('Network.enable')
await send('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 2, mobile: true })
await send('Emulation.setTouchEmulationEnabled', { enabled: true })

const report = []
for (const [n, u] of urls.entries()) {
  errors = []; failed = []
  const url = base + u
  console.error('[qa] ' + n + ' ' + u.slice(0, 60))
  await send('Page.navigate', { url })
  const isViewer = u.startsWith('viewer.html')
  let check = {}
  if (isViewer) {
    let dur = 0
    // loaded = the viewer wrote the real duration into #btime (the scrub max starts at a placeholder 14)
    for (let i = 0; i < 60 && !(dur > 0); i++) { await sleep(500); dur = (await evaluate("document.getElementById('btime').textContent")) === '—' ? 0 : parseFloat(await evaluate("document.getElementById('scrub').max")) || 0 }
    for (let i = 0; i < 30 && u.includes('video=') && !(await evaluate("document.getElementById('vid').readyState >= 2")); i++) await sleep(500)
    const t0 = await evaluate("[document.getElementById('atime').textContent, document.getElementById('vid').currentTime, document.getElementById('vid').readyState, document.getElementById('vid').getAttribute('src')]")
    await sleep(3000)
    const t1 = await evaluate("[document.getElementById('atime').textContent, document.getElementById('vid').currentTime, document.getElementById('vid').readyState]")
    check = { duration_s: dur, clock: [t0[0], t1[0]], video: t0[3] ? { t: [t0[1], t1[1]], readyState: t1[2] } : null,
      model_loaded: dur > 0, clock_advances: t0[0] !== t1[0], video_advances: t0[3] ? t1[1] > t0[1] : null,
      loaderr: await evaluate("document.getElementById('loaderr') ? document.getElementById('loaderr').textContent : ''") }
  } else {
    await sleep(2500)
    check = { title: await evaluate('document.title'), links: await evaluate('document.querySelectorAll("a").length') }
  }
  const shot = await send('Page.captureScreenshot', { format: 'jpeg', quality: 70 })
  const file = path.join(outDir, `shot_${String(n).padStart(2, '0')}.jpg`)
  writeFileSync(file, Buffer.from(shot.result.data, 'base64'))
  report.push({ url: u.slice(0, 140), ...check, console_errors: errors, failed_requests: failed, shot: file })
}
console.log(JSON.stringify(report, null, 1))
ws.close(); proc.kill()
