import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const source = readFileSync(new URL('../Ai.Tlbx.Inference.Playground/wwwroot/app.js', import.meta.url), 'utf8');
assert.doesNotMatch(source, /Ã|Â|â€/, 'Browser source must remain UTF-8');
function fixture({ failCreate = false, complete = true, cost = 0.01 } = {}) {
  const elements = new Map(), storage = new Map(), timers = new Map();
  let sequence = 0, releaseTurn, turnRequested = false, cancelRequested = false;
  class Element {
    disabled = false; hidden = false; value = ''; textContent = ''; children = [];
    scrollHeight = 0; className = ''; classList = { toggle() {} };
    append(...nodes) { this.children.push(...nodes); }
    replaceChildren(...nodes) { this.children = nodes; }
    querySelector() { return null; }
  }
  const element = id => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); };
  element('agentModel').value = 'gpt-6-astra';
  const result = value => ({ ok: true, status: 200, json: async () => value });
  const context = vm.createContext({
    document: { getElementById: element, createElement: () => new Element() },
    localStorage: { getItem: key => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, value), removeItem: key => storage.delete(key) },
    setTimeout: callback => { const id = ++sequence; timers.set(id, callback); return id; },
    clearTimeout: id => timers.delete(id), TextDecoder,
    fetch: async (path) => {
      if (path === '/api/agents') {
        if (failCreate) return { ok: false, text: async () => 'Session creation failed' };
        return result({ id: 'sess_mock', status: 'idle' });
      }
      if (path.endsWith('/cancel')) { cancelRequested = true; return { ok: true, status: 204 }; }
      if (path.endsWith('/turn')) {
        turnRequested = true;
        assert.equal(element('send').disabled, true, 'Send must remain disabled throughout automatic session creation');
        return await new Promise(resolve => { releaseTurn = () => {
          let delivered = false;
          resolve({ ok: true, body: { getReader: () => ({ read: async () => {
            if (delivered) return { done: true };
            delivered = true;
            return { done: false, value: new TextEncoder().encode(JSON.stringify({ type: 'agent.session.turn.output_text.delta', delta: 'answer' }) + '\n') };
          } }) } });
        }; });
      }
      if (path.includes('/usage?')) return result({ usage: { inputTokens: 5, outputTokens: 3, totalTokens: 8 }, tokenCostUsd: cost, complete });
      if (path.endsWith('/items')) return result([]);
      throw new Error('Unexpected request: ' + path);
    }
  });
  vm.runInContext(source, context);
  vm.runInContext("switchMode('agent')", context);
  return { context, element, timers, requested: () => turnRequested, cancelled: () => cancelRequested, release: () => releaseTurn() };
}

{
  const f = fixture({ complete: false, cost: null });
  f.element('prompt').value = 'first task';
  const submission = f.element('composer').onsubmit({ preventDefault() {} });
  for (let i = 0; i < 20 && !f.requested(); i++) await Promise.resolve();
  assert.equal(f.requested(), true);
  assert.equal(f.element('send').disabled, true);
  assert.ok(f.element('messages').children.some(node => node.children.some(child => child.textContent === 'first task')), 'First user task remains visible after automatic session creation');
  assert.equal(f.element('cancelTurn').disabled, false);
  await f.element('cancelTurn').onclick();
  assert.equal(f.cancelled(), true, 'Abort requests cancellation of remote work');
  f.release(); await submission;
  assert.equal(f.element('send').disabled, false);
  assert.match(f.element('agentUsage').textContent, /Preis unbekannt/);
  let retries = 0;
  while (f.timers.size) {
    assert.ok(retries < 10, 'Delayed billing polling must stop');
    const [id, callback] = f.timers.entries().next().value;
    f.timers.delete(id); await callback(); retries++;
  }
  assert.equal(retries, 9);
}
{
  const f = fixture({ failCreate: true });
  f.element('prompt').value = 'first task';
  await f.element('composer').onsubmit({ preventDefault() {} });
  assert.match(f.element('status').textContent, /Fehler: Session creation failed/);
  assert.equal(f.requested(), false, 'A failed session must not submit a turn or report completion');
  assert.equal(f.element('send').disabled, false);
}
console.log('Playground contracts passed: automatic-session isolation, first-message retention, creation failure, unknown pricing and bounded billing refresh.');
