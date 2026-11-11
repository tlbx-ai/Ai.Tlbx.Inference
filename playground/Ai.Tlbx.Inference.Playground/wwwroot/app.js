const $ = id => document.getElementById(id);
let mode = 'chat', busy = false, sessionId = localStorage.getItem('hostedAgentSessionId');
let chatHistory = [];
let chatCostUsd = Number(localStorage.getItem('playgroundChatCostUsd') || 0);
let chatTokens = Number(localStorage.getItem('playgroundChatTokens') || 0);
let agentCosts = JSON.parse(localStorage.getItem('playgroundAgentCosts') || '{}');
let usageRefreshTimer = null;
let usageRefreshAttempts = 0;
const showStatus = text => { $('status').textContent = text; };
function updateMeter() {
  const values = Object.values(agentCosts);
  const usd = chatCostUsd + values.reduce((sum, item) => sum + (item.costUsd || 0), 0);
  const tokens = chatTokens + values.reduce((sum, item) => sum + (item.tokens || 0), 0);
  $('costTotal').textContent = '$' + usd.toLocaleString('de-DE', { minimumFractionDigits: 6, maximumFractionDigits: 6 });
  $('tokenTotal').textContent = tokens.toLocaleString('de-DE') + ' Tokens · reine Modell-Tokenkosten';
  $('budgetTotal').textContent = 'Geschätzte Tokenkosten · kein Ausgabenlimit';
  localStorage.setItem('playgroundAgentCosts', JSON.stringify(agentCosts));
}
async function refreshAgentUsage() {
  if (!sessionId) return;
  const id = sessionId;
  const model = agentCosts[id]?.model || $('agentModel').value.trim();
  const result = await api('/api/agents/' + encodeURIComponent(id) + '/usage?model=' + encodeURIComponent(model));
  if (sessionId !== id || mode !== 'agent') return;
  agentCosts[id] = { model, costUsd: result.tokenCostUsd == null ? null : Number(result.tokenCostUsd), tokens: result.usage.totalTokens,
    complete: result.complete };
  $('agentUsage').textContent = result.usage.inputTokens + ' Input · ' + result.usage.outputTokens +
    ' Output · ' + (result.tokenCostUsd == null ? 'Preis unbekannt' : '$' + Number(result.tokenCostUsd).toFixed(6)) + (result.complete ? '' : ' · Abrechnung noch unvollständig');
  updateMeter();
  if (usageRefreshTimer) clearTimeout(usageRefreshTimer);
  if (!result.complete && sessionId === id && ++usageRefreshAttempts < 10)
    usageRefreshTimer = setTimeout(() => refreshAgentUsage().catch(e => showStatus('Kosten konnten nicht geladen werden: ' + e.message)), 2500);
}
function addMessage(role, text = '') {
  const empty = $('messages').querySelector('.empty'); if (empty) empty.remove();
  const node = document.createElement('div'); node.className = 'message ' + (role === 'Du' ? 'user' : 'agent');
  const label = document.createElement('div'); label.className = 'role'; label.textContent = role;
  const body = document.createElement('div'); body.textContent = text;
  node.append(label, body); $('messages').append(node); $('messages').scrollTop = $('messages').scrollHeight;
  return body;
}
function setBusy(value) { busy = value; $('send').disabled = value; $('newSession').disabled = value; $('deleteSession').disabled = value || !sessionId; $('cancelTurn').disabled = !value || mode !== 'agent' || !sessionId; }
function switchMode(next) {
  if (busy) return;
  if (usageRefreshTimer) clearTimeout(usageRefreshTimer);
  usageRefreshAttempts = 0;
  mode = next;
  $('chatTab').classList.toggle('active', next === 'chat'); $('agentTab').classList.toggle('active', next === 'agent');
  $('chatSettings').hidden = next !== 'chat'; $('agentSettings').hidden = next !== 'agent';
  $('messages').replaceChildren(); $('activity').hidden = true;
  if (next === 'chat') chatHistory.forEach(m => addMessage(m.role === 'user' ? 'Du' : 'Modell', m.content));
  else if (sessionId) { restoreItems(); refreshAgentUsage().catch(e => showStatus('Kosten konnten nicht geladen werden: ' + e.message)); }
  showStatus('Bereit');
}
async function api(path, method = 'GET', body) {
  const response = await fetch(path, { method, headers: body ? { 'Content-Type': 'application/json' } : {}, body: body ? JSON.stringify(body) : undefined });
  if (!response.ok) { const raw = await response.text(); throw new Error(raw); }
  return response.status === 204 ? null : response.json();
}
async function newSession(manageBusy = true) {
  if (manageBusy) setBusy(true); showStatus('Session wird erstellt…');
  try {
    const result = await api('/api/agents', 'POST', {
      model: $('agentModel').value.trim(), instructions: $('agentInstructions').value.trim() || null,
      reasoningEffort: $('reasoning').value || null, enableWebSearch: $('webSearch').checked,
      allowNetworkAccess: $('network').checked, mcpLabel: $('mcpLabel').value.trim() || null,
      mcpUrl: $('mcpUrl').value.trim() || null, mcpAllowedTools: $('mcpTools').value.trim() || null
    });
    sessionId = result.id; localStorage.setItem('hostedAgentSessionId', sessionId);
    agentCosts[sessionId] = { model: $('agentModel').value.trim(), costUsd: 0, tokens: 0, complete: true }; updateMeter();
    $('sessionInfo').textContent = sessionId + ' · ' + result.status;
    $('messages').replaceChildren(); $('activity').hidden = true; showStatus('Session bereit');
  } catch (e) { showStatus('Fehler: ' + e.message); if (!manageBusy) throw e; } finally { if (manageBusy) setBusy(false); }
}
async function restoreItems() {
  try {
    const items = await api('/api/agents/' + encodeURIComponent(sessionId) + '/items');
    $('messages').replaceChildren();
    for (const item of items) {
      const raw = JSON.parse(item.rawJson);
      if (item.type === 'mcp_call') {
        addMessage('MCP', [raw.server_label, raw.name].filter(Boolean).join('.') + ' · ' + (raw.status || 'unbekannt'));
      } else if (item.type === 'command_execution') {
        addMessage('Bash', (raw.command || '') + '\n' + (item.text || ''));
      } else if (item.text) {
        addMessage(raw.role === 'user' ? 'Du' : 'Agent', item.text);
      }
    }
  } catch (e) { showStatus('Verlauf konnte nicht geladen werden: ' + e.message); }
}
function activity(event) {
  if (!event.type || event.type.includes('output_text')) return;
  $('activity').hidden = false;
  const raw = event.rawJson ? JSON.parse(event.rawJson) : event;
  const tool = raw.item || raw;
  const detail = tool.name || tool.command || '';
  const line = document.createElement('div');
  line.textContent = event.type + (detail ? ' · ' + detail : '') + (event.error ? ' · ' + event.error : '');
  $('activity').append(line); $('activity').scrollTop = $('activity').scrollHeight;
}
async function runAgent(message) {
  if (!sessionId) { await newSession(false); addMessage('Du', message); }
  if (!sessionId) throw new Error('Keine Agent-Session erstellt.');
  setBusy(true);
  const output = addMessage('Agent'); let rendered = '';
  const response = await fetch('/api/agents/' + encodeURIComponent(sessionId) + '/turn', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message })
  });
  if (!response.ok) throw new Error(await response.text());
  const reader = response.body.getReader(), decoder = new TextDecoder(); let pending = '';
  while (true) {
    const { value, done } = await reader.read(); if (done) break;
    pending += decoder.decode(value, { stream: true });
    const lines = pending.split('\n'); pending = lines.pop();
    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line); activity(event);
      if (event.type === 'playground.error') throw new Error(event.error);
      if (event.type === 'agent.session.turn.output_text.delta' && event.delta) rendered += event.delta;
      if (event.type === 'agent.session.turn.output_text.done' && event.text) rendered = event.text;
      output.textContent = rendered; $('messages').scrollTop = $('messages').scrollHeight;
    }
  }
  if (!rendered) await restoreItems();
  usageRefreshAttempts = 0;
  await refreshAgentUsage();
}
$('chatTab').onclick = () => switchMode('chat'); $('agentTab').onclick = () => switchMode('agent');
$('newSession').onclick = () => newSession();
$('cancelTurn').onclick = async () => { if (!sessionId || !busy || mode !== 'agent') return; try { await api('/api/agents/' + encodeURIComponent(sessionId) + '/cancel', 'POST'); showStatus('Abbruch angefordert'); } catch (e) { showStatus('Abbruch fehlgeschlagen: ' + e.message); } };
$('deleteSession').onclick = async () => { if (!sessionId || busy) return; try { await api('/api/agents/' + encodeURIComponent(sessionId), 'DELETE'); sessionId = null; localStorage.removeItem('hostedAgentSessionId'); $('sessionInfo').textContent = 'Noch keine Session'; $('messages').replaceChildren(); $('activity').hidden = true; $('agentUsage').textContent = ''; if (usageRefreshTimer) clearTimeout(usageRefreshTimer); } catch (e) { showStatus('Fehler: ' + e.message); } setBusy(false); };
$('clearChat').onclick = () => { chatHistory = []; $('messages').replaceChildren(); showStatus('Verlauf geleert'); };
$('composer').onsubmit = async e => {
  e.preventDefault(); if (busy) return;
  const message = $('prompt').value.trim(); if (!message) return;
  $('prompt').value = ''; addMessage('Du', message); setBusy(true); showStatus('Antwort läuft…');
  try {
    if (mode === 'chat') {
      const messages = [...chatHistory, { role: 'user', content: message }];
      const answer = await api('/api/chat', 'POST', { model: $('chatModel').value, systemMessage: $('systemMessage').value || null, messages });
      chatHistory = [...messages, { role: 'assistant', content: answer.content }]; addMessage('Modell', answer.content);
      chatCostUsd += Number(answer.tokenCostUsd); chatTokens += answer.usage.totalTokens;
      localStorage.setItem('playgroundChatCostUsd', chatCostUsd);
      localStorage.setItem('playgroundChatTokens', chatTokens); updateMeter();
      showStatus('Bereit');
    } else { await runAgent(message); showStatus('Turn abgeschlossen'); }
  } catch (error) {
    showStatus('Fehler: ' + error.message);
    if (mode === 'agent' && sessionId) await restoreItems();
  }
  finally { setBusy(false); }
};
if (sessionId) { $('sessionInfo').textContent = sessionId + ' · gespeichert'; $('deleteSession').disabled = false; }
updateMeter();
