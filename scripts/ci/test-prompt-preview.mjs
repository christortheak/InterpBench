import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';

// Execute the actual bundled event handler with a controllable pending response.
// This checks asynchronous context ownership, not visual/browser qualification.
const html = await readFile(new URL('../../web/index.html', import.meta.url), 'utf8');
const start = html.indexOf('  if ($("ex-load-prompts")) $("ex-load-prompts").onclick =');
const end = html.indexOf('  // Save & Pin removed', start);
assert.ok(start >= 0 && end > start);
const source = html.slice(start, end);

function setup() {
  let resolve;
  const response = new Promise(r => { resolve = r; });
  const nodes = {
    'ex-load-prompts': {disabled: false}, 'ex-prompts': {value: 'prompts/input.jsonl'},
    'ex-prompt-text': {value: 'existing preview'}, 'ex-prompt-preview-status': {textContent: ''}
  };
  const calls = [];
  const promptPreviews = new Map();
  const context = vm.createContext({
    $: id => nodes[id], e: {name: 'study', workspaceRoot: '/workspace'}, promptPreviews,
    post: (path, body) => { calls.push({path, body}); return response; }
  });
  vm.runInContext(source, context);
  return {nodes, calls, promptPreviews, finish: resolve, run: () => nodes['ex-load-prompts'].onclick()};
}
const success = {ok: true, json: async () => ({ok: true, prompts: {
  text: 'loaded source', count: 1, promptsFileSHA256: 'a'.repeat(64)
}})};
{
  const test = setup();
  const pending = test.run();
  assert.equal(test.calls[0].path, '/api/experiment/prompts/load');
  assert.deepEqual(JSON.parse(JSON.stringify(test.calls[0].body)), {
    name: 'study', workspaceRoot: '/workspace', file: 'prompts/input.jsonl'
  });
  test.finish(success);
  await pending;
  assert.equal(test.nodes['ex-prompt-text'].value, 'loaded source');
  assert.match(test.nodes['ex-prompt-preview-status'].textContent, /Loaded 1 prompts/);
}
{
  const test = setup();
  const pending = test.run();
  // A new render may name the same source path in a different study/workspace.
  test.nodes['ex-load-prompts'] = {disabled: false};
  test.nodes['ex-prompt-text'] = {value: 'another editor'};
  test.nodes['ex-prompt-preview-status'] = {textContent: 'another status'};
  test.finish(success);
  await pending;
  assert.equal(test.nodes['ex-prompt-text'].value, 'another editor');
  assert.equal(test.nodes['ex-prompt-preview-status'].textContent, 'another status');
}
{
  const test = setup();
  const pending = test.run();
  test.nodes['ex-prompts'].value = 'prompts/different.jsonl';
  test.finish(success);
  await pending;
  assert.equal(test.nodes['ex-prompt-text'].value, 'existing preview');
}
{
  const test = setup();
  const pending = test.run();
  test.finish({ok: false, json: async () => ({ok: false, error: 'Read refused.', repairAction: 'Reconnect.'})});
  await pending;
  assert.equal(test.nodes['ex-prompt-text'].value, '');
  assert.equal(test.nodes['ex-prompt-preview-status'].textContent, 'Read refused. Reconnect.');
}
console.log('PASS: explicit prompt preview request, response, context switch and refusal behavior');
