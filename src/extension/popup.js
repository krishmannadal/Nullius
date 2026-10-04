document.addEventListener('DOMContentLoaded', async () => {
  const previewBox = document.getElementById('preview-box');
  const sourceLabel = document.getElementById('source-label');
  const pasteDetails = document.getElementById('paste-details');
  const pasteInput = document.getElementById('paste-input');
  const pasteBtn = document.getElementById('paste-btn');
  const pasteSelectionBtn = document.getElementById('paste-selection-btn');
  const scopeSelect = document.getElementById('scope-select');
  const scopeHelp = document.getElementById('scope-help');
  const quickBtn = document.getElementById('quick-btn');
  const fullBtn = document.getElementById('full-btn');
  const diagnosticsBtn = document.getElementById('diagnostics-btn');
  const previewDetails = document.getElementById('preview-details');
  const extensionVersion = chrome.runtime?.getManifest?.()?.version || 'unknown';
  document.getElementById('version-label').textContent = `v${extensionVersion}`;
  let lastDiagnostics = null;
  const refreshBtn = document.getElementById('refresh-btn');
  const connectionBtn = document.getElementById('connection-btn');
  const compareBtn = document.getElementById('compare-btn');
  const exportBtn = document.getElementById('export-btn');
  const comparisonArea = document.getElementById('comparison-area');
  const statusArea = document.getElementById('status-area');
  const resultsArea = document.getElementById('results-area');
  const endpoints = {
    quick: 'http://127.0.0.1:8000/verify/quick',
    full: 'http://127.0.0.1:8000/verify/full'
  };
  const timeouts = { quick: 5000, full: 60000, chatQuick: 15000, chatFull: 180000 };
  let capturedText = null;
  let captureSource = null;
  let captureScope = null;
  let capturedMessageCount = null;
  let busy = false;
  let inspection = null;
  let comparison = null;
  let comparisonExpired = false;
  const rules = ['max_entailment', 'noisy_or', 'weighted_by_retrieval', 'threshold_abstain', 'majority'];
  const ruleNames = {
    max_entailment: 'Max entailment', noisy_or: 'Noisy OR',
    weighted_by_retrieval: 'Rank weighted', threshold_abstain: 'Threshold + abstain',
    majority: 'Majority baseline'
  };

  function showStatus(msg, isError = false) {
    statusArea.style.display = 'block';
    statusArea.textContent = msg;
    statusArea.className = isError ? 'status-error' : 'status-info';
  }

  function setBusy(value) {
    busy = value;
    quickBtn.disabled = value || !capturedText;
    // The primary action can retry capture if the page finished loading after popup opening.
    fullBtn.disabled = value;
    diagnosticsBtn.disabled = value;
    refreshBtn.disabled = value;
    connectionBtn.disabled = value;
    pasteBtn.disabled = value;
    pasteInput.disabled = value;
    pasteSelectionBtn.disabled = value;
    scopeSelect.disabled = value;
    compareBtn.disabled = value || comparisonExpired || inspection?.kind !== 'full' ||
      !inspection.data.run_id || !inspection.data.config_hash || !inspection.data.claims.length;
    exportBtn.disabled = value || !inspection;
  }

  function clearInspection() {
    inspection = null;
    comparison = null;
    comparisonExpired = false;
    comparisonArea.replaceChildren();
    resultsArea.replaceChildren();
  }

  function describeScope() {
    const descriptions = {
      chat: 'All loaded ChatGPT answers in this conversation. Scroll up to load earlier answers before capturing.',
      latest: 'The full text of the latest completed ChatGPT answer.',
      selection: 'Highlight a passage in a completed ChatGPT answer before opening Nullius. Only that passage is inspected.',
      pasted: 'Inspect the pasted text shown in the preview. Choose another scope to return to ChatGPT.'
    };
    scopeHelp.textContent = descriptions[scopeSelect.value];
  }

  async function capturePage() {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab?.id || !tab.url || !['chatgpt.com', 'chat.openai.com'].includes(new URL(tab.url).hostname)) {
      throw new Error('Open a conversation on chatgpt.com, then click Analyze response.');
    }
    const scope = scopeSelect.value;
    await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ['content.js'] });
    const frames = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: selectedScope => globalThis.__nulliusCapture(selectedScope),
      args: [scope]
    });
    const response = frames.find(frame => frame.frameId === 0)?.result;
    lastDiagnostics = response?.diagnostics || null;
    if (!response?.success || typeof response.text !== 'string' || !response.text.trim()) {
      throw new Error(response?.error || 'Could not capture the latest ChatGPT answer. Refresh the page and try again.');
    }
    capturedText = response.text;
    captureSource = 'chatgpt-page';
    captureScope = scope;
    capturedMessageCount = Number.isInteger(response.message_count) ? response.message_count : null;
    sourceLabel.textContent = scope === 'chat' ? `Complete chat · ${capturedMessageCount ?? 'all loaded'} ChatGPT answers` :
      scope === 'latest' ? 'Latest ChatGPT answer · full text' : 'Selected ChatGPT passage only';
    previewBox.textContent = capturedText;
    return capturedText;
  }

  async function refresh() {
    if (busy) return;
    setBusy(true);
    clearInspection();
    capturedText = null;
    captureSource = null;
    captureScope = null;
    capturedMessageCount = null;
    sourceLabel.textContent = '';
    statusArea.style.display = 'none';
    try {
      if (scopeSelect.value === 'pasted') scopeSelect.value = 'chat';
      describeScope();
      await capturePage();
    } catch (error) {
      capturedText = null;
      previewBox.textContent = error.message;
      previewDetails.open = true;
      showStatus(error.message, true);
    } finally {
      setBusy(false);
    }
  }

  async function inspect(kind) {
    if (busy) return;
    setBusy(true);
    clearInspection();
    showStatus(kind === 'quick' ? 'Searching the reference corpus…' : 'Reading ChatGPT and analyzing its claims… Keep this popup open.');
    const button = kind === 'quick' ? quickBtn : fullBtn;
    button.textContent = kind === 'quick' ? 'Searching…' : 'Analyzing response…';
    let timeoutId;
    let stage = 'capture';
    try {
      // Page checks recapture; an explicitly chosen pasted answer keeps its previewed snapshot.
      const text = captureSource === 'pasted-answer' ? capturedText : await capturePage();
      stage = 'backend';
      if (kind === 'full') showStatus('Analyzing claims against the reference corpus… The first analysis may take longer while the model loads.');
      const controller = new AbortController();
      const timeout = captureScope === 'chat' ?
        (kind === 'quick' ? timeouts.chatQuick : timeouts.chatFull) : timeouts[kind];
      timeoutId = setTimeout(() => controller.abort(), timeout);
      const response = await fetch(endpoints[kind], {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
        body: JSON.stringify({ text, config_path: 'configs/extension.yaml', retrieve_k: 5,
          ...(kind === 'quick' ? { evidence_floor_score: 0.01 } : { stream: false }) }),
        signal: controller.signal
      });
      if (!response.ok) {
        const body = await response.text();
        let detail = body;
        try { detail = JSON.parse(body).detail || body; } catch { /* Plain-text server errors. */ }
        throw new Error(`Backend error (${response.status}): ${typeof detail === 'string' ? detail : JSON.stringify(detail)}`);
      }
      const data = await response.json();
      if (!data || !Array.isArray(data.claims) ||
          (kind === 'full' && (!Array.isArray(data.verdicts) || !data.evidence_by_claim))) {
        throw new Error('Backend returned an invalid inspection response. Update and restart Nullius.');
      }
      inspection = { kind, captured_text: text, capture_source: captureSource,
        analysis_scope: captureScope, message_count: capturedMessageCount, data };
      if (kind === 'quick') renderQuickResults(data);
      else renderFullResults(data);
      showStatus(kind === 'quick' ?
        'Evidence search complete. Click Analyze response to check the claims against the retrieved evidence.' :
        `Analysis complete · ${data.claims.length} claims extracted. Review each verdict and its evidence below.`);
    } catch (error) {
      clearInspection();
      if (stage === 'capture') {
        capturedText = null;
        captureSource = null;
        captureScope = null;
        capturedMessageCount = null;
        sourceLabel.textContent = '';
        previewBox.textContent = error.message;
        previewDetails.open = true;
        showStatus(error.message, true);
      } else if (error.name === 'AbortError') {
        showStatus(captureScope === 'chat' ?
          'Complete chat check timed out. Try Latest response or Selected text, and start the backend with --warm-full. Closing this request does not stop backend computation.' :
          'Request timed out. First-time model loading can take longer; warm the backend and try again. Closing this request does not stop backend computation.', true);
      } else if (error instanceof TypeError) {
        showStatus('Cannot connect to Nullius. Start the local backend on this computer at 127.0.0.1:8000, then use Test connection.', true);
      } else {
        showStatus(error.message, true);
      }
    } finally {
      clearTimeout(timeoutId);
      button.textContent = kind === 'quick' ? 'Search evidence only' : 'Analyze response';
      setBusy(false);
    }
  }

  function validateComparison(data) {
    if (!data || data.run_id !== inspection.data.run_id ||
        data.config_hash !== inspection.data.config_hash ||
        (data.git_sha ?? null) !== (inspection.data.git_sha ?? null) ||
        !data.comparisons || typeof data.comparisons !== 'object') {
      throw new Error('Rule comparison does not match this inspection. Click Analyze response again.');
    }
    const ids = inspection.data.claims.map(claim => claim.id);
    if (Object.keys(data.comparisons).length !== ids.length) {
      throw new Error('Backend returned an incomplete rule comparison.');
    }
    for (const id of ids) {
      for (const rule of rules) {
        const verdict = data.comparisons[id]?.[rule];
        if (!verdict || verdict.claim_id !== id || verdict.aggregator_name !== rule ||
            !['Supported', 'Contradicted', 'Insufficient', 'Abstain'].includes(verdict.label) ||
            !Number.isFinite(verdict.confidence) || verdict.confidence < 0 || verdict.confidence > 1 ||
            !Array.isArray(verdict.per_evidence) || !verdict.aggregation_trace ||
            !Array.isArray(verdict.aggregation_trace.decisive_evidence_ids) ||
            typeof verdict.aggregation_trace.rule !== 'string' ||
            typeof verdict.aggregation_trace.explanation !== 'string') {
          throw new Error('Backend returned an incomplete rule comparison.');
        }
        const original = inspection.data.verdicts.find(item => item.claim_id === id);
        const canonicalPair = pair => JSON.stringify(Object.entries(pair).sort(([a], [b]) => a.localeCompare(b)));
        if (!original || verdict.per_evidence.length !== original.per_evidence.length ||
            verdict.per_evidence.some((pair, index) => canonicalPair(pair) !== canonicalPair(original.per_evidence[index]))) {
          throw new Error('Comparison scores differ from the original inspection; the response was rejected.');
        }
      }
    }
  }

  function renderComparison(data) {
    comparisonArea.replaceChildren();
    const title = document.createElement('h2');
    title.textContent = 'Same evidence · different rules';
    comparisonArea.appendChild(title);
    const changed = inspection.data.claims.filter(claim =>
      new Set(rules.map(rule => data.comparisons[claim.id][rule].label)).size > 1
    ).length;
    const notice = document.createElement('p');
    notice.className = 'notice';
    notice.textContent = `${changed} of ${inspection.data.claims.length} claims receive different verdicts across these rule defaults. ` +
      'Every rule uses the same stored pairwise scores. Disagreement shows decision sensitivity; agreement does not prove truth.';
    comparisonArea.appendChild(notice);
    for (const claim of inspection.data.claims) {
      const card = document.createElement('section');
      card.className = 'claim-card';
      const claimText = document.createElement('p');
      claimText.className = 'claim-text';
      claimText.textContent = claim.text;
      card.appendChild(claimText);
      const table = document.createElement('table');
      const caption = document.createElement('caption');
      caption.textContent = 'Default rule settings; confidence is uncalibrated';
      table.appendChild(caption);
      const header = document.createElement('tr');
      for (const label of ['Rule', 'Verdict', 'Confidence']) {
        const cell = document.createElement('th');
        cell.scope = 'col'; cell.textContent = label; header.appendChild(cell);
      }
      const head = document.createElement('thead'); head.appendChild(header); table.appendChild(head);
      const body = document.createElement('tbody');
      for (const rule of rules) {
        const verdict = data.comparisons[claim.id][rule];
        const row = document.createElement('tr');
        for (const value of [ruleNames[rule], verdict.label, verdict.confidence.toFixed(3)]) {
          const cell = document.createElement('td'); cell.textContent = value; row.appendChild(cell);
        }
        body.appendChild(row);
        const details = document.createElement('details');
        const summary = document.createElement('summary');
        summary.textContent = `${ruleNames[rule]} · rationale and decisive evidence`;
        const trace = document.createElement('pre');
        trace.textContent = JSON.stringify(verdict.aggregation_trace, null, 2);
        details.append(summary, trace); card.appendChild(details);
      }
      table.appendChild(body); card.insertBefore(table, card.children[1] || null);
      comparisonArea.appendChild(card);
    }
  }

  compareBtn.addEventListener('click', async () => {
    if (busy || compareBtn.disabled) return;
    setBusy(true);
    showStatus('Comparing five rules on this inspection’s stored scores…');
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 5000);
    try {
      const response = await fetch('http://127.0.0.1:8000/reaggregate', {
        method: 'POST', headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
        // Never send client-supplied scores or a second copy of the response.
        body: JSON.stringify({ run_id: inspection.data.run_id, target_aggregators: rules, aggregator_configs: {} }),
        signal: controller.signal
      });
      if (response.status === 404) {
        comparisonExpired = true;
        throw new Error('This inspection expired or the backend restarted. Export your results, then click Analyze response again.');
      }
      if (!response.ok) throw new Error(`Rule comparison failed (${response.status}). Your original inspection is preserved.`);
      const data = await response.json();
      validateComparison(data);
      renderComparison(data);
      comparison = data;
      showStatus('Five rules compared on the same scores. No extraction, retrieval or model inference was repeated.');
    } catch (error) {
      showStatus(error.name === 'AbortError' ? 'Rule comparison timed out. Your inspection is preserved; retry when the backend is available.' : error.message, true);
    } finally {
      clearTimeout(timer);
      setBusy(false);
    }
  });

  exportBtn.addEventListener('click', () => {
    if (busy || !inspection) return;
    const report = {
      schema_version: 'nullius-inspection-export-v1',
      exported_at: new Date().toISOString(),
      extension_version: chrome.runtime?.getManifest?.()?.version || 'unknown',
      inspection_kind: inspection.kind,
      captured_text: inspection.captured_text,
      capture_source: inspection.capture_source,
      analysis_scope: inspection.analysis_scope,
      captured_message_count: inspection.message_count,
      original_result: inspection.data,
      comparison_request: comparison ? { target_aggregators: rules, aggregator_configs: {} } : null,
      rule_comparison: comparison,
      notice: 'Research inspection over a limited debug corpus. Scores are uncalibrated; this is not adjudicated gold or scientific validation.'
    };
    const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }));
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `nullius-${inspection.kind}-inspection.json`;
    anchor.hidden = true;
    document.body.appendChild(anchor); anchor.click(); anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    showStatus('Inspection exported locally. The file contains the captured answer; share it only when appropriate.');
  });

  function usePaste(selectedOnly) {
    if (busy) return;
    const text = selectedOnly ? pasteInput.value.slice(pasteInput.selectionStart, pasteInput.selectionEnd) : pasteInput.value;
    if (!text.trim()) {
      showStatus(selectedOnly ? 'Highlight a passage in the paste box first. The current preview has not changed.' :
        'Paste a completed answer first. The current preview has not changed.', true);
      return;
    }
    clearInspection();
    capturedText = text;
    captureSource = 'pasted-answer';
    captureScope = selectedOnly ? 'pasted-selection' : 'pasted-full';
    capturedMessageCount = null;
    scopeSelect.querySelector('[value="pasted"]').hidden = false;
    scopeSelect.value = 'pasted';
    describeScope();
    previewBox.textContent = capturedText;
    sourceLabel.textContent = selectedOnly ? 'Pasted answer · selected part only' : 'Pasted answer · entire text';
    showStatus('Preview ready. Click Analyze response to inspect this text. Read ChatGPT again returns to Complete chat.');
    setBusy(false);
  }
  pasteBtn.addEventListener('click', () => usePaste(false));
  pasteSelectionBtn.addEventListener('click', () => usePaste(true));
  diagnosticsBtn.addEventListener('click', async () => {
    const report = JSON.stringify({ extension_version: extensionVersion, scope: scopeSelect.value,
      capture: lastDiagnostics, error: capturedText ? null : previewBox.textContent }, null, 2);
    try {
      await navigator.clipboard.writeText(report);
      showStatus('Capture diagnostics copied. They contain page structure counts, not your conversation text.');
    } catch {
      showStatus(`Copy these capture diagnostics: ${report}`);
    }
  });
  scopeSelect.addEventListener('change', refresh);
  refreshBtn.addEventListener('click', refresh);
  quickBtn.addEventListener('click', () => inspect('quick'));
  fullBtn.addEventListener('click', () => inspect('full'));
  connectionBtn.addEventListener('click', async () => {
    if (busy) return;
    setBusy(true);
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 5000);
    try {
      const response = await fetch('http://127.0.0.1:8000/openapi.json', { signal: controller.signal });
      const schema = await response.json();
      if (!response.ok || !schema.paths?.['/verify/quick'] || !schema.paths?.['/verify/full']) {
        throw new Error('The local server does not provide the Nullius verification endpoints.');
      }
      showStatus('Connected to Nullius verification API. Click Analyze response to run the verifier.');
    } catch (error) {
      showStatus(`Backend unavailable. Run python -m scripts.run_extension_backend on this computer. ${error.message}`, true);
    } finally {
      clearTimeout(timer);
      setBusy(false);
    }
  });
  scopeSelect.value = 'chat';
  await refresh();

  // 4. Render Quick Check results (restored F1 behavior)
  function renderQuickResults(data) {
    // We strictly use document.createElement to avoid innerHTML injection vulnerabilities
    
    // Render counts badge summary
    const countsDiv = document.createElement('div');
    countsDiv.style.marginBottom = '12px';
    
    const countLabel = document.createElement('div');
    countLabel.className = 'label';
    countLabel.textContent = 'Status Summary:';
    countsDiv.appendChild(countLabel);

    if (data.counts) {
      for (const [status, count] of Object.entries(data.counts)) {
        if (count > 0 && !status.startsWith('total')) {
          const badge = document.createElement('span');
          badge.className = `badge badge-${status.replaceAll('_', '-')}`;
          badge.textContent = `${count} ${status.replaceAll('_', ' ')}`;
          badge.style.marginRight = '4px';
          countsDiv.appendChild(badge);
        }
      }
    }
    resultsArea.appendChild(countsDiv);

    // Render claims list
    const claimsLabel = document.createElement('div');
    claimsLabel.className = 'label';
    claimsLabel.textContent = 'Evidence Preview:';
    resultsArea.appendChild(claimsLabel);

    if (!data.claims || data.claims.length === 0) {
      const empty = document.createElement('div');
      empty.textContent = "No claims extracted.";
      resultsArea.appendChild(empty);
      return;
    }

    data.claims.forEach(claim => {
      const card = document.createElement('div');
      card.className = 'claim-card';

      const badge = document.createElement('span');
      badge.className = `badge badge-${claim.status}`;
      badge.textContent = claim.status;
      card.appendChild(badge);

      const claimText = document.createElement('div');
      claimText.className = 'claim-text';
      claimText.textContent = `"${claim.claim_text}"`;
      card.appendChild(claimText);

      if (claim.status === 'likely-checkable' && claim.top_evidence_text) {
        const evLabel = document.createElement('div');
        evLabel.className = 'label';
        evLabel.textContent = `Top hit (${claim.top_evidence_source}):`;
        card.appendChild(evLabel);
        
        const evText = document.createElement('div');
        evText.className = 'evidence-text';
        evText.textContent = claim.top_evidence_text;
        card.appendChild(evText);
      }

      resultsArea.appendChild(card);
    });
  }

  // 5. Render Full Inspection results (preserved F2 behavior)
  function renderFullResults(data) {
    // We strictly use document.createElement to avoid innerHTML injection vulnerabilities
    
    // Calculate counts dynamically from verdicts
    const counts = {};
    if (data.verdicts) {
      data.verdicts.forEach(v => {
        counts[v.label] = (counts[v.label] || 0) + 1;
      });
    }

    // Render counts badge summary
    const countsDiv = document.createElement('div');
    countsDiv.style.marginBottom = '12px';
    
    const countLabel = document.createElement('div');
    countLabel.className = 'label';
    countLabel.textContent = 'Status Summary:';
    countsDiv.appendChild(countLabel);

    for (const [status, count] of Object.entries(counts)) {
      if (count > 0 && !status.startsWith('total')) {
        const badge = document.createElement('span');
        badge.className = `badge badge-${status.replaceAll('_', '-')}`;
        badge.textContent = `${count} ${status.replaceAll('_', ' ')}`;
        badge.style.marginRight = '4px';
        countsDiv.appendChild(badge);
      }
    }
    resultsArea.appendChild(countsDiv);

    // Render claims list
    const claimsLabel = document.createElement('div');
    claimsLabel.className = 'label';
    claimsLabel.textContent = 'Claims in this response';
    resultsArea.appendChild(claimsLabel);

    if (!data.claims || data.claims.length === 0) {
      const empty = document.createElement('div');
      empty.textContent = "No claims extracted.";
      resultsArea.appendChild(empty);
      return;
    }

    // Create a map for verdicts
    const verdictMap = {};
    if (data.verdicts) {
      data.verdicts.forEach(v => {
        verdictMap[v.claim_id] = v;
      });
    }

    data.claims.forEach(claim => {
      const verdict = verdictMap[claim.id];
      if (!verdict) {
        const missing = document.createElement('div');
        missing.className = 'claim-card';
        missing.textContent = `${claim.text} — No verdict returned.`;
        resultsArea.appendChild(missing);
        return;
      }

      const card = document.createElement('div');
      card.className = 'claim-card';

      const badge = document.createElement('span');
      badge.className = `badge badge-${verdict.label}`;
      badge.textContent = verdict.label;
      card.appendChild(badge);

      const claimText = document.createElement('div');
      claimText.className = 'claim-text';
      claimText.textContent = `"${claim.text}"`;
      card.appendChild(claimText);

      const explanation = document.createElement('p');
      explanation.className = 'verdict-explanation';
      explanation.textContent = {
        Supported: 'The retrieved evidence supports this claim.',
        Contradicted: 'The retrieved evidence conflicts with this claim.',
        Insufficient: 'The available evidence is inconclusive. This claim has not been verified.',
        Abstain: 'The model could not verify this claim from the available evidence. This does not mean the claim is false.'
      }[verdict.label] || 'Review the evidence below.';
      card.appendChild(explanation);

      const evidenceDetails = document.createElement('details');
      evidenceDetails.className = 'evidence-details';
      const evidenceSummary = document.createElement('summary');
      evidenceSummary.textContent = 'View evidence and model scores';
      evidenceDetails.appendChild(evidenceSummary);

      // Aggregation info
      if (verdict.aggregation_trace) {
        const aggInfo = document.createElement('details');
        aggInfo.className = 'agg-info';
        const aggSummary = document.createElement('summary');
        aggSummary.textContent = 'Decision rule and research trace';
        aggInfo.appendChild(aggSummary);
        const rule = verdict.aggregation_trace.rule || "Aggregated";
        const conf = Math.round(verdict.confidence * 100);
        const ruleText = document.createElement('p');
        ruleText.textContent = `Rule: ${rule} (Uncalibrated confidence: ${conf}%)`;
        aggInfo.appendChild(ruleText);
        const trace = document.createElement('pre');
        trace.textContent = JSON.stringify(verdict.aggregation_trace, null, 2);
        aggInfo.appendChild(trace);
        evidenceDetails.appendChild(aggInfo);
      }

      // Evidence list
      const evidenceItems = (data.evidence_by_claim && data.evidence_by_claim[claim.id]) || [];
      const pairwise = verdict.per_evidence || [];
      if (pairwise.length > 0 || evidenceItems.length > 0) {
        const evList = document.createElement('div');
        evList.className = 'evidence-list';
        
        const evTextMap = {};
        evidenceItems.forEach(ev => {
          evTextMap[ev.id] = ev.text;
        });

        const decisiveIds = verdict.aggregation_trace?.decisive_evidence_ids || [];

        // Retain every evidence item, including items without pairwise scores.
        const pairwiseMap = new Map(pairwise.map(item => [item.evidence_id, item]));
        const renderedPairs = evidenceItems.map(item => pairwiseMap.get(item.id) || { evidence_id: item.id });
        const evidenceIds = new Set(evidenceItems.map(item => item.id));
        renderedPairs.push(...pairwise.filter(item => !evidenceIds.has(item.evidence_id)));
        renderedPairs.forEach(evVerdict => {
          const evItem = document.createElement('div');
          evItem.className = 'evidence-item';
          
          const isDecisive = decisiveIds.includes(evVerdict.evidence_id);
          if (isDecisive) {
            evItem.classList.add('evidence-decisive');
          }
          
          const evMeta = document.createElement('div');
          evMeta.className = 'evidence-meta';
          evMeta.textContent = evVerdict.evidence_id;
          
          if (isDecisive) {
            const decisiveBadge = document.createElement('span');
            decisiveBadge.className = 'badge badge-decisive';
            decisiveBadge.textContent = '★ DECISIVE';
            evMeta.appendChild(decisiveBadge);
          }
          
          evItem.appendChild(evMeta);

          const evText = document.createElement('div');
          evText.className = 'evidence-text';
          evText.textContent = evTextMap[evVerdict.evidence_id] || "[Evidence text missing]";
          evItem.appendChild(evText);

          const scores = document.createElement('div');
          scores.className = 'pairwise-scores';
          
          if (evVerdict.p_entail !== null && evVerdict.p_entail !== undefined) {
            const ent = document.createElement('span');
            ent.className = 'score';
            ent.textContent = `Entail: ${Math.round(evVerdict.p_entail * 100)}%`;
            scores.appendChild(ent);
          }
          if (evVerdict.p_neutral !== null && evVerdict.p_neutral !== undefined) {
            const neu = document.createElement('span');
            neu.className = 'score';
            neu.textContent = `Neutral: ${Math.round(evVerdict.p_neutral * 100)}%`;
            scores.appendChild(neu);
          }
          if (evVerdict.p_contra !== null && evVerdict.p_contra !== undefined) {
            const con = document.createElement('span');
            con.className = 'score';
            con.textContent = `Contra: ${Math.round(evVerdict.p_contra * 100)}%`;
            scores.appendChild(con);
          }
          if (evVerdict.similarity !== null && evVerdict.similarity !== undefined) {
            const sim = document.createElement('span');
            sim.className = 'score';
            sim.textContent = `Sim: ${evVerdict.similarity.toFixed(2)}`;
            scores.appendChild(sim);
          }
          
          evItem.appendChild(scores);
          evList.appendChild(evItem);
        });
        evidenceDetails.appendChild(evList);
      }

      card.appendChild(evidenceDetails);

      resultsArea.appendChild(card);
    });
  }
});
