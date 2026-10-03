document.addEventListener('DOMContentLoaded', async () => {
  const previewBox = document.getElementById('preview-box');
  const quickBtn = document.getElementById('quick-btn');
  const fullBtn = document.getElementById('full-btn');
  const refreshBtn = document.getElementById('refresh-btn');
  const connectionBtn = document.getElementById('connection-btn');
  const statusArea = document.getElementById('status-area');
  const resultsArea = document.getElementById('results-area');
  const endpoints = {
    quick: 'http://127.0.0.1:8000/verify/quick',
    full: 'http://127.0.0.1:8000/verify/full'
  };
  const timeouts = { quick: 5000, full: 60000 };
  let capturedText = null;
  let busy = false;

  function showStatus(msg, isError = false) {
    statusArea.style.display = 'block';
    statusArea.textContent = msg;
    statusArea.className = isError ? 'status-error' : 'status-info';
  }

  function setBusy(value) {
    busy = value;
    quickBtn.disabled = value || !capturedText;
    fullBtn.disabled = value || !capturedText;
    refreshBtn.disabled = value;
    connectionBtn.disabled = value;
  }

  async function captureLatest() {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab?.id || !tab.url || !['chatgpt.com', 'chat.openai.com'].includes(new URL(tab.url).hostname)) {
      throw new Error('Open a conversation on chatgpt.com, then click Refresh response.');
    }
    await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ['content.js'] });
    const response = await chrome.tabs.sendMessage(tab.id, { type: 'CAPTURE_RESPONSE' });
    if (!response?.success || typeof response.text !== 'string' || !response.text.trim()) {
      throw new Error(response?.error || 'Could not capture the latest ChatGPT answer. Refresh the page and try again.');
    }
    capturedText = response.text;
    previewBox.textContent = capturedText;
    return capturedText;
  }

  async function refresh() {
    if (busy) return;
    setBusy(true);
    resultsArea.replaceChildren();
    statusArea.style.display = 'none';
    try {
      await captureLatest();
    } catch (error) {
      capturedText = null;
      previewBox.textContent = error.message;
      showStatus(error.message, true);
    } finally {
      setBusy(false);
    }
  }

  async function inspect(kind) {
    if (busy) return;
    setBusy(true);
    resultsArea.replaceChildren();
    showStatus(kind === 'quick' ? 'Checking evidence availability…' : 'Inspecting claims and evidence…');
    const button = kind === 'quick' ? quickBtn : fullBtn;
    button.textContent = kind === 'quick' ? 'Checking…' : 'Inspecting…';
    let timeoutId;
    let stage = 'capture';
    try {
      // Capture again on each explicit check: never submit an old popup snapshot.
      const text = await captureLatest();
      stage = 'backend';
      const controller = new AbortController();
      timeoutId = setTimeout(() => controller.abort(), timeouts[kind]);
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
      if (kind === 'quick') renderQuickResults(data);
      else renderFullResults(data);
      showStatus(kind === 'quick' ?
        'Quick Check reports evidence availability. A missing hit does not mean a claim is false.' :
        'Inspection complete for the response shown above. Scores are uncalibrated research outputs.');
    } catch (error) {
      if (stage === 'capture') {
        capturedText = null;
        previewBox.textContent = error.message;
        showStatus(error.message, true);
      } else if (error.name === 'AbortError') {
        showStatus('Request timed out. First-time model loading can take longer; warm the backend and try again. Closing this request does not stop backend computation.', true);
      } else if (error instanceof TypeError) {
        showStatus('Cannot connect to Nullius. Start the local backend on this computer at 127.0.0.1:8000, then use Test connection.', true);
      } else {
        showStatus(error.message, true);
      }
    } finally {
      clearTimeout(timeoutId);
      button.textContent = kind === 'quick' ? 'Quick Check' : 'Full Inspection';
      setBusy(false);
    }
  }

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
      showStatus('Connected to Nullius verification API. Use Quick Check to test evidence availability; Full Inspection also needs downloaded NLI model files.');
    } catch (error) {
      showStatus(`Backend unavailable. Run python -m scripts.run_extension_backend on this computer. ${error.message}`, true);
    } finally {
      clearTimeout(timer);
      setBusy(false);
    }
  });
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
    claimsLabel.textContent = 'Inspection Details:';
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

      // Aggregation info
      if (verdict.aggregation_trace) {
        const aggInfo = document.createElement('div');
        aggInfo.className = 'agg-info';
        const rule = verdict.aggregation_trace.rule || "Aggregated";
        const conf = Math.round(verdict.confidence * 100);
        aggInfo.textContent = `Rule: ${rule} (Uncalibrated confidence: ${conf}%)`;
        const trace = document.createElement('pre');
        trace.textContent = JSON.stringify(verdict.aggregation_trace, null, 2);
        aggInfo.appendChild(trace);
        card.appendChild(aggInfo);
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
        card.appendChild(evList);
      }

      resultsArea.appendChild(card);
    });
  }
});
