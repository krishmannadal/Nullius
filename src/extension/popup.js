document.addEventListener('DOMContentLoaded', async () => {
  const previewBox = document.getElementById('preview-box');
  const quickBtn = document.getElementById('quick-btn');
  const fullBtn = document.getElementById('full-btn');
  const statusArea = document.getElementById('status-area');
  const resultsArea = document.getElementById('results-area');

  let capturedText = null;

  // Show status message
  function showStatus(msg, isError = false) {
    statusArea.style.display = 'block';
    statusArea.textContent = msg;
    statusArea.className = isError ? 'status-error' : 'status-info';
  }

  function hideStatus() {
    statusArea.style.display = 'none';
  }

  // 1. Inject content script and capture text on popup load
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    // Inject the content script if not already injected
    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      files: ['content.js']
    });

    // Send message to capture response
    chrome.tabs.sendMessage(tab.id, { type: "CAPTURE_RESPONSE" }, (response) => {
      if (chrome.runtime.lastError) {
        previewBox.textContent = "Error: Could not connect to page. Make sure you are on a supported site (e.g. chatgpt.com).";
        return;
      }

      if (!response || !response.success) {
        previewBox.textContent = response?.error || "Unknown error capturing response.";
        return;
      }

      // Success
      capturedText = response.text;
      const preview = capturedText.length > 200 ? capturedText.substring(0, 200) + '...' : capturedText;
      previewBox.textContent = preview;
      quickBtn.disabled = false;
      fullBtn.disabled = false;
    });
  } catch (err) {
    previewBox.textContent = `Error injecting script: ${err.message}`;
  }

  // 2. Handle Quick Check button click (F1 flow: POST /verify/quick, 5s timeout)
  quickBtn.addEventListener('click', async () => {
    if (!capturedText) return;

    quickBtn.disabled = true;
    fullBtn.disabled = true;
    quickBtn.textContent = 'Checking...';
    hideStatus();
    resultsArea.replaceChildren(); // safe clear without innerHTML

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000); // 5s timeout for quick check

    try {
      const response = await fetch('http://127.0.0.1:8000/verify/quick', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({
          text: capturedText,
          retrieve_k: 5,
          evidence_floor_score: 0.0
        }),
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        showStatus(`Backend error (${response.status}): ${errorText}`, true);
        return;
      }

      const data = await response.json();
      renderQuickResults(data);
      
    } catch (err) {
      if (err.name === 'AbortError') {
        showStatus("Request timed out. Backend took too long to respond.", true);
      } else {
        showStatus(`Backend offline or unreachable. Ensure Nullius is running on 127.0.0.1:8000. Detail: ${err.message}`, true);
      }
    } finally {
      clearTimeout(timeoutId);
      quickBtn.disabled = false;
      fullBtn.disabled = false;
      quickBtn.textContent = 'Quick Check';
    }
  });

  // 3. Handle Full Inspection button click (F2 flow: POST /verify/full, 60s timeout)
  fullBtn.addEventListener('click', async () => {
    if (!capturedText) return;

    quickBtn.disabled = true;
    fullBtn.disabled = true;
    fullBtn.textContent = 'Inspecting...';
    hideStatus();
    resultsArea.replaceChildren(); // safe clear without innerHTML

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 60000); // 60s timeout for full check

    try {
      const response = await fetch('http://127.0.0.1:8000/verify/full', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({
          text: capturedText,
          retrieve_k: 5,
          evidence_floor_score: 0.0
        }),
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        showStatus(`Backend error (${response.status}): ${errorText}`, true);
        return;
      }

      const data = await response.json();
      renderFullResults(data);
      
    } catch (err) {
      if (err.name === 'AbortError') {
        showStatus("Request timed out. Backend took too long to respond.", true);
      } else {
        showStatus(`Backend offline or unreachable. Ensure Nullius is running on 127.0.0.1:8000. Detail: ${err.message}`, true);
      }
    } finally {
      clearTimeout(timeoutId);
      quickBtn.disabled = false;
      fullBtn.disabled = false;
      fullBtn.textContent = 'Full Inspection';
    }
  });

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
        if (count > 0) {
          const badge = document.createElement('span');
          badge.className = `badge badge-${status}`;
          badge.textContent = `${count} ${status}`;
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
      if (count > 0) {
        const badge = document.createElement('span');
        badge.className = `badge badge-${status}`;
        badge.textContent = `${count} ${status}`;
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
      if (!verdict) return; // defensive

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
        aggInfo.textContent = `Rule: ${rule} (Conf: ${conf}%)`;
        card.appendChild(aggInfo);
      }

      // Evidence list
      if (verdict.per_evidence && verdict.per_evidence.length > 0) {
        const evList = document.createElement('div');
        evList.className = 'evidence-list';
        
        const evidenceItems = (data.evidence_by_claim && data.evidence_by_claim[claim.id]) || [];
        const evTextMap = {};
        evidenceItems.forEach(ev => {
          evTextMap[ev.id] = ev.text;
        });

        const decisiveIds = verdict.aggregation_trace?.decisive_evidence_ids || [];

        verdict.per_evidence.forEach(evVerdict => {
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
