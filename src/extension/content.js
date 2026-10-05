(() => {
  const captureResponse = () => {
    const visible = (element) => {
      for (let el = element; el; el = el.parentElement) {
        const style = getComputedStyle(el);
        if (el.hidden || el.getAttribute('aria-hidden') === 'true' ||
            style.display === 'none' || ['hidden', 'collapse'].includes(style.visibility) ||
            style.contentVisibility === 'hidden') return false;
      }
      return true;
    };
    const streamingError = () => ({success: false, error: 'Response is still generating. Please wait until it is complete.'});
    if ([...document.querySelectorAll('button[data-testid="stop-button"], button[aria-label="Stop generating"], button[aria-label="Stop streaming"]')]
        .some(el => !el.disabled && visible(el))) return streamingError();

    const region = document.querySelector('[role="region"][aria-label*="onversation"]');
    const legacy = [...document.querySelectorAll('[data-message-author-role="assistant"]')];
    const root = region || legacy[legacy.length - 1];
    if (!root) return {success: false, error: 'No assistant response found on page. Make sure you are on a ChatGPT conversation.'};
    if ([root, ...root.querySelectorAll('.result-streaming, [data-is-streaming="true"]')]
        .some(el => visible(el) && (el.classList.contains('result-streaming') || el.dataset.isStreaming === 'true'))) return streamingError();

    const turns = [];
    let role = region ? null : 'assistant';
    let chunks = [];
    if (!region) turns.push(chunks);
    const boundary = (next) => {
      role = next;
      chunks = [];
      if (role === 'assistant') turns.push(chunks); // retain empty newest turn
    };
    const marker = text => /^You said\s*:?\s*$/i.test(text.trim()) ? 'user' :
      /^ChatGPT said\s*:?\s*$/i.test(text.trim()) ? 'assistant' : null;
    const emit = value => { if (role === 'assistant') chunks.push(value); };
    const newline = () => {
      if (role === 'assistant' && chunks.length && !chunks[chunks.length - 1].endsWith('\n\n')) emit('\n\n');
    };
    const walk = (node, pre = false) => {
      if (node.nodeType === Node.TEXT_NODE) {
        const next = !pre && marker(node.textContent);
        if (next) { boundary(next); return; }
        let value = node.textContent;
        if (!pre) {
          value = value.replace(/\s+/g, ' ');
          if (!chunks.length || chunks[chunks.length - 1].endsWith('\n')) value = value.trimStart();
          if (!value.trim()) return;
        }
        emit(value);
        return;
      }
      if (node.nodeType !== Node.ELEMENT_NODE || !visible(node)) return;
      if (node.matches('button, nav, script, style, [role="toolbar"], svg, textarea')) return;
      const next = node.matches('h1,h2,h3,h4,h5,h6,[class*="sr-only"],[class*="visually-hidden"]') && marker(node.textContent);
      if (next) { boundary(next); return; }
      if (node.dataset.messageAuthorRole) boundary(node.dataset.messageAuthorRole);
      if (node.tagName === 'BR') { emit('\n'); return; }
      const block = /^(P|DIV|LI|PRE|H[1-6]|BLOCKQUOTE|TR|SECTION|ARTICLE)$/.test(node.tagName);
      if (block && !pre) newline();
      for (const child of node.childNodes) walk(child, pre || node.tagName === 'PRE');
      if (/^(TD|TH)$/.test(node.tagName)) emit('\t');
      if (block && !pre) newline();
    };
    // A legacy root itself carries the role. Traverse it so visibility applies.
    walk(root);
    if (!turns.length) return {success: false, error: 'No assistant response found on page.'};
    const text = turns[turns.length - 1].join('').replace(/^\n+|\n+$/g, '').replace(/[ \t]+$/g, '');
    if (!text.trim()) return {success: false, error: 'Assistant response is empty.'};
    return {success: true, text, capture_format: 'nullius-visible-text-v2'};
  };
  window.__nulliusCaptureResponse = captureResponse;
  if (!window.__nulliusListenerAdded && typeof chrome !== 'undefined' && chrome.runtime?.onMessage) {
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
      if (request.type !== 'CAPTURE_RESPONSE') return false;
      try { sendResponse(window.__nulliusCaptureResponse()); }
      catch (error) { sendResponse({success: false, error: error.message}); }
      return false;
    });
    window.__nulliusListenerAdded = true;
  }
})();
