(() => {
  // Popup reopening must not install duplicate message listeners.
  if (globalThis.__nulliusCaptureV2Installed) return;
  globalThis.__nulliusCaptureV2Installed = true;

  function captureResponse() {
    if (!['chatgpt.com', 'chat.openai.com'].includes(location.hostname)) {
      return { success: false, error: 'Open a conversation on chatgpt.com first.' };
    }
    const assistantNodes = [...document.querySelectorAll('[data-message-author-role="assistant"]')]
      .filter(node => node.getClientRects().length > 0);
    const lastNode = assistantNodes.at(-1);
    if (!lastNode) {
      return { success: false, error: 'No assistant response found. Open a ChatGPT conversation with a completed answer.' };
    }
    const stopButtons = document.querySelectorAll(
      'button[data-testid="stop-button"], button[aria-label="Stop generating"], button[aria-label="Stop streaming"]'
    );
    const activeStop = [...stopButtons].some(button => !button.disabled && button.getClientRects().length > 0);
    if (activeStop || lastNode.closest('.result-streaming') || lastNode.querySelector('.result-streaming')) {
      return { success: false, error: 'ChatGPT is still generating. Wait, then click Refresh response.' };
    }
    // Prefer answer bodies over action buttons. Keep multiple answer blocks in order.
    const markdownNodes = lastNode.matches('.markdown') ? [lastNode] :
      [...lastNode.querySelectorAll('.markdown')].filter(node => !node.parentElement.closest('.markdown'));
    const bodies = markdownNodes.length ? markdownNodes : [lastNode];
    const text = bodies.map(node => node.innerText || node.textContent || '').join('\n\n').trim();
    if (!text) return { success: false, error: 'The latest assistant response is empty.' };
    return { success: true, text };
  }

  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.type !== 'CAPTURE_RESPONSE') return false;
    try {
      sendResponse(captureResponse());
    } catch (error) {
      sendResponse({ success: false, error: error.message });
    }
    return false;
  });
})();
