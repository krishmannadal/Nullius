(() => {
  // Return one snapshot through executeScript. No persistent listeners or page edits.
  try {
    if (!['chatgpt.com', 'chat.openai.com'].includes(location.hostname)) {
      return { success: false, error: 'Open a conversation on chatgpt.com first.' };
    }

    function isVisible(node) {
      // display:contents wrappers have no client rect but can contain a visible answer.
      for (let ancestor = node; ancestor; ancestor = ancestor.parentElement) {
        const style = getComputedStyle(ancestor);
        if (style.display === 'none' || ['hidden', 'collapse'].includes(style.visibility) ||
            style.contentVisibility === 'hidden') return false;
      }
      return true;
    }

    const assistantSelector = '[data-message-author-role="assistant"], [data-turn="assistant"], [data-testid="assistant-message"]';
    const userSelector = '[data-message-author-role="user"], [data-turn="user"]';
    const candidates = new Set(document.querySelectorAll(assistantSelector));
    // Some conversation layouts put the role on the turn heading instead of the body.
    for (const turn of document.querySelectorAll('[data-testid^="conversation-turn-"]')) {
      if (turn.matches(userSelector) || turn.querySelector(userSelector)) continue;
      const heading = turn.querySelector('h5, h6');
      if (heading && /^ChatGPT said\s*:?\s*$/i.test(heading.textContent.trim())) candidates.add(turn);
    }
    const nodes = [...candidates].filter(node => isVisible(node) && !node.closest(userSelector));
    // Keep a whole assistant turn when role markers are nested; preserve DOM ordering.
    const turns = nodes.filter(node => !nodes.some(other => other !== node && other.contains(node)))
      .sort((a, b) => (a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING) ? -1 : 1);
    const lastNode = turns.at(-1);
    if (!lastNode) {
      return { success: false, error: 'No assistant answer detected. Open a completed ChatGPT conversation and refresh, or use Paste answer instead below.' };
    }
    const stopButtons = document.querySelectorAll(
      'button[data-testid="stop-button"], button[aria-label="Stop generating"], button[aria-label="Stop streaming"]'
    );
    if ([...stopButtons].some(button => !button.disabled && isVisible(button)) ||
        lastNode.closest('.result-streaming') || lastNode.querySelector('.result-streaming')) {
      return { success: false, error: 'ChatGPT is still generating. Wait, then click Refresh response.' };
    }
    const allMarkdown = lastNode.matches('.markdown') ? [lastNode] : [...lastNode.querySelectorAll('.markdown')];
    const markdownNodes = allMarkdown.filter(node =>
      isVisible(node) && !node.closest(userSelector) && !node.parentElement.closest('.markdown'));
    // Without markdown, clone the body to remove controls and headings without editing ChatGPT.
    let text;
    if (allMarkdown.length) {
      text = markdownNodes.map(node => node.innerText || node.textContent || '').join('\n\n').trim();
    } else {
      const clone = lastNode.cloneNode(true);
      const originals = [...lastNode.querySelectorAll('*')];
      [...clone.querySelectorAll('*')].forEach((node, index) => {
        if (!isVisible(originals[index]) || originals[index].closest(userSelector) ||
            node.matches('button, nav, [role="toolbar"], script, style') ||
            (node.matches('h5, h6') && /^ChatGPT said\s*:?\s*$/i.test(node.textContent.trim()))) node.remove();
      });
      clone.querySelectorAll('p, div, li, br, pre, h1, h2, h3, h4, h5, h6, blockquote, tr')
        .forEach(node => node.append('\n'));
      text = (clone.textContent || '').trim();
    }
    if (!text) return { success: false, error: 'The latest assistant answer is empty. Wait for a completed answer, then refresh.' };
    return { success: true, text };
  } catch (error) {
    return { success: false, error: error.message };
  }
})();
