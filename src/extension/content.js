(() => {
  // Reinstall the adapter on each capture; no persistent listeners or page edits.
  globalThis.__nulliusCapture = (scope = 'chat') => {
    try {
      if (!['chat', 'latest', 'selection'].includes(scope)) {
        return { success: false, error: 'Choose Complete chat, Latest response or Selected text.' };
      }
      if (!['chatgpt.com', 'chat.openai.com'].includes(location.hostname)) {
        return { success: false, error: 'Open a conversation on chatgpt.com first.' };
      }

      function isVisible(node) {
        // display:contents wrappers have no client rect but can contain a visible answer.
        // visibility is inherited but descendants may explicitly override it.
        if (['hidden', 'collapse'].includes(getComputedStyle(node).visibility)) return false;
        for (let ancestor = node; ancestor; ancestor = ancestor.parentElement) {
          const style = getComputedStyle(ancestor);
          if (style.display === 'none' || style.contentVisibility === 'hidden') return false;
        }
        return true;
      }

      const assistantSelector = '[data-message-author-role="assistant"], [data-turn="assistant"], [data-testid="assistant-message"], .agent-turn, [data-message-role="assistant"]';
      const userSelector = '[data-message-author-role="user"], [data-turn="user"], [data-testid="user-message"], .user-message-bubble, [data-message-role="user"]';
      const bodySelector = '.markdown, .prose, [class*="markdown"], [data-message-content], .message-content';
      const candidates = new Set(document.querySelectorAll(assistantSelector));
      // ChatGPT layouts also identify turns by their article/message container.
      // Require an answer body, and reject user bubbles/headings; never read the whole page.
      for (const turn of document.querySelectorAll('[data-testid^="conversation-turn-"], main article, main [data-message-id]')) {
        if (turn.matches(userSelector) || turn.querySelector(userSelector)) continue;
        const enclosingTurn = turn.closest('[data-testid^="conversation-turn-"], article');
        const enclosingHeading = enclosingTurn?.querySelector('h5, h6');
        if (enclosingHeading && /^You said\s*:?\s*$/i.test(enclosingHeading.textContent.trim())) continue;
        const heading = turn.querySelector('h5, h6');
        if (heading && /^You said\s*:?\s*$/i.test(heading.textContent.trim())) continue;
        if ((heading && /^ChatGPT said\s*:?\s*$/i.test(heading.textContent.trim())) ||
            turn.querySelector(`${bodySelector}, [data-testid="assistant-message"]`)) candidates.add(turn);
      }
      // Associate response feedback controls with their nearest answer body. This
      // supports layouts without role attributes or conversation-turn articles.
      const feedbackSelectors = [
        'button[data-testid="good-response-turn-action-button"]',
        'button[data-testid="bad-response-turn-action-button"]',
        'button[aria-label="Good response"]', 'button[aria-label="Bad response"]',
        'button[aria-label="Read aloud"]'
      ];
      for (const control of document.querySelectorAll(feedbackSelectors.join(','))) {
        const selector = feedbackSelectors.find(item => control.matches(item));
        for (let node = control.parentElement; node; node = node.parentElement) {
          if (node.matches('main, [role="main"], #thread, body, html, nav, aside') ||
              node.closest(userSelector) || node.querySelectorAll(selector).length > 1) break;
          if (node.querySelector(bodySelector)) {
            if (!node.querySelector(`${userSelector}, textarea, [contenteditable="true"]`)) candidates.add(node);
            break;
          }
        }
      }
      const nodes = [...candidates].filter(node =>
        (isVisible(node) || [...node.querySelectorAll(bodySelector)].some(isVisible)) && !node.closest(userSelector));
      // Keep a whole assistant turn when role markers are nested; preserve DOM ordering.
      const turns = nodes.filter(node => !nodes.some(other => other !== node && other.contains(node)))
        .sort((a, b) => (a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING) ? -1 : 1);
      const lastNode = turns.at(-1);
      const stopButtons = document.querySelectorAll(
        'button[data-testid="stop-button"], button[aria-label="Stop generating"], button[aria-label="Stop streaming"]'
      );
      const activeStop = [...stopButtons].some(button => !button.disabled && isVisible(button));
      if (!lastNode) {
        // Explicit user selection remains usable when an unfamiliar layout has
        // no recognizable answer containers. Never substitute the whole page.
        if (scope === 'selection') {
          if (activeStop) return { success: false, error: 'Wait for ChatGPT to finish generating, then select the passage again.' };
          const selection = window.getSelection();
          if (!selection || selection.isCollapsed || !selection.toString().trim()) {
            return { success: false, error: 'Highlight the answer on ChatGPT first, then choose Selected text only.' };
          }
          const excluded = `${userSelector}, nav, aside, button, textarea, input, [contenteditable="true"]`;
          for (let index = 0; index < selection.rangeCount; index++) {
            const range = selection.getRangeAt(index);
            const start = range.startContainer.nodeType === Node.ELEMENT_NODE ? range.startContainer : range.startContainer.parentElement;
            const end = range.endContainer.nodeType === Node.ELEMENT_NODE ? range.endContainer : range.endContainer.parentElement;
            if (!isVisible(start) || !isVisible(end) || start.closest(excluded) || end.closest(excluded) ||
                [...document.querySelectorAll(excluded)].some(node => range.intersectsNode(node))) {
              return { success: false, error: 'Highlight only answer text, without navigation, prompts or input fields.' };
            }
          }
          return { success: true, text: selection.toString(), scope, message_count: null };
        }
        return { success: false,
          error: 'No ChatGPT answer is visible to Nullius yet. Keep the conversation tab active, wait for an answer, then click Analyze response again. Capture diagnostics are available below.',
          diagnostics: { hostname: location.hostname, ready_state: document.readyState,
            assistant_markers: document.querySelectorAll(assistantSelector).length,
            conversation_turns: document.querySelectorAll('[data-testid^="conversation-turn-"]').length,
            articles: document.querySelectorAll('main article').length,
            markdown_bodies: document.querySelectorAll('.markdown').length,
            message_containers: document.querySelectorAll('[data-message-id]').length,
            user_markers: document.querySelectorAll(userSelector).length,
            candidate_containers: candidates.size, visible_containers: nodes.length,
            prose_bodies: document.querySelectorAll(bodySelector).length,
            feedback_controls: document.querySelectorAll(feedbackSelectors.join(',')).length,
            frames: document.querySelectorAll('iframe').length } };
      }
      if (scope === 'selection') {
        const selection = window.getSelection();
        if (!selection || selection.isCollapsed || !selection.toString().trim()) {
          return { success: false, error: 'Highlight a passage in a completed ChatGPT answer before opening Nullius, then refresh the preview.' };
        }
        // A selection must stay inside one assistant answer, excluding prompts and page controls.
        const elementFor = node => node.nodeType === Node.ELEMENT_NODE ? node : node.parentElement;
        for (let index = 0; index < selection.rangeCount; index++) {
          const range = selection.getRangeAt(index);
          const answer = turns.find(turn => turn.contains(range.startContainer) && turn.contains(range.endContainer));
          if (!answer || (answer === lastNode && activeStop) ||
              answer.closest('.result-streaming') || answer.querySelector('.result-streaming')) {
            return { success: false, error: 'Select text within one completed ChatGPT answer, then refresh the preview.' };
          }
          const excluded = `${userSelector}, button, nav, [role="toolbar"], script, style`;
          const start = elementFor(range.startContainer);
          const end = elementFor(range.endContainer);
          const bodies = answer.matches(bodySelector) ? [answer] : [...answer.querySelectorAll(bodySelector)];
          if (bodies.length && (!bodies.some(body => body.contains(start)) || !bodies.some(body => body.contains(end)))) {
            return { success: false, error: 'Highlight text in the answer body, then refresh the preview.' };
          }
          if (!isVisible(start) || !isVisible(end) || start.closest(excluded) || end.closest(excluded) ||
              [...answer.querySelectorAll(excluded)].some(node => range.intersectsNode(node))) {
            return { success: false, error: 'Select answer text without prompts or page controls, then refresh the preview.' };
          }
        }
        return { success: true, text: selection.toString(), scope, message_count: 1 };
      }
      if (activeStop ||
          turns.some(node => node.closest('.result-streaming') || node.querySelector('.result-streaming'))) {
        return { success: false, error: 'ChatGPT is still generating. Wait, then click Refresh preview.' };
      }
      function answerText(node) {
        const allMarkdown = node.matches(bodySelector) ? [node] : [...node.querySelectorAll(bodySelector)];
        const markdownNodes = allMarkdown.filter(node =>
          isVisible(node) && !node.closest(userSelector) && !allMarkdown.some(other => other !== node && other.contains(node)));
        // Without markdown, clone the body to remove controls and headings without editing ChatGPT.
        let text;
        if (allMarkdown.length) {
          text = markdownNodes.map(node => node.innerText || node.textContent || '').join('\n\n').trim();
        } else {
          const clone = node.cloneNode(true);
          const originals = [...node.querySelectorAll('*')];
          [...clone.querySelectorAll('*')].forEach((node, index) => {
            if (!isVisible(originals[index]) || originals[index].closest(userSelector) ||
                node.matches('button, nav, [role="toolbar"], script, style') ||
                (node.matches('h5, h6') && /^ChatGPT said\s*:?\s*$/i.test(node.textContent.trim()))) node.remove();
          });
          clone.querySelectorAll('p, div, li, br, pre, h1, h2, h3, h4, h5, h6, blockquote, tr')
            .forEach(node => node.append('\n'));
          text = (clone.textContent || '').trim();
        }
        return text;
      }
      const selectedTurns = scope === 'latest' ? [lastNode] : turns;
      const texts = selectedTurns.map(answerText);
      if (texts.some(text => !text)) {
        return { success: false, error: 'A captured assistant answer is empty. Wait for completed answers, then refresh.' };
      }
      return { success: true, text: texts.join('\n\n'), scope, message_count: texts.length };
    } catch (error) {
      return { success: false, error: error.message };
    }
  };
})();
