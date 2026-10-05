function captureResponse() {
  // ChatGPT adapter logic
  const assistantNodes = document.querySelectorAll('div[data-message-author-role="assistant"]');
  if (!assistantNodes || assistantNodes.length === 0) {
    return { success: false, error: "No assistant response found on page. Make sure you are on a ChatGPT conversation." };
  }

  // Get the most recent response (last in the DOM)
  const lastNode = assistantNodes[assistantNodes.length - 1];
  
  // Conservative guard against incomplete/streaming response
  // Checks for ChatGPT's streaming class or active stop button
  const isStreaming = lastNode.classList.contains('result-streaming') || 
                      lastNode.querySelector('.result-streaming') !== null ||
                      document.querySelector('button[aria-label="Stop generating"]') !== null;
                      
  if (isStreaming) {
    return { success: false, error: "Response is still generating. Please wait until it is complete." };
  }
  
  // Try to find the markdown container to avoid capturing internal UI text, or fallback to the node itself
  const markdownNode = lastNode.querySelector('.markdown') || lastNode;
  
  const text = markdownNode.innerText || markdownNode.textContent;
  if (!text || text.trim().length === 0) {
    return { success: false, error: "Assistant response is empty." };
  }

  return { success: true, text: text.trim() };
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === "CAPTURE_RESPONSE") {
    try {
      const result = captureResponse();
      sendResponse(result);
    } catch (e) {
      sendResponse({ success: false, error: e.message });
    }
  }
  // Return true to indicate we will send a response asynchronously (though we send it synchronously here)
  // This prevents the message port from closing immediately if we needed async.
  return true; 
});
