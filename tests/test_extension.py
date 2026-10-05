import unittest
from pathlib import Path

import pytest

# Require playwright to be installed.
try:
    from playwright.sync_api import sync_playwright

    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

EXTENSION = Path(__file__).resolve().parents[1] / "src" / "extension"


@pytest.mark.skipif(not HAS_PLAYWRIGHT, reason="playwright not installed")
class CaptureResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(headless=True, args=["--no-sandbox"])

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()

    def setUp(self):
        self.context = self.browser.new_context()
        self.page = self.context.new_page()

    def tearDown(self):
        self.context.close()

    def capture(self, html):
        self.page.set_content(html)
        self.page.evaluate(EXTENSION.joinpath("content.js").read_text())
        return self.page.evaluate("window.__nulliusCaptureResponse()")

    # A. One normal completed assistant response
    def test_normal_completed_response(self):
        html = """
        <div role="region" aria-label="Conversation">
            You said:
            <button><p>Hello</p></button>
            ChatGPT said:
            <p>Hi there! How can I help?</p>
        </div>
        """
        result = self.capture(html)
        self.assertTrue(result["success"])
        self.assertEqual(result["text"], "Hi there! How can I help?")

    # B. Multiple user/assistant turns → newest assistant turn must be selected.
    def test_multiple_turns_selects_newest(self):
        html = """
        <div role="region" aria-label="Conversation">
            You said:
            <button><p>First</p></button>
            ChatGPT said:
            <p>Old answer</p>
            You said:
            <button><p>Second</p></button>
            ChatGPT said:
            <p>New answer</p>
        </div>
        """
        result = self.capture(html)
        self.assertTrue(result["success"])
        self.assertEqual(result["text"], "New answer")

    # C. Assistant response followed by user turn → extraction must stop at the assistant boundary.
    def test_stops_at_assistant_boundary(self):
        html = """
        <div role="region" aria-label="Conversation">
            ChatGPT said:
            <p>Assistant answer</p>
            You said:
            <button><p>New prompt</p></button>
        </div>
        """
        result = self.capture(html)
        self.assertTrue(result["success"])
        self.assertEqual(result["text"], "Assistant answer")

    # D. No assistant response → safe failure.
    def test_no_assistant_response(self):
        html = """
        <div role="region" aria-label="Conversation">
            You said:
            <button><p>Hello</p></button>
        </div>
        """
        result = self.capture(html)
        self.assertFalse(result["success"])
        self.assertIn("No assistant response found", result["error"])

    # E. Assistant currently streaming → safe streaming failure.
    def test_assistant_streaming(self):
        html = """
        <div role="region" aria-label="Conversation">
            ChatGPT said:
            <p>Streaming...</p>
        </div>
        <button aria-label="Stop generating">Stop</button>
        """
        result = self.capture(html)
        self.assertFalse(result["success"])
        self.assertIn("Response is still generating", result["error"])

    # F. Multi-paragraph response → paragraph boundaries preserved.
    def test_multi_paragraph(self):
        html = """
        <div role="region" aria-label="Conversation">
            ChatGPT said:
            <p>Para 1</p>
            <p>Para 2</p>
        </div>
        """
        result = self.capture(html)
        self.assertTrue(result["success"])
        self.assertEqual(result["text"], "Para 1\n\nPara 2")

    # G. List response → list item boundaries preserved.
    def test_list_response(self):
        html = """
        <div role="region" aria-label="Conversation">
            ChatGPT said:
            <p>List:</p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
        </div>
        """
        result = self.capture(html)
        self.assertTrue(result["success"])
        self.assertEqual(result["text"], "List:\n\nItem 1\n\nItem 2")

    # H. Legacy data-message-author-role DOM → legacy fallback still works.
    def test_legacy_fallback(self):
        html = """
        <main>
            <div data-message-author-role="assistant">
                <div class="markdown">Legacy answer</div>
            </div>
        </main>
        """
        result = self.capture(html)
        self.assertTrue(result["success"])
        self.assertEqual(result["text"], "Legacy answer")

    # I. Repeated injection → no duplicate listener/declaration failure.
    def test_repeated_injection(self):
        html = """
        <main>
            <div data-message-author-role="assistant">
                <div class="markdown">Legacy answer</div>
            </div>
        </main>
        """
        self.page.set_content(html)
        self.page.evaluate(EXTENSION.joinpath("content.js").read_text())
        # Inject a second time, it should not throw SyntaxError
        self.page.evaluate(EXTENSION.joinpath("content.js").read_text())

        result = self.page.evaluate("window.__nulliusCaptureResponse()")
        self.assertTrue(result["success"])
        self.assertEqual(result["text"], "Legacy answer")


if __name__ == "__main__":
    unittest.main()


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.mark.parametrize(
    "body,expected",
    [
        ("<article><h3>ChatGPT said:</h3><p>Answer</p></article>", "Answer"),
        (
            '<section><h3>ChatGPT said:</h3><p>Answer</p><div><button aria-label="Good response">Good</button></div></section>',
            "Answer",
        ),
        ("ChatGPT said:<span></span>Direct answer", "Direct answer"),
        ("<div><div><h4>ChatGPT said:</h4></div><p>Answer</p></div>", "Answer"),
        (
            '<h3>ChatGPT said:</h3><p>Visible<span hidden>Hidden</span></p><p style="display:none">Hidden</p>',
            "Visible",
        ),
        (
            "<h3>ChatGPT said:</h3><pre><code>if x:\n    work()\n\n    done()</code></pre>",
            "if x:\n    work()\n\n    done()",
        ),
        (
            "<h3>ChatGPT said:</h3><table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>",
            "A\tB\t\n\n1\t2",
        ),
    ],
)
def test_capture_nested_and_formatted_content(browser, body, expected):
    page = browser.new_page()
    try:
        page.set_content('<div role="region" aria-label="Conversation">' + body + "</div>")
        page.evaluate(EXTENSION.joinpath("content.js").read_text())
        assert page.evaluate("window.__nulliusCaptureResponse()")["text"] == expected
    finally:
        page.close()


@pytest.mark.parametrize(
    "tail,error",
    [
        ("<h3>ChatGPT said:</h3>", "empty"),
        ('<h3>ChatGPT said:</h3><div class="result-streaming">New</div>', "still generating"),
    ],
)
def test_empty_latest_does_not_return_previous(browser, tail, error):
    page = browser.new_page()
    try:
        page.set_content(
            '<div role="region" aria-label="Conversation"><h3>ChatGPT said:</h3><p>Old</p><h3>You said:</h3><p>Next</p>'
            + tail
            + "</div>"
        )
        page.evaluate(EXTENSION.joinpath("content.js").read_text())
        result = page.evaluate("window.__nulliusCaptureResponse()")
        assert not result["success"] and error in result["error"]
    finally:
        page.close()


def test_actual_listener_injection_guard(browser):
    page = browser.new_page()
    try:
        page.evaluate(
            "window.listeners=[]; window.chrome={runtime:{onMessage:{addListener: callback=>window.listeners.push(callback)}}}"
        )
        for _ in range(3):
            page.evaluate(EXTENSION.joinpath("content.js").read_text())
        assert page.evaluate("window.listeners.length") == 1
    finally:
        page.close()


def test_extension_package_is_deterministic():
    from scripts.package_extension import package_bytes

    assert package_bytes() == package_bytes()


@pytest.mark.parametrize("mode", ["fresh", "unsupported", "timeout"])
def test_popup_refresh_and_capture_failures(browser, mode):
    page = browser.new_page()
    try:
        page.clock.install()
        page.set_content(
            '<div id="preview-box"></div><button id="quick-btn" disabled>Quick</button>'
            '<button id="full-btn" disabled>Full</button><div id="status-area"></div>'
            '<div id="results-area"></div>'
        )
        page.evaluate(
            """mode => {
          window.sent = []; window.captureCount = 0;
          window.chrome = {
            tabs: {
              query: async () => [{id:1, url:mode === 'unsupported' ? 'https://example.com/' : 'https://chatgpt.com/c/fixture'}],
              sendMessage: () => mode === 'timeout' ? new Promise(() => {}) : Promise.resolve({success:true,text:++window.captureCount === 1 ? 'Old answer' : 'Fresh answer'})
            },
            scripting: {executeScript: async () => []}
          };
          window.fetch = async (url, options) => {
            window.sent.push(JSON.parse(options.body));
            return {ok:true,json:async()=>({claims:[]})};
          };
        }""",
            mode,
        )
        page.evaluate(EXTENSION.joinpath("popup.js").read_text())
        page.evaluate("document.dispatchEvent(new Event('DOMContentLoaded'))")
        if mode == "fresh":
            page.wait_for_function("!document.getElementById('quick-btn').disabled")
            page.locator("#quick-btn").click()
            page.wait_for_function("window.sent.length === 1")
            assert page.evaluate("window.sent[0].text") == "Fresh answer"
        elif mode == "unsupported":
            page.wait_for_function(
                "document.getElementById('preview-box').textContent.includes('Open a ChatGPT')"
            )
            assert page.locator("#quick-btn").is_disabled()
        else:
            page.clock.run_for(4500)
            assert "timed out" in page.locator("#preview-box").inner_text()
            assert page.locator("#quick-btn").is_disabled()
        if mode != "fresh":
            assert page.evaluate("window.sent.length") == 0
    finally:
        page.close()
