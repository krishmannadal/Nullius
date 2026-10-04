"""Browser regression tests with ChatGPT DOM fixtures and mocked Chrome APIs.

Run: python -m scripts.test_extension_browser
Requires Playwright and Chromium. No ChatGPT login or real model output is used.
Chrome API mocks exercise popup behavior; they do not validate actual activeTab grants.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from playwright.sync_api import sync_playwright

EXTENSION = Path(__file__).resolve().parents[1] / "src" / "extension"


class BrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(
            executable_path=shutil.which("chromium"), headless=True, args=["--no-sandbox"]
        )

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()

    def tearDown(self):
        self.context.close()


class ChatGPTCaptureTests(BrowserTests):
    def setUp(self):
        self.context = self.browser.new_context()
        self.page = self.context.new_page()
        self.page.route(
            "https://chatgpt.com/**",
            lambda route: route.fulfill(
                content_type="text/html",
                body='<main><div data-message-author-role="user">Question</div>'
                '<div data-message-author-role="assistant"><div class="markdown">Old answer.</div></div>'
                '<section data-message-author-role="assistant"><div class="markdown">Latest answer.</div>'
                "<button>Copy</button></section></main>",
            ),
        )
        self.page.goto("https://chatgpt.com/c/fixture")

    def capture(self, scope="latest"):
        self.page.evaluate((EXTENSION / "content.js").read_text())
        return self.page.evaluate("scope => window.__nulliusCapture(scope)", scope)

    def select_text(self, selector, start, end):
        self.page.locator(selector).evaluate("""(node, offsets) => {
          const range = document.createRange();
          range.setStart(node.firstChild, offsets[0]); range.setEnd(node.firstChild, offsets[1]);
          const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
        }""", [start, end])

    def test_default_scope_includes_all_answers_in_order_without_prompts(self):
        self.page.evaluate((EXTENSION / "content.js").read_text())
        result = self.page.evaluate("() => window.__nulliusCapture()")
        self.assertEqual(result, {"success": True, "text": "Old answer.\n\nLatest answer.", "scope": "chat", "message_count": 2})

    def test_conversation_articles_without_role_attributes_are_captured(self):
        self.page.set_content('''<main>
          <article data-testid="conversation-turn-1"><div class="user-message-bubble">User prompt.</div></article>
          <article data-testid="conversation-turn-2"><div class="markdown">First answer.</div><button>Copy</button></article>
          <article data-testid="conversation-turn-3"><div class="markdown">Second answer.</div></article></main>''')
        result = self.capture("chat")
        self.assertEqual(result["text"], "First answer.\n\nSecond answer.")
        self.assertEqual(result["message_count"], 2)

    def test_nested_message_id_in_user_heading_turn_is_excluded(self):
        self.page.set_content('''<main><article><h5>You said:</h5>
          <div data-message-id="user"><div class="markdown">User prompt.</div></div></article>
          <article><div data-message-id="assistant"><div class="markdown">Answer.</div></div></article></main>''')
        self.assertEqual(self.capture("chat")["text"], "Answer.")

    def test_agent_turn_layout_without_legacy_role_marker(self):
        self.page.set_content('<main><section class="agent-turn"><div class="markdown">Agent answer.</div></section></main>')
        self.assertEqual(self.capture("chat")["text"], "Agent answer.")

    def test_complete_chat_deduplicates_nested_roles_but_keeps_repeated_answers(self):
        self.page.set_content('''<article data-turn="assistant"><div data-message-author-role="assistant">
          <div class="markdown">Same answer.</div></div></article>
          <article data-turn="assistant"><div data-message-author-role="assistant">
          <div class="markdown">Same answer.</div></div></article>''')
        result = self.capture("chat")
        self.assertEqual(result["text"], "Same answer.\n\nSame answer.")
        self.assertEqual(result["message_count"], 2)

    def test_complete_chat_does_not_skip_empty_earlier_answer(self):
        self.page.locator('div[data-message-author-role="assistant"] .markdown').evaluate("node => node.textContent = ''")
        self.assertFalse(self.capture("chat")["success"])
        self.assertTrue(self.capture("latest")["success"])

    def test_selected_text_is_exact_substring_of_one_answer(self):
        self.select_text("section .markdown", 2, 9)
        result = self.capture("selection")
        self.assertEqual(result, {"success": True, "text": "test an", "scope": "selection", "message_count": 1})

    def test_missing_or_user_selection_does_not_fall_back_to_complete_chat(self):
        self.assertFalse(self.capture("selection")["success"])
        self.select_text('[data-message-author-role="user"]', 0, 8)
        self.assertFalse(self.capture("selection")["success"])

    def test_selection_spanning_answers_is_rejected(self):
        self.page.evaluate("""() => {
          const bodies = document.querySelectorAll('.markdown'); const range = document.createRange();
          range.setStart(bodies[0].firstChild, 0); range.setEnd(bodies[1].firstChild, 6);
          window.getSelection().removeAllRanges(); window.getSelection().addRange(range);
        }""")
        self.assertFalse(self.capture("selection")["success"])

    def test_selection_of_answer_controls_is_rejected(self):
        self.select_text("section button", 0, 4)
        self.assertFalse(self.capture("selection")["success"])

    def test_selection_of_role_heading_is_not_answer_body_text(self):
        self.page.locator("section").evaluate("node => node.insertAdjacentHTML('afterbegin', '<h6>ChatGPT said:</h6>')")
        self.select_text("section h6", 0, 7)
        self.assertFalse(self.capture("selection")["success"])

    def test_selection_in_streaming_answer_is_blocked_but_earlier_answer_is_allowed(self):
        self.page.evaluate("document.body.insertAdjacentHTML('beforeend', '<button data-testid=stop-button>Stop</button>')")
        self.select_text("section .markdown", 0, 6)
        self.assertFalse(self.capture("selection")["success"])
        self.select_text('div[data-message-author-role="assistant"] .markdown', 0, 3)
        self.assertEqual(self.capture("selection")["text"], "Old")
        self.assertFalse(self.capture("chat")["success"])

    def test_unknown_scope_is_rejected(self):
        self.assertFalse(self.capture("unknown")["success"])

    def test_latest_answer_excludes_toolbar_and_user_text(self):
        self.assertEqual(self.capture(), {"success": True, "text": "Latest answer.", "scope": "latest", "message_count": 1})

    def test_repeat_injection_returns_fresh_snapshot_without_listeners(self):
        self.page.evaluate("""() => {
          window.__nulliusCaptureV2Installed = true;
          window.chrome = { runtime: { onMessage: { addListener: () => {
            throw new Error('Persistent listeners must not be installed');
          } } } };
        }""")
        self.assertTrue(self.capture()["success"])
        self.page.locator("section .markdown").evaluate("node => node.textContent = 'Changed answer.'")
        self.assertEqual(self.capture()["text"], "Changed answer.")

    def test_display_contents_wrapper_is_detected(self):
        self.page.locator("section").evaluate("node => node.style.display = 'contents'")
        self.assertEqual(self.page.locator("section").evaluate("node => node.getClientRects().length"), 0)
        self.assertEqual(self.capture()["text"], "Latest answer.")

    def test_explicit_assistant_turn_and_nested_roles_preserve_all_blocks(self):
        self.page.set_content('''<article data-turn="assistant">
          <div data-message-author-role="assistant"><div class="markdown">First.</div></div>
          <div data-message-author-role="assistant"><div class="markdown">Second.</div></div>
          <button>Copy answer</button></article>
          <article data-turn="user"><div class="markdown">User question.</div></article>''')
        self.assertEqual(self.capture()["text"], "First.\n\nSecond.")

    def test_heading_layout_excludes_later_user_turn(self):
        self.page.set_content('''<article data-testid="conversation-turn-1">
          <h6 class="sr-only">ChatGPT said:</h6><div class="markdown">Heading answer.</div>
          </article><article data-testid="conversation-turn-2"><h5>You said:</h5>
          <div class="markdown"><h6>ChatGPT said:</h6>Quoted user text.</div></article>''')
        self.assertEqual(self.capture()["text"], "Heading answer.")

    def test_unmarked_markdown_is_not_guessed_to_be_an_answer(self):
        self.page.set_content('<main><div class="markdown">Unidentified text.</div></main>')
        result = self.capture()
        self.assertFalse(result["success"])
        self.assertIn("Analyze response again", result["error"])
        self.assertEqual(result["diagnostics"]["markdown_bodies"], 1)

    def test_hidden_answers_and_hidden_body_blocks_are_excluded(self):
        self.page.locator("section").evaluate('''node => {
          const hidden = document.createElement('div'); hidden.className = 'markdown';
          hidden.style.display = 'none'; hidden.textContent = 'Hidden text.'; node.append(hidden);
        }''')
        self.assertEqual(self.capture()["text"], "Latest answer.")
        self.page.locator("section").evaluate("node => node.style.display = 'none'")
        self.assertEqual(self.capture()["text"], "Old answer.")

    def test_plain_body_removes_controls_without_changing_page(self):
        self.page.set_content('''<article data-testid="assistant-message"><h6>Answer heading.</h6><p>Plain answer.</p>
          <button>Copy</button><div role="toolbar">Actions</div>
          <span style="display:none">Hidden text.</span></article>''')
        before = self.page.content()
        self.assertEqual(self.capture()["text"], "Answer heading.\nPlain answer.")
        self.assertEqual(self.page.content(), before)

    def test_hidden_latest_markdown_does_not_leak_or_fall_back(self):
        self.page.locator("section .markdown").evaluate("node => node.style.display = 'none'")
        self.assertFalse(self.capture()["success"])

    def test_stop_button_blocks_streaming_but_hidden_button_does_not(self):
        self.page.evaluate("""() => {
          const button = document.createElement('button');
          button.dataset.testid = 'stop-button'; document.body.appendChild(button);
        }""")
        self.assertFalse(self.capture()["success"])
        self.page.locator('[data-testid="stop-button"]').evaluate(
            "node => node.style.display = 'none'"
        )
        self.assertTrue(self.capture()["success"])

    def test_empty_latest_does_not_fall_back_to_old_answer(self):
        self.page.locator("section .markdown").evaluate("node => node.textContent = ' '")
        self.assertFalse(self.capture()["success"])

    def test_multiple_body_blocks_without_duplicate_nested_markdown(self):
        self.page.locator("section").evaluate("""node => {
          node.replaceChildren();
          for (const text of ['First block.', 'Second block.']) {
            const body=document.createElement('div'); body.className='markdown'; body.textContent=text;
            node.appendChild(body);
          }
        }""")
        self.assertEqual(self.capture()["text"], "First block.\n\nSecond block.")


class PopupTests(BrowserTests):
    def setUp(self):
        self.context = self.browser.new_context()
        self.page = self.context.new_page()
        self.requests = []
        self.backend_status = 200
        self.backend_data = None
        self.comparison_data = None
        self.comparison_status = 200
        self.page.add_init_script("""
          window.testCapture = {success:true,text:'Initial answer.'};
          window.injectionCount = 0;
          window.requestedScopes = [];
          window.chrome = {
            tabs: {
              query: async () => [{id:7,url:'https://chatgpt.com/c/fixture'}]
            },
            scripting: {executeScript: async options => {
              window.injectionCount++;
              if (options.files) return [{frameId:0}];
              window.requestedScopes.push(options.args[0]);
              return [{frameId:0,result:window.testCaptures?.[options.args[0]] || window.testCapture}];
            }},
            runtime: {getManifest: () => ({version:'1.4.0'})}
          };
        """)
        self.page.route(
            "http://127.0.0.1:8501/popup.html",
            lambda route: route.fulfill(
                content_type="text/html", body=(EXTENSION / "popup.html").read_text()
            ),
        )
        self.page.route(
            "http://127.0.0.1:8501/popup.js",
            lambda route: route.fulfill(
                content_type="application/javascript", body=(EXTENSION / "popup.js").read_text()
            ),
        )
        self.page.route("http://127.0.0.1:8000/**", self.backend)
        self.page.goto("http://127.0.0.1:8501/popup.html")
        self.page.wait_for_function("!document.getElementById('quick-btn').disabled")
        # Research actions are intentionally secondary; open their section for those tests.
        self.page.locator("#advanced-details").evaluate("node => node.open = true")
        self.page.locator("#preview-details").evaluate("node => node.open = true")

    def backend(self, route):
        cors = {
            "Access-Control-Allow-Origin": "http://127.0.0.1:8501",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Accept",
        }
        if route.request.method == "OPTIONS":
            route.fulfill(status=204, headers=cors)
            return
        if route.request.url.endswith("openapi.json"):
            data = {"paths": {"/verify/quick": {}, "/verify/full": {}}}
        else:
            payload = route.request.post_data_json
            self.requests.append((route.request.url, payload))
            data = self.backend_data or {
                "claims": [{"claim_text": payload["text"], "status": "no-evidence-found"}],
                "counts": {"no_evidence_found": 1, "likely_checkable": 0, "total_claims": 1},
            }
        status = self.backend_status
        if route.request.url.endswith("/reaggregate"):
            status = self.comparison_status
            if self.comparison_data is not None:
                data = self.comparison_data
            else:
                original = self.backend_data
                data = {
                    "run_id": original["run_id"],
                    "config_hash": original["config_hash"],
                    "git_sha": original["git_sha"],
                    "comparisons": {},
                }
                for claim in original["claims"]:
                    data["comparisons"][claim["id"]] = {
                        rule: dict(
                            original["verdicts"][0],
                            label="Supported" if index == 0 else "Abstain",
                            aggregator_name=rule,
                        )
                        for index, rule in enumerate(payload["target_aggregators"])
                    }
        route.fulfill(
            status=status,
            headers=cors,
            content_type="application/json",
            body=json.dumps(data),
        )

    def test_opening_popup_sends_no_backend_request(self):
        self.assertEqual(self.requests, [])
        self.assertEqual(self.page.locator("#preview-box").inner_text(), "Initial answer.")
        self.assertEqual(self.page.locator("#scope-select").input_value(), "chat")
        self.assertEqual(self.page.evaluate("window.requestedScopes"), ["chat"])

    def test_primary_analyze_button_is_visible_without_paste_or_research_tools(self):
        self.page.reload()
        self.page.wait_for_function("!document.getElementById('full-btn').disabled")
        self.assertEqual(self.page.locator("#full-btn").inner_text(), "Analyze response")
        self.assertLess(self.page.locator("#full-btn").bounding_box()["y"], 300)
        self.assertFalse(self.page.locator("#paste-details").evaluate("node => node.open"))
        self.assertFalse(self.page.locator("#advanced-details").evaluate("node => node.open"))
        self.assertEqual(self.page.locator("#version-label").inner_text(), "v1.4.0")
        self.assertEqual(self.requests, [])

    def test_primary_button_recovers_capture_after_answer_appears(self):
        self.page.evaluate("window.testCapture = {success:false,error:'No answer yet.'}")
        self.page.click("#refresh-btn")
        self.page.wait_for_function("document.getElementById('status-area').className === 'status-error'")
        self.assertFalse(self.page.locator("#full-btn").is_disabled())
        self.assertEqual(self.requests, [])
        self.page.evaluate("window.testCapture = {success:true,text:'Answer appeared.'}")
        self.complete_full_fixture()
        self.assertEqual(self.requests[0][1]["text"], "Answer appeared.")
        self.assertEqual(self.page.locator("#full-btn").inner_text(), "Analyze response")

    def exported_report(self):
        with self.page.expect_download() as download_info:
            self.page.click("#export-btn")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "inspection.json"
            download_info.value.save_as(target)
            return json.loads(target.read_text())

    def test_complete_chat_submission_and_export_keep_scope_and_count(self):
        self.page.evaluate("""window.testCaptures = {chat: {
          success:true, text:'First answer.\\n\\nSecond answer.', scope:'chat', message_count:2
        }}""")
        self.page.click("#refresh-btn")
        self.page.wait_for_function("document.getElementById('source-label').textContent.includes('2 ChatGPT answers')")
        self.assertEqual(self.requests, [])
        self.page.click("#quick-btn")
        self.page.wait_for_function("!document.getElementById('export-btn').disabled")
        self.assertEqual(self.requests[0][1]["text"], "First answer.\n\nSecond answer.")
        report = self.exported_report()
        self.assertEqual(report["analysis_scope"], "chat")
        self.assertEqual(report["captured_message_count"], 2)

    def test_switching_to_latest_clears_previous_inspection_without_submitting(self):
        self.complete_full_fixture()
        self.page.evaluate("window.testCaptures = {latest:{success:true,text:'Only latest answer.',message_count:1}}")
        self.page.select_option("#scope-select", "latest")
        self.page.wait_for_function("document.getElementById('preview-box').textContent === 'Only latest answer.'")
        self.assertEqual(self.page.locator("#results-area").inner_text(), "")
        self.assertTrue(self.page.locator("#export-btn").is_disabled())
        self.assertTrue(self.page.locator("#compare-btn").is_disabled())
        self.assertEqual(len(self.requests), 1)
        self.page.click("#full-btn")
        self.page.wait_for_function("!document.getElementById('export-btn').disabled")
        self.assertEqual(self.requests[1][1]["text"], "Only latest answer.")
        report = self.exported_report()
        self.assertEqual(report["analysis_scope"], "latest")
        self.assertEqual(report["captured_message_count"], 1)

    def test_selection_failure_never_submits_whole_chat_and_refresh_recovers(self):
        self.page.evaluate("window.testCaptures = {selection:{success:false,error:'Highlight a passage first.'}}")
        self.page.select_option("#scope-select", "selection")
        self.page.wait_for_function("document.getElementById('status-area').className === 'status-error'")
        self.assertEqual(self.requests, [])
        self.assertTrue(self.page.locator("#quick-btn").is_disabled())
        self.page.evaluate("window.testCaptures.selection = {success:true,text:'Selected passage.',message_count:1}")
        self.page.click("#refresh-btn")
        self.page.wait_for_function("!document.getElementById('quick-btn').disabled")
        self.page.click("#quick-btn")
        self.page.wait_for_function("!document.getElementById('export-btn').disabled")
        self.assertEqual(self.requests[0][1]["text"], "Selected passage.")
        self.assertEqual(self.page.evaluate("window.requestedScopes"), ["chat", "selection", "selection", "selection"])
        self.assertEqual(self.exported_report()["analysis_scope"], "selection")

    def test_selected_pasted_part_uses_exact_preview_without_page_capture(self):
        self.page.locator("#paste-details summary").click()
        self.page.fill("#paste-input", "Before. Only this passage. After.")
        self.page.locator("#paste-input").evaluate("node => node.setSelectionRange(8, 26)")
        self.page.click("#paste-selection-btn")
        self.assertEqual(self.page.locator("#preview-box").text_content(), "Only this passage.")
        self.assertEqual(self.page.locator("#scope-select").input_value(), "pasted")
        injections = self.page.evaluate("window.injectionCount")
        self.page.click("#quick-btn")
        self.page.wait_for_function("!document.getElementById('export-btn').disabled")
        self.assertEqual(self.page.evaluate("window.injectionCount"), injections)
        self.assertEqual(self.requests[0][1]["text"], "Only this passage.")
        report = self.exported_report()
        self.assertEqual(report["analysis_scope"], "pasted-selection")
        self.assertEqual(report["capture_source"], "pasted-answer")

    def test_missing_paste_selection_preserves_current_preview(self):
        self.page.locator("#paste-details summary").click()
        self.page.fill("#paste-input", "Unselected draft.")
        self.page.click("#paste-selection-btn")
        self.assertIn("Highlight a passage", self.page.locator("#status-area").inner_text())
        self.assertEqual(self.page.locator("#preview-box").text_content(), "Initial answer.")
        self.assertEqual(self.page.locator("#scope-select").input_value(), "chat")
        self.assertEqual(self.requests, [])

    def test_capture_failure_offers_paste_and_checks_only_explicitly_selected_text(self):
        self.page.evaluate("window.testCapture = {success:false,error:'No assistant answer detected.'}")
        self.page.click("#refresh-btn")
        self.page.wait_for_function("document.getElementById('status-area').className === 'status-error'")
        self.assertFalse(self.page.locator("#paste-details").evaluate("node => node.open"))
        self.assertTrue(self.page.locator("#preview-details").evaluate("node => node.open"))
        self.assertTrue(self.page.locator("#quick-btn").is_disabled())
        self.page.locator("#paste-details summary").click()
        answer = "  Pasted completed answer.\nSecond paragraph.  "
        self.page.fill("#paste-input", answer)
        self.page.click("#paste-btn")
        self.assertEqual(self.requests, [])
        self.assertEqual(self.page.locator("#preview-box").text_content(), answer)
        self.assertIn("Pasted answer", self.page.locator("#source-label").inner_text())
        injections = self.page.evaluate("window.injectionCount")
        # Editing the draft is not the same as selecting it for inspection.
        self.page.fill("#paste-input", "Unselected draft.")
        self.page.click("#quick-btn")
        self.page.wait_for_function("!document.getElementById('export-btn').disabled")
        self.assertEqual(self.requests[0][1]["text"], answer)
        self.assertEqual(self.page.evaluate("window.injectionCount"), injections)
        with self.page.expect_download() as download_info:
            self.page.click("#export-btn")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "pasted.json"
            download_info.value.save_as(target)
            report = json.loads(target.read_text())
        self.assertEqual(report["capture_source"], "pasted-answer")
        self.assertEqual(report["captured_text"], answer)
        self.assertEqual(report["analysis_scope"], "pasted-full")
        self.assertIsNone(report["captured_message_count"])

    def test_refresh_switches_pasted_answer_back_to_page_capture(self):
        self.page.locator("#paste-details summary").click()
        self.page.fill("#paste-input", "Pasted answer.")
        self.page.click("#paste-btn")
        self.page.click("#refresh-btn")
        self.page.wait_for_function("document.getElementById('source-label').textContent.startsWith('Complete chat')")
        self.assertEqual(self.page.locator("#preview-box").inner_text(), "Initial answer.")
        self.assertEqual(self.requests, [])
        self.page.evaluate("window.testCapture.text = 'Updated page answer.'")
        self.page.click("#quick-btn")
        self.page.wait_for_function("!document.getElementById('export-btn').disabled")
        self.assertEqual(self.requests[0][1]["text"], "Updated page answer.")

    def test_empty_paste_preserves_previous_preview_and_inspection(self):
        self.page.click("#quick-btn")
        self.page.wait_for_function("!document.getElementById('export-btn').disabled")
        self.page.locator("#paste-details summary").click()
        self.page.fill("#paste-input", " \n ")
        self.page.click("#paste-btn")
        self.assertIn("Paste a completed answer first", self.page.locator("#status-area").inner_text())
        self.assertEqual(self.page.locator("#preview-box").inner_text(), "Initial answer.")
        self.assertFalse(self.page.locator("#export-btn").is_disabled())
        self.assertEqual(len(self.requests), 1)

    def test_new_pasted_answer_clears_previous_full_results_and_comparison(self):
        self.complete_full_fixture()
        self.page.click("#compare-btn")
        self.page.wait_for_function("document.getElementById('comparison-area').querySelector('table') !== null")
        self.page.locator("#paste-details summary").click()
        self.page.fill("#paste-input", "Different pasted answer.")
        self.page.click("#paste-btn")
        self.assertEqual(self.page.locator("#results-area").inner_text(), "")
        self.assertEqual(self.page.locator("#comparison-area").inner_text(), "")
        self.assertTrue(self.page.locator("#compare-btn").is_disabled())
        self.assertTrue(self.page.locator("#export-btn").is_disabled())
        self.assertEqual(len(self.requests), 2)

    def test_failed_refresh_does_not_reuse_pasted_answer(self):
        self.page.locator("#paste-details summary").click()
        self.page.fill("#paste-input", "Pasted answer.")
        self.page.click("#paste-btn")
        self.page.evaluate("window.testCapture = {success:false,error:'No answer.'}")
        self.page.click("#refresh-btn")
        self.page.wait_for_function("document.getElementById('status-area').className === 'status-error'")
        self.assertTrue(self.page.locator("#quick-btn").is_disabled())
        self.assertFalse(self.page.locator("#full-btn").is_disabled())
        self.assertEqual(self.requests, [])

    def test_quick_recaptures_fresh_response_and_skips_total_badge(self):
        self.page.evaluate("window.testCapture.text = 'New answer.'")
        self.page.click("#quick-btn")
        self.page.wait_for_function(
            "document.getElementById('status-area').textContent.includes('Evidence search complete')"
        )
        self.assertEqual(len(self.requests), 1)
        endpoint, body = self.requests[0]
        self.assertTrue(endpoint.endswith("/verify/quick"))
        self.assertEqual(body["text"], "New answer.")
        self.assertEqual(body["config_path"], "configs/extension.yaml")
        self.assertEqual(body["evidence_floor_score"], 0.01)
        self.assertNotIn("total_claims", self.page.locator("#results-area").inner_text())
        self.assertEqual(self.page.locator("#preview-box").inner_text(), "New answer.")

    def test_streaming_recapture_blocks_request_then_refresh_recovers(self):
        self.page.evaluate("window.testCapture = {success:false,error:'Still generating.'}")
        self.page.click("#quick-btn")
        self.page.wait_for_function(
            "document.getElementById('status-area').className === 'status-error'"
        )
        self.assertEqual(self.requests, [])
        self.assertTrue(self.page.locator("#quick-btn").is_disabled())
        self.page.evaluate("window.testCapture = {success:true,text:'Finished.'}")
        self.page.click("#refresh-btn")
        self.page.wait_for_function("!document.getElementById('quick-btn').disabled")
        self.assertEqual(self.page.locator("#preview-box").inner_text(), "Finished.")

    def test_full_keeps_scores_trace_all_evidence_and_safe_text(self):
        self.backend_data = {
            "claims": [{"id": "c1", "text": "Fixture claim."}],
            "evidence_by_claim": {
                "c1": [
                    {"id": "e1", "text": '<img src=x onerror="alert(1)"> Literal evidence.'},
                    {"id": "e2", "text": "Evidence without scores."},
                ]
            },
            "verdicts": [
                {
                    "claim_id": "c1",
                    "label": "Abstain",
                    "confidence": 0,
                    "aggregation_trace": {
                        "rule": "fixture-rule",
                        "explanation": "Fixture rationale.",
                        "decisive_evidence_ids": ["e1"],
                    },
                    "per_evidence": [
                        {
                            "evidence_id": "e1",
                            "p_entail": 0,
                            "p_contra": 0,
                            "p_neutral": 1,
                            "similarity": None,
                        }
                    ],
                }
            ],
        }
        self.page.click("#full-btn")
        self.page.wait_for_function(
            "document.getElementById('status-area').textContent.includes('Analysis complete')"
        )
        self.assertIn("could not verify", self.page.locator("#results-area").inner_text())
        self.assertFalse(self.page.locator(".evidence-details").evaluate("node => node.open"))
        self.page.locator(".evidence-details > summary").click()
        self.page.locator(".agg-info > summary").click()
        text = self.page.locator("#results-area").inner_text()
        self.assertIn("Neutral: 100%", text)
        self.assertIn("Entail: 0%", text)
        self.assertIn("Fixture rationale.", text)
        self.assertIn("Evidence without scores.", text)
        self.assertEqual(self.page.locator(".evidence-decisive").count(), 1)
        self.assertEqual(self.page.locator("#results-area img").count(), 0)
        self.assertTrue(self.requests[0][0].endswith("/verify/full"))
        self.assertNotIn("evidence_floor_score", self.requests[0][1])

    def complete_full_fixture(self):
        self.backend_data = {
            "run_id": "fixture-run",
            "config_hash": "fixture-config",
            "git_sha": "fixture-commit",
            "response_text": "Initial answer.",
            "timestamp": "2026-01-01T00:00:00Z",
            "schema_version": "1.0.0",
            "mode": "retrieved",
            "timings": {},
            "resolved_config": {"config": {"harness": {"kind": "debug"}}},
            "claims": [{"id": "c1", "text": "Fixture claim."}],
            "evidence_by_claim": {"c1": [{"id": "e1", "text": "Fixture evidence."}]},
            "verdicts": [
                {
                    "claim_id": "c1",
                    "label": "Abstain",
                    "confidence": 0,
                    "aggregator_name": "threshold_abstain",
                    "aggregation_trace": {
                        "rule": "fixture-rule",
                        "explanation": "Fixture rationale.",
                        "decisive_evidence_ids": ["e1"],
                    },
                    "per_evidence": [
                        {
                            "evidence_id": "e1",
                            "p_entail": 0,
                            "p_contra": 0,
                            "p_neutral": 1,
                            "similarity": None,
                        }
                    ],
                }
            ],
        }
        self.page.click("#full-btn")
        self.page.wait_for_function("!document.getElementById('compare-btn').disabled")

    def test_compare_sends_only_run_and_rule_configuration(self):
        self.complete_full_fixture()
        original_text = self.page.locator("#results-area").inner_text()
        injections = self.page.evaluate("window.injectionCount")
        self.page.evaluate("window.testCapture.text = 'A later answer.'")
        self.page.click("#compare-btn")
        self.page.wait_for_function(
            "document.getElementById('comparison-area').querySelector('table') !== null"
        )
        endpoint, payload = self.requests[-1]
        self.assertTrue(endpoint.endswith("/reaggregate"))
        self.assertEqual(set(payload), {"run_id", "target_aggregators", "aggregator_configs"})
        self.assertEqual(payload["run_id"], "fixture-run")
        self.assertEqual(len(payload["target_aggregators"]), 5)
        self.assertEqual(self.page.evaluate("window.injectionCount"), injections)
        self.assertEqual(self.page.locator("#results-area").inner_text(), original_text)
        self.assertIn(
            "1 of 1 claims receive different verdicts",
            self.page.locator("#comparison-area").inner_text(),
        )
        self.assertEqual(self.page.locator("#comparison-area tbody tr").count(), 5)
        self.assertIn(
            "Fixture rationale.", self.page.locator("#comparison-area pre").first.text_content()
        )

    def test_expired_run_keeps_inspection_exportable(self):
        self.complete_full_fixture()
        original = self.page.locator("#results-area").inner_text()
        self.comparison_status = 404
        self.comparison_data = {"detail": "Expired fixture run"}
        self.page.click("#compare-btn")
        self.page.wait_for_function(
            "document.getElementById('status-area').textContent.includes('expired')"
        )
        self.assertEqual(self.page.locator("#results-area").inner_text(), original)
        self.assertTrue(self.page.locator("#compare-btn").is_disabled())
        self.assertFalse(self.page.locator("#export-btn").is_disabled())

    def test_mismatched_comparison_is_rejected(self):
        self.complete_full_fixture()
        self.comparison_data = {
            "run_id": "another-run",
            "config_hash": "fixture-config",
            "comparisons": {},
        }
        self.page.click("#compare-btn")
        self.page.wait_for_function(
            "document.getElementById('status-area').className === 'status-error'"
        )
        self.assertEqual(self.page.locator("#comparison-area").inner_text(), "")
        self.assertIn("does not match", self.page.locator("#status-area").inner_text())
        self.assertFalse(self.page.locator("#compare-btn").is_disabled())

    def test_altered_pairwise_scores_are_rejected(self):
        self.complete_full_fixture()
        original = self.backend_data
        rules = [
            "max_entailment",
            "noisy_or",
            "weighted_by_retrieval",
            "threshold_abstain",
            "majority",
        ]
        comparisons = {rule: dict(original["verdicts"][0], aggregator_name=rule) for rule in rules}
        comparisons["majority"] = dict(
            comparisons["majority"],
            per_evidence=[
                {
                    "evidence_id": "e1",
                    "p_entail": 1,
                    "p_contra": 0,
                    "p_neutral": 0,
                    "similarity": None,
                }
            ],
        )
        self.comparison_data = {
            "run_id": "fixture-run",
            "config_hash": "fixture-config",
            "git_sha": "fixture-commit",
            "comparisons": {"c1": comparisons},
        }
        self.page.click("#compare-btn")
        self.page.wait_for_function(
            "document.getElementById('status-area').className === 'status-error'"
        )
        self.assertIn("scores differ", self.page.locator("#status-area").inner_text())
        self.assertEqual(self.page.locator("#comparison-area").inner_text(), "")
        self.assertFalse(self.page.locator("#export-btn").is_disabled())

    def test_export_keeps_original_text_provenance_and_comparison(self):
        self.complete_full_fixture()
        self.page.click("#compare-btn")
        self.page.wait_for_function(
            "document.getElementById('comparison-area').querySelector('table') !== null"
        )
        with self.page.expect_download() as download_info:
            self.page.click("#export-btn")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "inspection.json"
            download_info.value.save_as(target)
            report = json.loads(target.read_text())
        self.assertEqual(report["schema_version"], "nullius-inspection-export-v1")
        self.assertEqual(report["extension_version"], "1.4.0")
        self.assertEqual(report["capture_source"], "chatgpt-page")
        self.assertEqual(report["captured_text"], "Initial answer.")
        self.assertEqual(report["original_result"], self.backend_data)
        self.assertEqual(report["rule_comparison"]["run_id"], report["original_result"]["run_id"])
        self.assertEqual(
            report["rule_comparison"]["config_hash"], report["original_result"]["config_hash"]
        )
        self.assertEqual(len(self.requests), 2)  # Export makes no extra request.

    def test_refresh_discards_comparison_and_stale_export(self):
        self.complete_full_fixture()
        self.page.click("#compare-btn")
        self.page.wait_for_function(
            "document.getElementById('comparison-area').querySelector('table') !== null"
        )
        self.page.click("#refresh-btn")
        self.page.wait_for_function("!document.getElementById('quick-btn').disabled")
        self.assertEqual(self.page.locator("#comparison-area").inner_text(), "")
        self.assertTrue(self.page.locator("#compare-btn").is_disabled())
        self.assertTrue(self.page.locator("#export-btn").is_disabled())

    def test_quick_result_export_preserves_tier_one_semantics(self):
        self.page.click("#quick-btn")
        self.page.wait_for_function("!document.getElementById('export-btn').disabled")
        self.assertTrue(self.page.locator("#compare-btn").is_disabled())
        with self.page.expect_download() as download_info:
            self.page.click("#export-btn")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "quick.json"
            download_info.value.save_as(target)
            report = json.loads(target.read_text())
        self.assertEqual(report["inspection_kind"], "quick")
        self.assertIsNone(report["rule_comparison"])
        self.assertIn("claims", report["original_result"])
        self.assertEqual(len(self.requests), 1)

    def test_backend_error_clears_results_and_recovers_buttons(self):
        self.backend_status = 503
        self.backend_data = {"detail": "Models unavailable."}
        self.page.click("#quick-btn")
        self.page.wait_for_function(
            "document.getElementById('status-area').className === 'status-error'"
        )
        self.assertIn("Models unavailable.", self.page.locator("#status-area").inner_text())
        self.assertEqual(self.page.locator("#results-area").inner_text(), "")
        self.assertFalse(self.page.locator("#quick-btn").is_disabled())

    def test_invalid_response_is_not_reported_as_success(self):
        self.backend_data = {"claims": "invalid"}
        self.page.click("#quick-btn")
        self.page.wait_for_function(
            "document.getElementById('status-area').className === 'status-error'"
        )
        self.assertIn("invalid inspection response", self.page.locator("#status-area").inner_text())

    def test_unsupported_site_is_rejected_before_injection(self):
        self.page.evaluate(
            "() => { chrome.tabs.query = async () => [{id:7,url:'https://example.com/'}]; }"
        )
        injections = self.page.evaluate("window.injectionCount")
        self.page.click("#refresh-btn")
        self.page.wait_for_function(
            "document.getElementById('status-area').className === 'status-error'"
        )
        self.assertEqual(self.page.evaluate("window.injectionCount"), injections)
        self.assertEqual(self.requests, [])
        self.assertTrue(self.page.locator("#quick-btn").is_disabled())

    def test_connection_probe_is_explicit(self):
        self.page.click("#connection-btn")
        self.page.wait_for_function(
            "document.getElementById('status-area').textContent.includes('Connected to Nullius')"
        )
        self.assertEqual(self.requests, [])

    def test_timeout_releases_buttons(self):
        self.page.select_option("#scope-select", "latest")
        self.page.wait_for_function("!document.getElementById('quick-btn').disabled")
        self.page.clock.install()
        self.page.evaluate("""() => { window.fetch = (url, options) => {
          window.testWaiting = true;
          return new Promise((resolve, reject) => options.signal.addEventListener('abort',
            () => reject(new DOMException('Aborted', 'AbortError'))));
        }; }""")
        self.page.click("#quick-btn")
        self.page.wait_for_function("window.testWaiting === true")
        self.page.clock.fast_forward(5001)
        self.assertIn("timed out", self.page.locator("#status-area").inner_text())
        self.assertFalse(self.page.locator("#quick-btn").is_disabled())

    def test_complete_chat_has_longer_deadline_and_locks_scope_until_timeout(self):
        self.page.clock.install()
        self.page.evaluate("""() => { window.fetch = (url, options) => {
          window.testWaiting = true;
          return new Promise((resolve, reject) => options.signal.addEventListener('abort',
            () => reject(new DOMException('Aborted', 'AbortError'))));
        }; }""")
        self.page.click("#quick-btn")
        self.page.wait_for_function("window.testWaiting === true")
        self.assertTrue(self.page.locator("#scope-select").is_disabled())
        self.assertTrue(self.page.locator("#paste-selection-btn").is_disabled())
        self.page.clock.fast_forward(5001)
        self.assertTrue(self.page.locator("#quick-btn").is_disabled())
        self.page.clock.fast_forward(10000)
        self.assertIn("Complete chat check timed out", self.page.locator("#status-area").inner_text())
        self.assertFalse(self.page.locator("#scope-select").is_disabled())
        self.assertFalse(self.page.locator("#quick-btn").is_disabled())


if __name__ == "__main__":
    unittest.main(verbosity=2)
