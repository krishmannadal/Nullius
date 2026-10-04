# Nullius for ChatGPT

A Chrome Manifest V3 extension for inspecting ChatGPT answers on `chatgpt.com`
(also recognizes `chat.openai.com`). It shows results in its popup and defaults
to **Complete chat**, collecting all loaded assistant answers in the current conversation.
The extension captures text only after you open it. Only clicking **Quick Check**
or **Full Inspection** sends that text to the local backend. **Compare rules**
sends only the stored run ID and rule settings; exports stay on your computer. It does not edit the
ChatGPT page or run in the background.

## Run the backend on your own computer

From your local Nullius repository, update the code and activate its existing
Python 3.11 virtual environment. On Windows PowerShell:

```powershell
git pull origin master
.\.venv\Scripts\Activate.ps1
python -m scripts.run_extension_backend --warm-full
```

On macOS/Linux:

```bash
git pull origin master
source .venv/bin/activate
python -m scripts.run_extension_backend --warm-full
```

For a fresh checkout, create a Python 3.11 virtual environment and install the
existing `requirements.lock.txt` with `python -m pip install -r requirements.lock.txt`
first. The lock includes the spaCy English model. Keep TLS verification enabled.

Wait for the server to announce startup on `127.0.0.1:8000`. `--warm-full` downloads
and loads the real NLI model before accepting requests; public Hugging Face model
access is required. Stop any previous server on port 8000 first. Leave this
terminal running. CPU inference is supported; CUDA is optional.

To start Quick Check without transformer-model downloads, omit `--warm-full`:

```bash
python -m scripts.run_extension_backend
```

This warms spaCy and BM25. Full Inspection will still need the NLI model and can
time out on its first request; restart with `--warm-full` to prepare it beforehand.
A failed warmup prints an error and must be resolved before claiming full readiness.
The Chrome browser connects to the server on **the same computer**, so starting
only a cloud backend does not connect your local Chrome to it.

## Install in Chrome

1. Open `chrome://extensions`.
2. Enable **Developer mode**.
3. Choose **Load unpacked**, then select the repository's `src/extension` folder
   (the folder directly containing `manifest.json`).
4. Pin Nullius using Chrome's extensions menu.
5. Open a ChatGPT conversation and wait for the answer to finish.
6. Click Nullius, choose an analysis scope, check **Analysis preview**, and choose **Test connection**.
7. Click **Quick Check** or **Full Inspection**, keeping the popup open until done.

If you use the release ZIP, extract it first and load the extracted folder that
contains `manifest.json`. A ZIP alone cannot be loaded using **Load unpacked**.
This is a developer installation, not a Chrome Web Store release.

After pulling an update, click the extension's **Reload** button in
`chrome://extensions`, then reload your ChatGPT tab. **Refresh preview** updates
the snapshot in an already-open popup. Each check of a page-captured answer captures
the chosen scope again, so it does not silently send an older snapshot after the
conversation changes. The adapter is reinstalled on each capture and returns a
one-time script result, with no persistent message listeners.

## Choose what to analyze

- **Complete chat (default):** all loaded ChatGPT answers in the current conversation,
  in order, with your prompts and page controls excluded. This default is restored
  whenever the popup opens. Scroll up in ChatGPT to load older answers before
  capturing; the extension cannot inspect unloaded history, other chats or hidden
  answer branches. The preview shows how many answers were captured.
- **Latest response · full text:** the complete latest assistant answer.
- **Selected text only:** highlight a passage inside one completed ChatGPT answer
  before opening the extension, then choose this scope. Only the highlighted text
  is submitted. Missing selections, prompts, controls and selections spanning
  multiple answers are rejected, with no fallback to full-chat submission.

Changing scopes refreshes the preview and clears previous results and exports.
It sends no text to the backend until you click a check. Full-chat and latest
checks wait for generation to finish; a selection in an earlier completed answer
can be inspected while a later answer is generating. Complete chat allows up to
15 seconds for Quick Check and three minutes for Full Inspection; other scopes
allow five seconds and one minute respectively. Keep the popup open until done.
Long chats may still time out; use a smaller scope or prepare the NLI model first.

## If the answer is not detected

Open **Paste answer instead**, copy the completed answer from ChatGPT, paste it,
and click **Use entire pasted text**. Alternatively, highlight a passage in the
paste box and click **Use selected pasted part**. Check **Analysis preview**, then click **Quick
Check** or **Full Inspection**. Pasting and selecting text make no backend request.
Checks use that selected snapshot; editing the paste box does not change the
preview until you use one of those buttons again. The preview identifies pasted
text and exports record `capture_source: "pasted-answer"` rather than
`"chatgpt-page"`, plus `analysis_scope: "pasted-full"` or `"pasted-selection"`.
The scope control displays **Pasted text** while this snapshot is active. Choose
another scope to return to ChatGPT; **Refresh preview** returns to Complete chat.
The paste box and selected snapshot disappear when the popup closes.

## What the results mean

- **Quick Check** reports whether BM25 found candidate sentences in the configured
  corpus. It does not run the verifier and does not declare claims true or false.
- **Full Inspection** shows the backend's pairwise entailment, contradiction,
  neutral and similarity scores, verdicts, aggregation trace and evidence IDs.
- **DECISIVE** means the backend aggregation rule marked that evidence decisive.
- Scores and confidence values are uncalibrated research outputs.

`configs/extension.yaml` uses the checked-in FEVER-derived **debug corpus** and
BM25. That corpus has limited coverage and contains its gold pages by construction.
It cannot settle arbitrary ChatGPT answers about current events or other uncovered
topics. A missing hit or an Abstain verdict does not establish a hallucination.
No scientific accuracy is claimed.

## Compare rules without rerunning the model

After Full Inspection succeeds, click **Compare rules · no new inference**.
Nullius compares five backend rules on the exact pairwise scores already stored
for this inspection: max entailment, noisy OR, rank weighting, threshold with
abstention, and the majority baseline. The original verdict and all evidence
remain visible. A table shows each rule's verdict and uncalibrated confidence;
expand a rule to inspect its rationale and decisive evidence IDs.

The count of claims receiving different verdicts describes **decision sensitivity**,
not hallucination accuracy. Agreement does not prove truth. Comparisons use rule
defaults and do not tune thresholds. No extraction, retrieval or NLI pass runs
again. The browser sends no pairwise scores and rejects a comparison whose run
ID, config hash, Git commit or pairwise inputs differ from the original.

The backend keeps at most 50 runs for up to one hour. Restarting the backend,
expiry or eviction can make a run unavailable. Export the inspection while it is
visible, then run Full Inspection again to create a new comparison-capable run.
Quick Check has no verifier scores, so its rule-comparison button stays disabled.

## Export an inspection for research review

Click **Export inspection JSON** after either kind of check. The download keeps:

- The exact captured answer, whether it came from the page or was pasted, and the
  original, unmodified backend result.
- Original run ID, configuration hash, Git commit, timestamp, resolved configuration,
  evidence and pairwise scores when returned by Full Inspection.
- Validated rule comparisons and their requested settings, when available.
- Export timestamp, extension version and a research-use notice.
- Analysis scope (`chat`, `latest`, `selection`, `pasted-full`, `pasted-selection`)
  and the captured answer count for page captures. Pasted text has no inferred message count.

An export is an inspection snapshot, not human annotation, adjudicated gold or
evaluation metrics. It includes the captured answer, so review its content before
sharing. Nothing is uploaded by the export action. **Refresh preview**, a scope change or a new
check clears the previous inspection and comparison to prevent stale exports.
Closing the popup loses local state; download your inspection before closing.

## Troubleshooting and manual smoke test

- **No answer found:** open a conversation containing an assistant answer, reload
  ChatGPT after extension updates, and click Refresh preview. ChatGPT's DOM can
  change; the adapter recognizes explicit assistant message/turn markers and the
  English "ChatGPT said:" conversation heading, including `display:contents`
  wrappers. It does not guess that arbitrary markdown is an assistant answer.
  If automatic capture still fails, use **Paste answer instead**.
- **Still generating:** wait for ChatGPT to finish, then Refresh preview.
- **Cannot connect:** start the backend locally; check its terminal for dependency,
  model-download or port-in-use errors. Test connection checks API availability,
  not successful model inference.
- **Timeout:** warm the NLI model first, try a shorter answer, and inspect the server
  log. Aborting a popup request does not cancel computation already running.
- **Popup closed:** reopen it and resubmit; popup-local results are not persisted.

On your real Chrome profile, check a completed ChatGPT answer, start another answer
while the popup is open and confirm streaming is blocked, then refresh after it
finishes. Confirm Complete chat captures all loaded answers in order without prompts,
Latest response captures only the latest full answer, and Selected text captures
exactly the highlighted passage. With no selection, confirm that scope disables
checks instead of falling back to the entire chat. Try both checks; compare
the full evidence IDs and scores with the backend. Then compare rules and export
the JSON, checking that the run IDs and pairwise scores remain identical. Also
select a pasted answer and a highlighted part in the paste box, confirm their
previews and exported scopes, and click Refresh preview to return to Complete chat.
Stop the backend and confirm
Test connection displays an error. Never use these smoke checks as scientific
benchmark results.

## Developer verification

```bash
python -m pytest tests/test_extension_service.py tests/test_api.py::test_extension_popup_contract
node --check src/extension/content.js
node --check src/extension/popup.js
```

Optional browser regression tools can be installed in a separate development
Python environment without changing the application lock:

```bash
python -m pip install playwright==1.62.0
python -m playwright install chromium
python -m scripts.test_extension_browser
```

The browser suite uses ChatGPT DOM fixtures and mocked Chrome APIs to test capture,
popup interactions, recapture, streaming guards, timeouts, safe text rendering and
error recovery. It does not log in to ChatGPT or validate Chrome's actual activeTab
permission grant. The onboarding machine's Chromium administrator policy prevents
loading unpacked extensions; the final real-profile smoke test is manual.
