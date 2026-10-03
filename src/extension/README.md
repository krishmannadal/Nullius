# Nullius for ChatGPT

A Chrome Manifest V3 extension for inspecting the latest completed answer on
`chatgpt.com` (also recognizes `chat.openai.com`). It shows results in its popup.
The extension captures text only after you open it. Only clicking **Quick Check**
or **Full Inspection** sends that text to the local backend. It does not edit the
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
6. Click Nullius, check **Response Preview**, and choose **Test connection**.
7. Click **Quick Check** or **Full Inspection**, keeping the popup open until done.

If you use the release ZIP, extract it first and load the extracted folder that
contains `manifest.json`. A ZIP alone cannot be loaded using **Load unpacked**.
This is a developer installation, not a Chrome Web Store release.

After pulling an update, click the extension's **Reload** button in
`chrome://extensions`, then reload your ChatGPT tab. **Refresh response** updates
the snapshot in an already-open popup. Every check captures the latest answer
again, so it does not silently send an older answer after the conversation changes.

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

## Troubleshooting and manual smoke test

- **No answer found:** open a conversation containing an assistant answer, reload
  ChatGPT after extension updates, and click Refresh response. ChatGPT's DOM can
  change; the adapter looks for `data-message-author-role="assistant"` and answer
  `.markdown` containers.
- **Still generating:** wait for ChatGPT to finish, then Refresh response.
- **Cannot connect:** start the backend locally; check its terminal for dependency,
  model-download or port-in-use errors. Test connection checks API availability,
  not successful model inference.
- **Timeout:** warm the NLI model first, try a shorter answer, and inspect the server
  log. Aborting a popup request does not cancel computation already running.
- **Popup closed:** reopen it and resubmit; popup-local results are not persisted.

On your real Chrome profile, check a completed ChatGPT answer, start another answer
while the popup is open and confirm streaming is blocked, then refresh after it
finishes. Confirm the new text appears in the preview. Try both checks; compare
the full evidence IDs and scores with the backend. Stop the backend and confirm
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
