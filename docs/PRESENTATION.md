# Nullius: presentation walkthrough

## Start and update

From the local repository, activate the existing Python environment and run:

```bash
git pull origin master
python -m scripts.run_extension_backend --warm-full
```

Stop an older Nullius backend before starting another on port 8000. Wait for the
server startup message. Model loading happens before the presentation when
`--warm-full` is used; the first download needs internet access.

In Chrome, open `chrome://extensions` and reload Nullius. Reload the ChatGPT tab.
Open the popup and confirm **v1.4.0** is displayed at the top. If you loaded an
extracted ZIP, replace that folder with the new ZIP contents and reload it;
updating a separate Git checkout does not update the folder Chrome loaded.

## Demonstrate the main workflow

1. Open a fresh ChatGPT conversation and send this controlled demo prompt:

   > Repeat these two sentences exactly, without correcting them or adding commentary:
   > Earth is the third planet from the Sun.
   > Earth is the closest planet to the Sun.

2. Wait for the answer to finish. Open Nullius. Leave **Complete chat** selected
   and click **Analyze response**. No copying or pasting is needed when capture succeeds.
3. Show the claim verdicts. Expand **View evidence and model scores** to explain
   which retrieved sentences support or conflict with each claim.
4. Close the popup, highlight one sentence in ChatGPT, reopen Nullius, choose
   **Selected text only**, and click **Analyze response**. This checks that passage.
5. Optional: expand **Connection, evidence search & research tools**, compare
   aggregation rules on the same stored scores, or export the analysis JSON.

The real local NLI backend returned **Supported** and **Contradicted**, respectively,
for these two sentences during the development smoke check. This is a controlled
demonstration, not an accuracy benchmark. The smoke check used a ChatGPT-shaped
browser fixture with the actual capture adapter, popup and real backend; it did
not validate a logged-in ChatGPT page or Chrome's native extension permissions.

## Explain the project in a few sentences

“Nullius is a Chrome extension that helps inspect factual claims in ChatGPT
answers. It extracts claims, retrieves candidate evidence from a local reference
corpus, and uses an NLI model to score support, contradiction and neutrality. It
shows the verdict and the evidence behind it. Users can inspect all loaded answers,
one full response, or a selected passage. Researchers can compare decision rules
without rerunning inference.”

The current reference corpus is small. **Abstain** means the model declined to
verify the claim; missing evidence does not establish falsehood. Complete chat
includes loaded assistant answers, excluding user prompts and unloaded history.

## Resolve a demo failure

- **No answer detected:** confirm a completed answer is visible on the active
  ChatGPT tab. Click **Read ChatGPT again**, then **Analyze response**. If capture
  still fails, use **Copy capture diagnostics** under research tools. It copies
  page structure counts without conversation text, making the layout failure diagnosable.
- **Connection error:** keep the backend terminal running on the same computer as
  Chrome. Use **Test connection** under research tools.
- **Loading or timeout:** prepare the model with `--warm-full`, and try Latest
  response or a selected sentence while testing a long chat.
- **Many Abstain results:** explain the reference corpus coverage honestly. Use
  the controlled astronomy example to demonstrate evidence inspection.
- **Popup closes:** reopen it and analyze again; keep it open during a request.
