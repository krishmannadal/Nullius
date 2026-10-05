"""Generate real S1 outputs through an explicitly configured chat-completions provider.

No provider/model is chosen implicitly. Requires frozen human gold before any call.
Credentials are read from NULLIUS_LLM_API_KEY and never persisted. The endpoint is
NULLIUS_LLM_BASE_URL (ending in /v1), with an explicit --model and --provider.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.parse import urlparse

import requests

from src.core.types import utcnow_iso
from src.eval.provenance import software_identity
from src.eval.s1_annotations import sha256
from src.eval.s1_workflow import ROOT, benchmark, load_gold, require, write_new

SYSTEM_PROMPT = """Extract atomic factual assertions from the supplied response, not their truth.
Preserve modality, attribution, negation, quantifiers, conditions, disjunction and
comparative direction. Do not endorse a quotation by deleting its attribution.
Resolve coreference only from the supplied text. Split independently asserted
conjunctions, never disjunctions into separately asserted facts. No external facts.
Return only a JSON object with a claims array, including an empty array when appropriate.
Each claim has claim_text, source_spans (ordered disjoint {start,end} regions using
zero-based Unicode character offsets, end exclusive, minimal and whitespace-trimmed),
operations (decomposition, rewrite, coreference as appropriate), decomposed (true only
for decomposition), and hedged. Use [] for source_spans if unable to map a span.
Preserve duplicates in your response. Do not include explanations outside JSON."""
PROCESSING_VERSION = "s1-provider-json-v2-no-dedup-no-span-inference"


def provider_complete(base_url, api_key, payload):
    # No retries: a failed request may already have consumed a generation. Retrying
    # requires a new run so successful raw responses are never silently replaced.
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = requests.post(
        base_url.rstrip("/") + "/chat/completions", headers=headers, json=payload, timeout=(15, 180)
    )
    response.raise_for_status()
    return response.json()


def parse_claims(raw, source):
    import json

    value = json.loads(raw["choices"][0]["message"]["content"])
    claims = value.get("claims")
    require(isinstance(claims, list), "provider output requires claims array")
    output = []
    for i, claim in enumerate(claims):
        require(
            isinstance(claim, dict)
            and isinstance(claim.get("claim_text"), str)
            and bool(claim["claim_text"].strip()),
            "invalid provider claim text",
        )
        spans = claim.get("source_spans")
        require(isinstance(spans, list), "provider claim requires source_spans (possibly empty)")
        previous = -1
        for span in spans:
            require(
                isinstance(span, dict) and set(span) == {"start", "end"}, "invalid provider span"
            )
            a, b = span["start"], span["end"]
            require(
                type(a) is int
                and type(b) is int
                and 0 <= a < b <= len(source)
                and a >= previous
                and source[a:b] == source[a:b].strip(),
                "invalid provider span bounds",
            )
            previous = b
        operations = claim.get("operations")
        require(
            isinstance(operations, list)
            and all(op in {"decomposition", "rewrite", "coreference"} for op in operations),
            "invalid operations",
        )
        require(
            type(claim.get("decomposed")) is bool and type(claim.get("hedged")) is bool,
            "provider claim requires boolean flags",
        )
        require(
            claim["decomposed"] == ("decomposition" in operations),
            "provider decomposition inconsistency",
        )
        exact = (
            len(spans) == 1 and source[spans[0]["start"] : spans[0]["end"]] == claim["claim_text"]
        )
        output.append(
            {
                "claim_text": claim["claim_text"],
                "source_spans": spans,
                "source_span": spans[0] if spans else None,
                "span_exactness": "exact" if exact else "approximate" if spans else "missing",
                "operations": operations,
                "decomposed": claim["decomposed"],
                "hedged": claim["hedged"],
                "raw_claim_index": i,
                "processed_claim_index": i,
            }
        )
    return output


def generate_llm_cache(
    s1_dir=ROOT,
    model=None,
    *,
    provider=None,
    base_url=None,
    output_dir=None,
    temperature=0.0,
    seed=None,
    model_revision=None,
    complete=None,
):
    s1_dir = Path(s1_dir)
    frozen, _ = load_gold(s1_dir)
    metadata, _, responses = benchmark(s1_dir)
    base_url = base_url or os.environ.get("NULLIUS_LLM_BASE_URL", "")
    api_key = os.environ.get("NULLIUS_LLM_API_KEY", "")
    require(
        bool(model) and bool(provider) and bool(base_url) and bool(api_key),
        "No cache was written. Supply --model, --provider, NULLIUS_LLM_BASE_URL and NULLIUS_LLM_API_KEY after freezing human gold.",
    )
    parsed = urlparse(base_url)
    require(
        not parsed.username and not parsed.password and not parsed.query,
        "Credentials must not be embedded in provider URL",
    )
    require(
        parsed.scheme == "https"
        or (parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1"}),
        "Use HTTPS, or an explicitly local HTTP provider",
    )
    require(0 <= temperature <= 2, "invalid temperature")
    out = Path(output_dir or s1_dir / "llm_run")
    out.mkdir(parents=True, exist_ok=False)
    settings = {
        "model": model,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    if seed is not None:
        settings["seed"] = seed
    metadata_out = {
        "schema_version": "s1-real-provider-run-v2",
        "provider": provider,
        "endpoint": base_url,
        "requested_model": model,
        "requested_revision": model_revision,
        "revision_note": "Requested revision is descriptive unless provider enforces it; raw response records resolved identity when supplied.",
        "settings": settings,
        "system_prompt": SYSTEM_PROMPT,
        "system_prompt_sha256": sha256(SYSTEM_PROMPT.encode()),
        "processing_version": PROCESSING_VERSION,
        "gold_manifest_sha256": sha256((s1_dir / "gold_manifest.json").read_bytes()),
        "responses_sha256": metadata["responses_hash_sha256"],
        "response_ids": frozen["response_ids"],
        "generated_at": utcnow_iso(),
        "software_identity": software_identity(),
        "responses": [],
    }
    call = complete or provider_complete
    for rid, source in responses.items():
        user_prompt = "Extract from this exact response:\n" + source
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        raw = call(base_url, api_key, {**settings, "messages": messages})
        raw_path = out / f"{rid}.raw.json"
        write_new(
            raw_path,
            {
                "response_id": rid,
                "messages": messages,
                "request": settings,
                "generated_at": utcnow_iso(),
                "raw_provider_output": raw,
            },
        )
        # Raw artifact survives parsing errors. No complete manifest is written then.
        processed = parse_claims(raw, source)
        metadata_out["responses"].append(
            {
                "response_id": rid,
                "raw_file": raw_path.name,
                "raw_sha256": sha256(raw_path.read_bytes()),
                "user_prompt_sha256": sha256(user_prompt.encode()),
                "resolved_model": raw.get("model"),
                "provider_fingerprint": raw.get("system_fingerprint"),
                "processed_claims": processed,
            }
        )
    write_new(out / "manifest.json", metadata_out)
    return metadata_out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1-dir", type=Path, default=ROOT)
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model-revision")
    args = parser.parse_args()
    try:
        generate_llm_cache(**vars(args))
    except (OSError, ValueError, KeyError, requests.RequestException) as exc:
        parser.exit(1, f"Blocked: {exc}\n")


if __name__ == "__main__":
    main()
