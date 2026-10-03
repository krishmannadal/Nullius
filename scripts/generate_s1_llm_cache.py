import json
import argparse
from pathlib import Path
import sys
import hashlib
from datetime import datetime, UTC

def cache_key(response_text: str) -> str:
    return hashlib.blake2b(response_text.encode("utf-8"), digest_size=16).hexdigest()[:16]

def generate_llm_cache(s1_dir: Path, model_name: str):
    responses_file = s1_dir / "responses.jsonl"
    cache_file = s1_dir / "llm_claims_cache.json"
    metadata_file = s1_dir / "metadata.json"
    
    if not responses_file.exists():
        print(f"Error: {responses_file} not found.", file=sys.stderr)
        sys.exit(1)
        
    responses = []
    with open(responses_file, encoding="utf-8") as f:
        for line in f:
            responses.append(json.loads(line))
            
    cache_payload = {"entries": {}}
    
    print(f"Generating LLM cache using model: {model_name}...")
    
    for r in responses:
        text = r["text"]
        key = cache_key(text)
        
        # =====================================================================
        # TODO: RESEARCHER MANUAL IMPLEMENTATION REQUIRED HERE
        # Nullius does not ship with an LLM provider client by design.
        # You must implement the API call to your preferred LLM here.
        # Example:
        #
        # claims = call_openai_api(text, model=model_name)
        # OR
        # claims = call_anthropic_api(text, model=model_name)
        #
        # `claims` should be a list of strings (the decomposed atomic claims).
        # =====================================================================
        
        print(f"Skipping API call for response {r['response_id']} - IMPLEMENTATION REQUIRED")
        claims = [] # Replace this with actual API output
        
        cache_payload["entries"][key] = claims
        
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache_payload, f, indent=2, ensure_ascii=False)
        
    print(f"Wrote {cache_file}.")
    print("WARNING: Do not forget to update metadata.json with the new frozen date and model info once the cache is legitimate.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate legitimate LLM Claims Cache for S1")
    parser.add_argument("--model", type=str, required=True, help="Name of the LLM model used for generation")
    args = parser.parse_args()
    
    s1_dir = Path("data/eval/s1")
    generate_llm_cache(s1_dir, args.model)
