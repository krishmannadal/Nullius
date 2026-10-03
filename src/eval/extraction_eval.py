import json
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
import csv
import sys
import argparse
from typing import List, Dict, Any

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def compute_token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_text(pred).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(gold_tokens)
    return (2 * prec * rec) / (prec + rec)

def compute_span_iou(span_sys, span_gold):
    if not span_sys or not span_gold:
        return 0.0
    sys_start, sys_end = span_sys.start, span_sys.end
    g_start, g_end = span_gold["start"], span_gold["end"]
    
    inter_start = max(sys_start, g_start)
    inter_end = min(sys_end, g_end)
    inter = max(0, inter_end - inter_start)
    
    union = (sys_end - sys_start) + (g_end - g_start) - inter
    return inter / union if union > 0 else 0.0

def load_data(s1_dir: Path):
    responses_file = s1_dir / "responses.jsonl"
    gold_file = s1_dir / "gold_annotations.jsonl"
    
    if not responses_file.exists():
        print(f"Error: {responses_file} not found.", file=sys.stderr)
        sys.exit(1)
        
    responses = []
    with open(responses_file, encoding="utf-8") as f:
        for line in f:
            responses.append(json.loads(line))
            
    gold_by_resp = defaultdict(list)
    if gold_file.exists():
        with open(gold_file, encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                gold_by_resp[c["response_id"]].append(c)
    else:
        print(f"Warning: {gold_file} not found. Ensure human annotation is completed.", file=sys.stderr)
        
    return responses, gold_by_resp

def generate_adjudication_csv(responses, gold_by_resp, s1_dir: Path):
    """
    Simulates the first part of the evaluation by generating a CSV for human adjudication.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.components.extractors import SpacySentenceExtractor, LLMClaimExtractor
    
    spacy_ext = SpacySentenceExtractor()
    cache_path = s1_dir / "llm_claims_cache.json"
    if not cache_path.exists():
        print(f"Warning: {cache_path} not found. LLM Extractor will fail if on_miss='error'.", file=sys.stderr)
        llm_ext = None
    else:
        llm_ext = LLMClaimExtractor(cache_path=str(cache_path), on_miss="error")
    
    extractors = [("SpacySentenceExtractor", spacy_ext)]
    if llm_ext:
        extractors.append(("LLMClaimExtractor", llm_ext))
        
    csv_path = s1_dir / "adjudication_candidates.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["extractor", "response_id", "system_claim", "candidate_gold_claim", "token_f1", "human_match_decision", "error_taxonomy_code", "adjudicator_notes"])
        
        for ext_name, ext in extractors:
            for r in responses:
                r_id = r["response_id"]
                text = r["text"]
                golds = gold_by_resp[r_id]
                
                if not golds:
                    continue
                    
                sys_claims = ext.extract(text)
                
                cost_matrix = np.zeros((len(sys_claims), len(golds)))
                for i, sc in enumerate(sys_claims):
                    for j, gc in enumerate(golds):
                        f1 = compute_token_f1(sc.text, gc["claim_text"])
                        if f1 >= 0.8:
                            cost_matrix[i, j] = f1
                            
                row_ind, col_ind = linear_sum_assignment(-cost_matrix)
                
                matched_sys = set()
                matched_gold = set()
                
                # Output candidates
                for i, j in zip(row_ind, col_ind):
                    if cost_matrix[i, j] >= 0.8:
                        matched_sys.add(i)
                        matched_gold.add(j)
                        writer.writerow([ext_name, r_id, sys_claims[i].text, golds[j]["claim_text"], cost_matrix[i,j], "", "", ""])
                
                # Output unmatched system claims (Spurious, etc)
                for i, sc in enumerate(sys_claims):
                    if i not in matched_sys:
                        writer.writerow([ext_name, r_id, sc.text, "[NONE]", 0.0, "0", "", "Unmatched System Claim"])
                        
                # Output unmatched gold claims (Missed)
                for j, gc in enumerate(golds):
                    if j not in matched_gold:
                        writer.writerow([ext_name, r_id, "[NONE]", gc["claim_text"], 0.0, "0", "M", "Missed Gold Claim"])
                        
    print(f"Generated {csv_path} for human adjudication.")
    print("Please fill out the 'human_match_decision' (1 or 0) and 'error_taxonomy_code' (S, U, O, R, P, B, C, D, N) columns.")

def bootstrap_ci(data: List[float], n_resamples: int = 10000):
    if not data:
        return 0.0, 0.0, 0.0
    data = np.array(data)
    means = np.zeros(n_resamples)
    for i in range(n_resamples):
        indices = np.random.randint(0, len(data), len(data))
        means[i] = np.mean(data[indices])
    return np.mean(data), np.percentile(means, 2.5), np.percentile(means, 97.5)

def run_evaluation(s1_dir: Path):
    """
    This function would load the adjudicated CSV and compute the final bootstrap CIs and taxonomy metrics.
    Currently a stub until the CSV is filled out by a human.
    """
    csv_path = s1_dir / "adjudication_candidates.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run --generate-adjudication first and fill it out.", file=sys.stderr)
        return
        
    print("Evaluation logic goes here (requires filled adjudication CSV).")
    print("1. Load adjudicated CSV.")
    print("2. Compute macro-averaged P/R/F1 from human_match_decision.")
    print("3. Compute decomposition split ratios for compound spans.")
    print("4. Compute bootstrap CIs over response IDs.")
    print("5. Aggregate Error Taxonomy frequencies.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S1 Extraction Evaluation")
    parser.add_argument("--generate-adjudication", action="store_true", help="Generate the CSV for human adjudication")
    parser.add_argument("--evaluate", action="store_true", help="Run the evaluation using the adjudicated CSV")
    args = parser.parse_args()
    
    s1_dir = Path("data/eval/s1")
    
    if args.generate_adjudication:
        resp, gold = load_data(s1_dir)
        generate_adjudication_csv(resp, gold, s1_dir)
    elif args.evaluate:
        run_evaluation(s1_dir)
    else:
        parser.print_help()
