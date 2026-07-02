#!/usr/bin/env python3
"""Quick AB vs ABC comparison."""
import json, os, sys, time, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip()
            if k and v and k not in os.environ:
                os.environ[k] = v

from llm_pipeline import LLMPipeline
from term_corrector import TermCorrector
from hotword_loader import HotwordLoader

gemini_key = os.environ["GEMINI_API_KEY"]
base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
model = "gemini-2.5-flash-lite"

pipeline_ab = LLMPipeline(api_key=gemini_key, base_url=base_url, model=model, timeout=30, stages="AB")
pipeline_abc = LLMPipeline(api_key=gemini_key, base_url=base_url, model=model, timeout=30, stages="ABC")

tc = TermCorrector("config/term_corrections.json")
hl = HotwordLoader("config/hotwords.json")
known_terms = sorted(set(list(tc.list_corrections().values()) + list(hl.get_custom_words())))

def glossary(segs):
    combined = " ".join(s.get("text", "") for s in segs)
    kw = set(re.findall(r"[\u30A0-\u30FF]{3,}", combined))
    gl = [t for t in known_terms if t in combined]
    gl += [k for k in sorted(kw) if k not in set(gl)]
    return gl[:50]

sessions = [
    "2026-03-27/23-47-21",
    "2026-03-28/05-46-37",
    "2026-03-27/22-42-13",
    "2026-03-27/22-18-21",
    "2026-03-27/23-32-25",
]

print(f"{'Session':>25} | {'AB chg':>6} {'calls':>5} {'time':>6} | {'ABC chg':>7} {'calls':>5} {'time':>6} {'retry':>6}")
print("-" * 85)

tot = dict(ab_ch=0, ab_calls=0, ab_t=0, abc_ch=0, abc_calls=0, abc_t=0, retries=0)

for sess_id in sessions:
    path = os.path.join("recordings", sess_id, "transcript.json")
    with open(path) as f:
        segs = json.load(f).get("segments", [])[:20]
    gl = glossary(segs)

    t0 = time.time()
    r_ab = pipeline_ab.process_segments(segs, known_terms=known_terms, session_glossary=gl, min_text_length=20)
    t_ab = time.time() - t0

    t0 = time.time()
    r_abc = pipeline_abc.process_segments(segs, known_terms=known_terms, session_glossary=gl, min_text_length=20)
    t_abc = time.time() - t0

    ab_ch = sum(1 for o, c in zip(segs, r_ab.corrected_segments) if o.get("text", "") != c.get("text", ""))
    abc_ch = sum(1 for o, c in zip(segs, r_abc.corrected_segments) if o.get("text", "") != c.get("text", ""))

    retry_str = "YES" if r_abc.retried else "no"
    print(f"{sess_id:>25} | {ab_ch:>6} {r_ab.api_calls:>5} {t_ab:>5.1f}s | {abc_ch:>7} {r_abc.api_calls:>5} {t_abc:>5.1f}s {retry_str:>6}")

    tot["ab_ch"] += ab_ch
    tot["ab_calls"] += r_ab.api_calls
    tot["ab_t"] += t_ab
    tot["abc_ch"] += abc_ch
    tot["abc_calls"] += r_abc.api_calls
    tot["abc_t"] += t_abc
    tot["retries"] += (1 if r_abc.retried else 0)

    # Show where AB and ABC differ
    for i, (o, ab, abc) in enumerate(zip(segs, r_ab.corrected_segments, r_abc.corrected_segments)):
        ot = o.get("text", "")
        abt = ab.get("text", "")
        abct = abc.get("text", "")
        if abt != abct:
            print(f"  ** seg {i} AB vs ABC differ:")
            print(f"     orig: {ot[:80]}")
            print(f"     AB:   {abt[:80]}")
            print(f"     ABC:  {abct[:80]}")

print("-" * 85)
n = len(sessions)
print(f"{'TOTAL':>25} | {tot['ab_ch']:>6} {tot['ab_calls']:>5} {tot['ab_t']:>5.1f}s | {tot['abc_ch']:>7} {tot['abc_calls']:>5} {tot['abc_t']:>5.1f}s {tot['retries']:>6}")
print()

cost_per_call = (2500 * 0.10 + 500 * 0.40) / 1_000_000
ab_avg = tot["ab_calls"] / n
abc_avg = tot["abc_calls"] / n
ab_mo = 480 * 30 * ab_avg * cost_per_call
abc_mo = 480 * 30 * abc_avg * cost_per_call
print(f"Avg calls/batch:  AB={ab_avg:.1f}  ABC={abc_avg:.1f}")
print(f"Monthly (8h/day): AB=${ab_mo:.2f}  ABC=${abc_mo:.2f}  diff=+${abc_mo - ab_mo:.2f}")
print(f"AB vs Legacy:     AB=${ab_mo:.2f}  Legacy=$6.48  diff=+${ab_mo - 6.48:.2f}")
