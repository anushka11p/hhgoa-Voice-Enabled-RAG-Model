"""Pick the guardrail thresholds from data instead of from intuition.

Scores two populations through the same retrieval path:

  in-domain   -- real queries sampled from the dataset's own eval split
  off-domain  -- queries about things the corpus provably does not cover

and reports the separating percentiles for both signals the off-topic gate
uses (nearest-centroid cosine, top cross-encoder score). A good threshold sits
below the in-domain 5th percentile and above the off-domain 95th, so the gate
almost never refuses a real question and almost always catches a bogus one.

Run:  python -m analytics.calibrate_guardrails
"""
import json
import random
import statistics as stats

import config
from embeddings.encoder import encode_query
from retrieval.pipeline import get_engine

# Deliberately outside an MS MARCO general-knowledge corpus: current events,
# personal questions to the assistant, and local specifics. Hindi + English,
# because the demo takes Hindi speech but users code-switch.
OFF_DOMAIN = [
    "who won the cricket world cup in 2011",
    "what is my bank account balance",
    "tell me a joke about programmers",
    "what time does the Panaji ferry leave tomorrow",
    "who is the current prime minister of Japan",
    "write me a python function to reverse a list",
    "what did you have for breakfast",
    "translate this sentence into French for me",
    "what is the wifi password here",
    "book me a table for two at eight",
    "आज गोवा में मौसम कैसा है",
    "मेरा फोन नंबर क्या है",
    "कल का क्रिकेट स्कोर बताओ",
    "मुझे एक कविता सुनाओ",
    "इस कमरे का किराया कितना है",
    "मेरी अगली मीटिंग कब है",
    "तुम्हारा नाम क्या है",
    "मुझे हवाई जहाज़ का टिकट बुक करो",
    "क्या तुम गाना गा सकते हो",
    "मेरे दोस्त को मैसेज भेजो",
    "what is the capital of Mars",
    "how do I reset my iphone passcode",
    "sing me a lullaby in punjabi",
    "what are today's stock prices",
    "who is speaking right now",
    "set an alarm for six in the morning",
    "what is the score of the match",
    "recommend a restaurant near me",
    "how old are you",
    "what is on my calendar today",
    "play some music please",
    "मुझे आज की खबरें सुनाओ",
    "मेरा पासवर्ड बदल दो",
    "गोवा में कौन सा बीच सबसे अच्छा है",
    "मेरी कार की चाबी कहाँ है",
    "अगली ट्रेन कितने बजे है",
    "मुझे एक चुटकुला सुनाओ",
    "तुम कौन से मॉडल हो",
    "मेरे खाते में कितने पैसे हैं",
]


def percentiles(values, ps=(1, 5, 10, 50, 90, 95, 99)):
    if not values:
        return {}
    s = sorted(values)
    return {
        f"p{p}": round(s[min(len(s) - 1, int(len(s) * p / 100))], 4) for p in ps
    }


def main(n_in_domain: int = 120):
    engine = get_engine()
    evals = json.loads(
        (config.DATA_DIR / "eval_queries.json").read_text(encoding="utf-8")
    )
    random.seed(42)
    sample = random.sample(evals, min(n_in_domain, len(evals)))

    rows = {"in_domain": {"centroid": [], "rerank": []},
            "off_domain": {"centroid": [], "rerank": []}}

    for label, queries in (
        ("in_domain", [r["query"] for r in sample]),
        ("off_domain", OFF_DOMAIN),
    ):
        for q in queries:
            qvec = encode_query(q)
            rows[label]["centroid"].append(engine.topic_similarity(qvec))
            res = engine.search(q, qvec=qvec)
            if res.chunks and res.reranked:
                rows[label]["rerank"].append(res.chunks[0].score)

    print("\n=== nearest-centroid cosine ===")
    for label in rows:
        vals = rows[label]["centroid"]
        print(f"{label:11} n={len(vals):4} mean={stats.mean(vals):.4f} {percentiles(vals)}")

    print("\n=== top cross-encoder score ===")
    for label in rows:
        vals = rows[label]["rerank"]
        print(f"{label:11} n={len(vals):4} mean={stats.mean(vals):.4f} {percentiles(vals)}")

    # Suggest thresholds: catch as much off-domain as possible while keeping
    # in-domain false-refusals under ~5%.
    print("\n=== suggested thresholds ===")
    for signal in ("centroid", "rerank"):
        ind = sorted(rows["in_domain"][signal])
        off = sorted(rows["off_domain"][signal])
        if not ind or not off:
            continue
        ind_p5 = ind[max(0, int(len(ind) * 0.05))]
        off_p90 = off[min(len(off) - 1, int(len(off) * 0.90))]
        print(f"{signal:9} in_domain_p5={ind_p5:.4f}  off_domain_p90={off_p90:.4f}")
        for cand in sorted({round(ind_p5, 3), round(off_p90, 3),
                            round((ind_p5 + off_p90) / 2, 3)}):
            fr = sum(1 for v in ind if v < cand) / len(ind)
            catch = sum(1 for v in off if v < cand) / len(off)
            print(f"   threshold {cand:>8}: false-refuse {fr:5.1%} | off-topic caught {catch:5.1%}")

    # A single signal does not separate these populations cleanly, so search
    # the two-signal rule the pipeline actually implements:
    #   refuse if nearest-centroid < C  (pre-retrieval, cheap)
    #   or      if top cross-encoder < R (post-retrieval, accurate)
    print("\n=== combined rule grid (refuse if centroid < C or rerank < R) ===")
    ind = list(zip(rows["in_domain"]["centroid"], rows["in_domain"]["rerank"]))
    off = list(zip(rows["off_domain"]["centroid"], rows["off_domain"]["rerank"]))
    best = []
    for c in [round(x * 0.01, 2) for x in range(20, 46, 2)]:
        for r in [round(x * 0.5, 2) for x in range(-10, 5)]:
            fr = sum(1 for a, b in ind if a < c or b < r) / len(ind)
            catch = sum(1 for a, b in off if a < c or b < r) / len(off)
            best.append((catch, -fr, c, r))
    best.sort(reverse=True)
    print(f"{'centroid':>9} {'rerank':>8} {'false-refuse':>13} {'caught':>8}")
    seen = set()
    for catch, negfr, c, r in best:
        if -negfr > 0.10:      # keep false refusals on real questions under 10%
            continue
        key = round(catch, 2)
        if key in seen:
            continue
        seen.add(key)
        print(f"{c:>9} {r:>8} {-negfr:>12.1%} {catch:>8.1%}")
        if len(seen) >= 8:
            break


if __name__ == "__main__":
    main()
