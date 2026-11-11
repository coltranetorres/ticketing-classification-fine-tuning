"""Sample first-contact customer tweets to retail/e-commerce brands from Customer Support on Twitter (twcs.csv).
Brand choice fixed BEFORE looking at any message content. Writes data/twitter/candidates.jsonl (unlabeled)."""
import csv
import json
import random
import re

BRANDS = {"amazonhelp": 300, "upshelp": 60, "askpaypal": 60, "tesco": 60, "sainsburys": 60,
          "marksandspencer": 30, "morrisons": 30}
STOP = {"the", "to", "my", "i", "is", "a", "and", "it", "you", "me", "for", "have", "not", "can", "of", "in", "this", "on"}


def clean(t: str) -> str:
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"^(@\w+\s*)+", "", t)          # leading handles
    t = re.sub(r"@\d+", "", t)                  # anonymised handles
    return re.sub(r"\s+", " ", t).strip()


def english(t: str) -> bool:
    w = re.findall(r"[a-z']+", t.lower())
    return len(w) >= 4 and sum(x in STOP for x in w) >= 2 and sum(c.isascii() for c in t) / max(len(t), 1) > 0.95


pool = {b: [] for b in BRANDS}
with open("data/twitter/twcs/twcs.csv", encoding="utf-8", newline="") as f:
    for r in csv.DictReader(f):
        if r["inbound"] != "True" or r["in_response_to_tweet_id"]:
            continue
        m = re.match(r"@(\w+)", r["text"])
        b = m.group(1).lower() if m else None
        if b in pool:
            t = clean(r["text"])
            if 5 <= len(t.split()) <= 70 and english(t):
                pool[b].append({"id": r["tweet_id"], "brand": b, "text": t})
rng = random.Random(0)
out, seen = [], set()
for b, n in BRANDS.items():
    rng.shuffle(pool[b])
    k = 0
    for r in pool[b]:
        if r["text"].lower() in seen:
            continue
        seen.add(r["text"].lower()); out.append(r); k += 1
        if k == n:
            break
rng.shuffle(out)
with open("data/twitter/candidates.jsonl", "w") as f:
    for i, r in enumerate(out):
        f.write(json.dumps({"cid": i, **r}) + "\n")
print(len(out), {b: sum(r["brand"] == b for r in out) for b in BRANDS})
