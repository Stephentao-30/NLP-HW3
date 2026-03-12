import json, string, re

def normalize_answer(s):
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))

def exact_match_score(p, g): return normalize_answer(p) == normalize_answer(g)

with open("reference_answers.json") as f:
    ref = json.load(f)
with open("predictions_100.json") as f:
    preds = json.load(f)
with open("contexts.txt") as f:
    contexts = [line.strip() for line in f]

keys = sorted(ref.keys(), key=lambda x: int(x[1:]))
verbose, no_recall, wrong_ent, unsure_list = [], [], [], []

for i, key in enumerate(keys):
    expected = [a.strip() for a in ref[key]["answers"].split("|")]
    pred = preds.get(key, "")
    em = max(int(exact_match_score(pred, gt)) for gt in expected)
    if em: continue
    
    ctx = contexts[i] if i < len(contexts) else ""
    norm_ctx = normalize_answer(ctx)
    recall_hit = any(normalize_answer(gt) in norm_ctx for gt in expected)
    norm_pred = normalize_answer(pred)
    
    icon = "R" if recall_hit else "X"
    if "unsure" in pred.lower():
        unsure_list.append((key, "has_ctx" if recall_hit else "no_ctx"))
    elif any(normalize_answer(gt) in norm_pred for gt in expected):
        verbose.append(key)
    elif not recall_hit:
        no_recall.append(key)
    else:
        wrong_ent.append(key)

    q = ref[key]["question"]
    answers = ref[key]["answers"]
    print(f"[{icon}] {key}: Q={q}")
    print(f"    Exp={answers}  Got={pred}")

print()
print(f"Verbose: {len(verbose)} -> {verbose}")
print(f"No recall: {len(no_recall)} -> {no_recall}")
print(f"Wrong entity: {len(wrong_ent)} -> {wrong_ent}")
print(f"Unsure: {len(unsure_list)} -> {unsure_list}")
print(f"Total wrong: {len(verbose)+len(no_recall)+len(wrong_ent)+len(unsure_list)}")
