import json

with open("reference_answers.json") as f:
    ref = json.load(f)
with open("predictions_100.json") as f:
    preds = json.load(f)
with open("contexts.txt") as f:
    contexts = [line.strip() for line in f]

wrong_entity = ["q3", "q4", "q23", "q29", "q30", "q32", "q33", "q79", "q90", "q93"]
for key in wrong_entity:
    idx = int(key[1:]) - 1
    expected = ref[key]["answers"]
    ctx = contexts[idx][:400] if idx < len(contexts) else ""
    print(f"{key}: {ref[key]['question']}")
    print(f"  Expected: {expected}  Got: {preds[key]}")
    print(f"  Ctx start: {ctx[:250]}...")
    print()
