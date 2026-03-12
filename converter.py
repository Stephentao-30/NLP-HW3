import json

def convert_preds(preds_txt, output_json):
    """Converts line-by-line text to SQuAD-style prediction dict."""
    with open(preds_txt, 'r') as f:
        preds = [line.strip() for line in f.readlines()]
    
    formatted_preds = {f"q{i+1}": pred for i, pred in enumerate(preds)}
    
    with open(output_json, 'w') as f:
        json.dump(formatted_preds, f, indent=4)
    print(f"✅ Formatted predictions saved to {output_json}")

if __name__ == "__main__":
    import sys
    preds_txt = sys.argv[1] if len(sys.argv) > 1 else 'predictions.txt'
    output_json = sys.argv[2] if len(sys.argv) > 2 else 'predictions.json'
    convert_preds(preds_txt, output_json)