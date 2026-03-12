import json

def extract_questions():
    with open('reference_answers.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    with open('questions.txt', 'w', encoding='utf-8') as f:
        for key, item in data.items():
            f.write(f"{item['question']}\n")
            
    print(f"Successfully extracted {len(data)} questions into questions.txt!")

if __name__ == "__main__":
    extract_questions()