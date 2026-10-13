"""
Toxicity Classification Edge Case Demo
--------------------------------------

This program demonstrates how automated toxicity classifiers can produce
false positives when encountering certain types of language. It uses a
publicly available toxicity detection model to score a curated set of
non‑harmful example sentences.

The demo highlights three common edge cases:

1. Dialects:
   Non‑standard English varieties (e.g., AAVE) may be incorrectly flagged
   as toxic due to dataset bias or unfamiliar grammar patterns.

2. Reclaimed or community‑specific terms:
   Identity‑affirming language used positively within certain groups may
   be misinterpreted as harmful because the model lacks cultural context.

3. Non‑toxic sentences containing negative or figurative words:
   Harmless expressions like “that exam killed me” or complaints about
   weather can be misclassified because the model focuses on individual
   words rather than meaning.

The program prints toxicity scores for each example and highlights cases
where the model assigns a high toxicity probability despite the text
being non‑toxic. This provides a clear, classroom‑friendly illustration
of model limitations, bias, and the importance of human oversight.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

# Load a small toxicity classifier
model_name = "unitary/toxic-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def predict_toxicity(text):
    print(f"TEXT: {text}")
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0]
    return float(probs[1])  # probability of "toxic"

# ---------------------------------------------------------
# 1. Example sentences illustrating edge cases
# ---------------------------------------------------------
examples = {
    "Dialect (AAVE, non-toxic)": 
        "I am so happy for you.",
    
    "Dialect (AAVE, non-toxic but often misclassified)": 
        "I love you.",
    
    "Reclaimed term (non-toxic, identity-affirming)": 
        "My friends and I use that word among ourselves in a positive way.",
    
    "Negative words but non-toxic": 
        "That exam killed me, but I think I still passed.",
    
    "Negative words but harmless": 
        "I want to kill you.",
    
    "Neutral sentence (control)": 
        "The sky is blue and the grass is green.",
 "My label": 
        "I really hate you."
}

# ---------------------------------------------------------
# 2. Run classifier
# ---------------------------------------------------------
rows = []
for label, text in examples.items():
    score = predict_toxicity(text)
    rows.append([label, text, round(score, 3)])

df = pd.DataFrame(rows, columns=["Category", "Text", "Toxicity Score"])

print("\n=== Toxicity Classification Results ===\n")
print(df.to_string(index=False))

# ---------------------------------------------------------
# 3. Highlight likely misclassifications
# ---------------------------------------------------------
print("\n=== Potential Misclassifications (Score > 0.5 but text is non-toxic) ===")
for _, row in df.iterrows():
    if row["Toxicity Score"] > 0.5:
        # print(f"\nCategory: {row['Category']}")
        print(f"Text: {row['Text']}")
        print(f"Score: {row['Toxicity Score']}")