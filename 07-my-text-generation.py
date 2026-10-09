from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load a small, fast model for classroom demos
model_name = "gpt2"   # You can swap in distilgpt2 for even faster performance
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate(prompt, temperature=1.0, top_k=50, max_new_tokens=60):
    """
    Generate text using a prompt, temperature, and top-k sampling.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    
    output = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)


# -----------------------------
# Example usage for a classroom
# -----------------------------
if __name__ == "__main__":
    prompt = "Tell me a joke."

    print("\n=== Low Temperature (0.2) — Deterministic ===")
    print(generate(prompt, temperature=0.2, top_k=50))

    print("\n=== Medium Temperature (1.0) — Balanced ===")
    print(generate(prompt, temperature=1.0, top_k=50))

    print("\n=== High Temperature (1.8) — Creative/Chaotic ===")
    print(generate(prompt, temperature=1.8, top_k=50))

    print("\n=== Top‑k = 10 (More constrained sampling) ===")
    print(generate(prompt, temperature=1.0, top_k=10))

    print("\n=== Top‑k = 100 (More freedom) ===")
    print(generate(prompt, temperature=1.0, top_k=100))

    print("\n=== Top‑k = 150 (Even more freedom) ===")
    print(generate(prompt, temperature=1.9, top_k=150))