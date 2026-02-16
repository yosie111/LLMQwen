import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

text = "Explain shortly what is Python programming language."
inputs = tokenizer(text, return_tensors="pt")

print(f"השאלה: {text}\n")

# --- ניסוי 1: קמצן במילים (Max Tokens = 10) ---
print("--- ניסוי 1: max_new_tokens=10 ---")
output1 = model.generate(
    **inputs, 
    max_new_tokens=10  # קצר מאוד!
)
print(tokenizer.decode(output1[0], skip_special_tokens=True))
print("(שים לב: הוא כנראה נקטע באמצע משפט)")

# --- ניסוי 2: נדיב במילים (Max Tokens = 100) ---
print("\n--- ניסוי 2: max_new_tokens=100 ---")
output2 = model.generate(
    **inputs, 
    max_new_tokens=100 # נותן לו אוויר
)
print(tokenizer.decode(output2[0], skip_special_tokens=True))