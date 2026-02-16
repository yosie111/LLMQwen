import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- הגדרות ראשוניות ---
model_id = "Qwen/Qwen2.5-0.5B"

print(f"1. Loading Tokenizer for {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("2. Loading Model (CPU Mode)...")
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

print("\n" + "="*50)
print("   חלק א': איך המודל רואה טקסט (Tokenization)")
print("="*50)

# טקסט לדוגמה שמשלב אנגלית ועברית כדי לראות את ההבדלים
text = "Hello world, שלום עולם"
inputs = tokenizer(text, return_tensors="pt")

print(f"הטקסט המקורי: '{text}'")
print(f"המספרים שהמודל רואה (IDs): {inputs.input_ids[0].tolist()}")

print("\n--- פירוק למרכיבים (Decoding) ---")
# עוברים מספר-מספר ורואים איזה חלק מהמילה הוא מייצג
for t_id in inputs.input_ids[0]:
    decoded_word = tokenizer.decode([t_id])
    print(f"ID: {t_id:<6} -> Token: '{decoded_word}'")


print("\n" + "="*50)
print("   חלק ב': איך המודל מחשב הסתברויות (Prediction)")
print("="*50)

input_text = "The color of the sky is"
print(f"הקלט למודל: '{input_text}'")

inputs = tokenizer(input_text, return_tensors="pt")

# הרצת המודל (Forward Pass)
# אנחנו לא מבקשים ממנו לייצר טקסט (generate), אלא רק לחשב את המתמטיקה
with torch.no_grad():
    outputs = model(**inputs)

# outputs.logits מכיל את הניקוד לכל המילים במילון עבור כל מילה במשפט.
# אנחנו רוצים רק את הניחוש עבור המילה *האחרונה* (הבאה בתור).
next_token_logits = outputs.logits[0, -1, :]

# המרת הניקוד הגולמי (Logits) לאחוזים (Softmax)
probs = F.softmax(next_token_logits, dim=-1)

# שליפת 5 הניחושים המובילים (Top 5)
top_5_probs, top_5_indices = torch.topk(probs, 5)

print("\n--- 5 הניחושים של המודל למילה הבאה ---")
for i in range(5):
    token_id = top_5_indices[i].item()
    probability = top_5_probs[i].item()
    predicted_word = tokenizer.decode([token_id])
    
    print(f"{i+1}. '{predicted_word}' \t(סיכוי: {probability:.2%})")


print("\n" + "="*50)
print("   חלק ג': המבנה הפנימי (Architecture)")
print("="*50)

# הדפסה של עץ השכבות המלא של המודל
print(model)

# בדיקה עמוקה יותר של השכבה הראשונה
first_layer = model.model.layers[0]
print("\n--- מה יש בתוך השכבה הראשונה (Layer 0)? ---")
print(f"מנגנון הקשב (Attention): {first_layer.self_attn}")
print(f"רשת הנוירונים (MLP):      {first_layer.mlp}")

print("\nסיימנו! כעת אתה רואה את המודל 'מבפנים'.")