import torch
from transformers import AutoModelForCausalLM

model_id = "Qwen/Qwen2.5-0.5B"
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_id)

# 1. בוא ניגש לשכבה הראשונה, למנגנון ה-Attention
# המשקולות נמצאות בתוך אובייקט שנקרא 'weight' והן מסוג PyTorch Tensor
layer0_weights = model.model.layers[0].self_attn.q_proj.weight

print("\n--- לפני השינוי ---")
print(f"הצורה של המטריצה: {layer0_weights.shape}")
print(f"הערך הראשון במטריצה: {layer0_weights[0, 0].item()}")

# 2. האם המודל באמת שלי? בוא ננסה לשנות ערך
# torch.no_grad() אומר לפייתון: "אני עושה ניתוח ידני, אל תחשב נגזרות כרגע"
with torch.no_grad():
    # נשנה את המספר הראשון ל-999 (סתם מספר לא הגיוני)
    layer0_weights[0, 0] = 999.0

print("\n--- אחרי השינוי ---")
print(f"הערך הראשון במטריצה: {layer0_weights[0, 0].item()}")

# 3. בדיקה האם זה השפיע?
if layer0_weights[0, 0].item() == 999.0:
    print("\n✅ הצלחה! שינית פיזית את המוח של המודל.")
else:
    print("\n❌ נכשל.")

# 4. בונוס: "לובוטומיה" (הריסת המודל)
# בוא נראה מה קורה אם מאפסים שכבה שלמה
print("\nמבצע איפוס לשכבה שלמה...")
with torch.no_grad():
    model.model.layers[0].self_attn.q_proj.weight.fill_(0.0)

print(f"הערך הראשון החדש: {layer0_weights[0, 0].item()}")
print("עכשיו המודל 'פגוע מוח' בשכבה הראשונה שלו.")