import torch
from transformers import AutoModelForCausalLM

# 1. טעינת המודל (במצב ברירת מחדל הוא "פתוח" לשינויים)
model_id = "Qwen/Qwen2.5-0.5B"
print(f"Loading {model_id}...")
model = AutoModelForCausalLM.from_pretrained(model_id)

# פונקציית עזר לבדוק כמה פרמטרים פתוחים לשינוי
def count_trainable_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

print(f"\nפרמטרים פתוחים לשינוי לפני הקפאה: {count_trainable_parameters(model):,}")
# (במודל הזה תראה כ-494 מיליון פרמטרים פתוחים)

# --- שלב ההגנה (הקפאה) ---
print("מבצע הקפאה (Freezing)...")

# עוברים על כל הפרמטרים במודל ונועלים אותם
for param in model.parameters():
    param.requires_grad = False

# --- בדיקה ---
print(f"פרמטרים פתוחים לשינוי אחרי הקפאה: {count_trainable_parameters(model)}")
# (התוצאה צריכה להיות 0 - המודל נעול לחלוטין!)

# --- ניסוי: האם האימון באמת נחסם? ---
print("\nמנסה לבצע צעד אימון (Backpropagation)...")
try:
    # מריצים קלט דמה דרך המודל
    inputs = torch.randint(0, 1000, (1, 10)) # סתם מספרים
    outputs = model(inputs, labels=inputs)
    
    loss = outputs.loss
    loss.backward() # כאן המחשב מנסה לחשב איך לשנות את המשקולות
    
    print("❌ שגיאה: הצלחתי לבצע Backward (זה לא אמור לקרות אם הכל קפוא!)")
except RuntimeError as e:
    # זו השגיאה שאנחנו מצפים לה!
    print("✅ הצלחה! המודל זרק שגיאה כי אין לו מה לעדכן.")
    print(f"השגיאה שהתקבלה: {e}")

# הערה חשובה: אם תבדוק את ה-Gradients של המשקולות, תראה שהם None (ריק)
first_weight = list(model.parameters())[0]
print(f"\nהאם יש נגזרת למשקולת הראשונה? {first_weight.grad}")