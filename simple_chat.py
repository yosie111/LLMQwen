import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. הגדרות המודל
# אנחנו משתמשים בגרסת ה-Instruct כי היא אומנה במיוחד לנהל שיחה (שאלות ותשובות)
# ולא סתם להשלים משפטים כמו גרסת ה-Base.
#model_id = "Qwen/Qwen2.5-0.5B-Instruct"
model_id = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"Loading model: {model_id}...")
print("This might take a minute mostly for the download...")

# 2. טעינת ה'מתרגם' (Tokenizer)
# המחשב לא מבין מילים, הוא מבין מספרים. הטוקנייזר הופך טקסט למספרים והפוך.
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 3. טעינת המוח (Model)
# אנחנו טוענים אותו ל-CPU (כי אין GPU) בפורמט float32 (הכי יציב למעבדים)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float32,
    device_map="cpu" 
)

print("\n--- Model Loaded! Start chatting (type 'exit' to stop) ---\n")

# 4. לולאת השיחה (The Chat Loop)
# כאן אנחנו בונים את ההיסטוריה של השיחה כדי שהמודל יזכור מה אמרנו לו
chat_history = [] 

while True:
    # קבלת קלט מהמשתמש
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        break

    # הוספת ההודעה החדשה להיסטוריה
    # הפורמט הזה (user/assistant) הוא השפה הפנימית ש-Qwen מבין
    chat_history.append({"role": "user", "content": user_input})

    # הכנת הטקסט למודל (מחיל את התבנית המיוחדת של הצ'אט)
    text = tokenizer.apply_chat_template(
        chat_history, 
        tokenize=False, 
        add_generation_prompt=True
    )

    # המרת הטקסט למספרים (Tokens)
    model_inputs = tokenizer([text], return_tensors="pt").to("cpu")

    # הרצת המודל! (Generating)
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=256,   # מקסימום אורך התשובה
        do_sample=True,       # מאפשר יצירתיות (לא בוחר תמיד את המילה הכי סבירה)
        temperature=0.7       # רמת היצירתיות (0.7 זה מאוזן)
    )

    # חילוץ התשובה בלבד (מסנן את השאלה המקורית)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # המרה חזרה ממספרים למילים
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"AI: {response}")

    # הוספת התשובה להיסטוריה (כדי שיזכור אותה לפעם הבאה)
    chat_history.append({"role": "assistant", "content": response})