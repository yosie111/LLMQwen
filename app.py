"""
Chat UI Backend - Flask server for Qwen chatbot
Dual model support: 0.5B and 1.5B
Full parameter control + Weight Protection System
"""
import torch
import copy
import gc
from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import time

app = Flask(__name__, template_folder="templates", static_folder="static")

# ========== Available Models ==========
AVAILABLE_MODELS = {
    "0.5B": {
        "id": "Qwen/Qwen2.5-0.5B-Instruct",
        "name": "Qwen 2.5 — 0.5B",
        "description_he": "מודל קטן ומהיר. מומלץ למחשבים חלשים.",
        "description_en": "Small & fast. Recommended for low-spec machines.",
        "size_hint": "~1GB RAM",
    },
    "1.5B": {
        "id": "Qwen/Qwen2.5-1.5B-Instruct",
        "name": "Qwen 2.5 — 1.5B",
        "description_he": "מודל גדול וחכם יותר. דורש יותר זיכרון ואיטי יותר.",
        "description_en": "Larger & smarter. Requires more RAM, slower responses.",
        "size_hint": "~3GB RAM",
    },
}

# ========== Model State ==========
current_model_key = None   # "0.5B" or "1.5B"
model = None
tokenizer = None
original_state_dict = None

weight_protection = {
    "frozen": True,
    "modified_layers": [],
    "freeze_history": [],
}

chat_sessions = {}

DEFAULT_PARAMS = {
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "length_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "num_beams": 1,
    "do_sample": True,
    "early_stopping": False,
}


def load_model(model_key: str):
    """Load a model by key, unloading any previous model first."""
    global model, tokenizer, original_state_dict, current_model_key, weight_protection

    if model_key not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model key: {model_key}")

    model_info = AVAILABLE_MODELS[model_key]
    model_id = model_info["id"]

    # Unload previous model to free RAM
    if model is not None:
        print(f"🗑 Unloading previous model: {current_model_key}...")
        del model
        del tokenizer
        del original_state_dict
        gc.collect()

    print(f"🔄 Loading model: {model_id}...")
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        #torch_dtype=torch.float32,
        dtype=torch.float32,
        device_map="cpu"
    )

    elapsed = time.time() - start
    print(f"✅ Model loaded in {elapsed:.1f}s!")

    # Save original weights
    print("💾 Saving original weights snapshot...")
    original_state_dict = copy.deepcopy(model.state_dict())
    print("✅ Original weights saved!")

    # Freeze all by default
    for param in model.parameters():
        param.requires_grad = False

    current_model_key = model_key
    weight_protection = {
        "frozen": True,
        "modified_layers": [],
        "freeze_history": [{
            "action": f"model_loaded ({model_key})",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }],
    }

    return elapsed


# ========== Load default model on startup ==========
load_model("0.5B")


# ========== Helper Functions ==========

def get_layer_groups():
    """Get organized layer groups for the UI."""
    groups = {}
    for name, param in model.named_parameters():
        parts = name.split('.')
        group_key = '.'.join(parts[:3]) if len(parts) >= 3 else parts[0]

        if group_key not in groups:
            groups[group_key] = {
                "name": group_key,
                "params": [],
                "total_params": 0,
                "frozen": True,
            }

        groups[group_key]["params"].append(name)
        groups[group_key]["total_params"] += param.numel()
        if param.requires_grad:
            groups[group_key]["frozen"] = False

    return groups


def compute_weight_diff():
    """Compute which layers have changed from original."""
    current = model.state_dict()
    changed = []
    total_diff = 0.0

    for key in original_state_dict:
        if key in current:
            diff = torch.sum(torch.abs(current[key] - original_state_dict[key])).item()
            if diff > 1e-8:
                changed.append({
                    "layer": key,
                    "diff": round(diff, 6),
                    "shape": list(current[key].shape),
                    "num_params": current[key].numel(),
                })
                total_diff += diff

    return {
        "changed_count": len(changed),
        "total_layers": len(original_state_dict),
        "total_diff": round(total_diff, 6),
        "changed_layers": changed[:50],
    }


def generate_response(chat_history: list, params: dict) -> dict:
    """Generate a response from the model given chat history and full params."""
    text = tokenizer.apply_chat_template(
        chat_history, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to("cpu")
    input_token_count = model_inputs.input_ids.shape[1]

    gen_kwargs = {
        "max_new_tokens": params.get("max_tokens", 512),
        "do_sample": params.get("do_sample", True),
        "num_beams": params.get("num_beams", 1),
        "length_penalty": params.get("length_penalty", 1.0),
        "early_stopping": params.get("early_stopping", False),
    }

    if gen_kwargs["do_sample"]:
        gen_kwargs["temperature"] = params.get("temperature", 0.7)
        gen_kwargs["top_p"] = params.get("top_p", 0.9)
        gen_kwargs["top_k"] = params.get("top_k", 50)

    rep_penalty = params.get("repetition_penalty", 1.1)
    if rep_penalty != 1.0:
        gen_kwargs["repetition_penalty"] = rep_penalty

    ngram = params.get("no_repeat_ngram_size", 0)
    if ngram > 0:
        gen_kwargs["no_repeat_ngram_size"] = ngram

    start_time = time.time()
    generated_ids = model.generate(model_inputs.input_ids, **gen_kwargs)
    elapsed = time.time() - start_time

    output_ids = generated_ids[0][input_token_count:]
    output_token_count = len(output_ids)
    tokens_per_sec = output_token_count / elapsed if elapsed > 0 else 0
    response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    return {
        "response": response,
        "elapsed_seconds": round(elapsed, 2),
        "input_tokens": input_token_count,
        "output_tokens": output_token_count,
        "tokens_per_sec": round(tokens_per_sec, 1),
        "params_used": {k: v for k, v in gen_kwargs.items()},
    }


# ========== Routes ==========

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "").strip()
    session_id = data.get("session_id", "default")
    system_prompt = data.get("system_prompt", "")
    params = data.get("params", DEFAULT_PARAMS)

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
        if system_prompt:
            chat_sessions[session_id].append({"role": "system", "content": system_prompt})

    history = chat_sessions[session_id]
    history.append({"role": "user", "content": user_message})

    try:
        result = generate_response(history, params)
        result["model"] = current_model_key
        history.append({"role": "assistant", "content": result["response"]})
        result["history_length"] = len(history)
        return jsonify(result)
    except Exception as e:
        history.pop()
        return jsonify({"error": str(e)}), 500


@app.route("/api/reset", methods=["POST"])
def reset():
    data = request.json
    session_id = data.get("session_id", "default")
    chat_sessions.pop(session_id, None)
    return jsonify({"status": "ok"})


# ========== Model Management API ==========

@app.route("/api/models", methods=["GET"])
def list_models():
    """List available models and which one is active."""
    models = []
    for key, info in AVAILABLE_MODELS.items():
        models.append({
            "key": key,
            "id": info["id"],
            "name": info["name"],
            "description_he": info["description_he"],
            "description_en": info["description_en"],
            "size_hint": info["size_hint"],
            "active": key == current_model_key,
        })
    return jsonify({
        "models": models,
        "current": current_model_key,
    })


@app.route("/api/models/switch", methods=["POST"])
def switch_model():
    """Switch to a different model."""
    data = request.json
    model_key = data.get("model_key", "")

    if model_key == current_model_key:
        return jsonify({"status": "already_active", "model": model_key})

    if model_key not in AVAILABLE_MODELS:
        return jsonify({"error": f"Unknown model: {model_key}"}), 400

    try:
        elapsed = load_model(model_key)

        # Clear all chat sessions on model switch
        chat_sessions.clear()

        return jsonify({
            "status": "ok",
            "model": model_key,
            "load_time": round(elapsed, 1),
            "message": f"Switched to {AVAILABLE_MODELS[model_key]['name']}",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/model_info", methods=["GET"])
def model_info():
    """Return model metadata for the status panel."""
    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = param_count - trainable
    info = AVAILABLE_MODELS[current_model_key]
    return jsonify({
        "model_id": info["id"],
        "model_key": current_model_key,
        "model_name": info["name"],
        "parameters": param_count,
        "parameters_human": f"{param_count / 1e6:.0f}M",
        "trainable_params": trainable,
        "frozen_params": frozen,
        "dtype": str(next(model.parameters()).dtype),
        "device": str(next(model.parameters()).device),
        "vocab_size": tokenizer.vocab_size,
        "max_model_length": getattr(tokenizer, "model_max_length", "N/A"),
        "size_hint": info["size_hint"],
    })


# ========== Weight Protection API ==========

@app.route("/api/weights/status", methods=["GET"])
def weights_status():
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    diff = compute_weight_diff()
    groups = get_layer_groups()
    layer_summary = []
    for key, g in sorted(groups.items()):
        layer_summary.append({
            "name": g["name"],
            "params": g["total_params"],
            "frozen": g["frozen"],
            "count": len(g["params"]),
        })
    return jsonify({
        "frozen": weight_protection["frozen"],
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": frozen,
        "frozen_pct": round(frozen / total * 100, 1) if total > 0 else 0,
        "trainable_pct": round(trainable / total * 100, 1) if total > 0 else 0,
        "diff": diff,
        "layer_groups": layer_summary,
        "history": weight_protection["freeze_history"][-20:],
        "model_key": current_model_key,
    })


@app.route("/api/weights/freeze", methods=["POST"])
def freeze_weights():
    for param in model.parameters():
        param.requires_grad = False
    weight_protection["frozen"] = True
    weight_protection["freeze_history"].append({
        "action": "freeze_all",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    return jsonify({"status": "ok", "message": "All weights frozen"})


@app.route("/api/weights/unfreeze", methods=["POST"])
def unfreeze_weights():
    data = request.json or {}
    layer_filter = data.get("layer_filter", None)
    if layer_filter:
        count = 0
        for name, param in model.named_parameters():
            if layer_filter in name:
                param.requires_grad = True
                count += 1
        weight_protection["freeze_history"].append({
            "action": f"unfreeze_partial ({layer_filter}): {count}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
    else:
        for param in model.parameters():
            param.requires_grad = True
        weight_protection["freeze_history"].append({
            "action": "unfreeze_all",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
    weight_protection["frozen"] = False
    return jsonify({"status": "ok"})


@app.route("/api/weights/freeze_layer", methods=["POST"])
def freeze_layer():
    data = request.json
    layer_name = data.get("layer_name", "")
    freeze = data.get("freeze", True)
    count = 0
    for name, param in model.named_parameters():
        if name.startswith(layer_name):
            param.requires_grad = not freeze
            count += 1
    action = "freeze" if freeze else "unfreeze"
    weight_protection["freeze_history"].append({
        "action": f"{action}_layer ({layer_name}): {count}",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    any_trainable = any(p.requires_grad for p in model.parameters())
    weight_protection["frozen"] = not any_trainable
    return jsonify({"status": "ok", "count": count})


@app.route("/api/weights/reset", methods=["POST"])
def reset_weights():
    model.load_state_dict(copy.deepcopy(original_state_dict))
    for param in model.parameters():
        param.requires_grad = False
    weight_protection["frozen"] = True
    weight_protection["modified_layers"] = []
    weight_protection["freeze_history"].append({
        "action": "reset_to_original",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    gc.collect()
    return jsonify({"status": "ok"})


@app.route("/api/weights/perturb", methods=["POST"])
def perturb_weights():
    data = request.json or {}
    layer_filter = data.get("layer_filter", "")
    noise_scale = data.get("noise_scale", 0.001)
    if weight_protection["frozen"]:
        return jsonify({"error": "Weights are frozen. Unfreeze first."}), 400
    if noise_scale > 0.1:
        return jsonify({"error": "Noise scale too high (max 0.1)"}), 400
    count = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad and (not layer_filter or layer_filter in name):
                noise = torch.randn_like(param) * noise_scale
                param.add_(noise)
                count += 1
    weight_protection["freeze_history"].append({
        "action": f"perturb (scale={noise_scale}): {count}",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    return jsonify({"status": "ok", "perturbed_count": count})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
