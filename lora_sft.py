import json
import yaml
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
import subprocess

CONFIG = {
    "model": "mlx-community/Qwen3-4B-4bit",
    "dataset": "gbharti/finance-alpaca",
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_layers": 16,
    "learning_rate": 2e-5,
    "batch_size": 2,
    "epochs": 0.5,
    "save_every": 2000,
    "max_seq_length": 512,
}

ADAPTER_DIR = Path("./adapters") / f"qwen3-4b-finance-{datetime.now():%Y%m%d_%H%M%S}"
ADAPTER_DIR.mkdir(parents=True, exist_ok=True)



def estimate_lora_params(hidden_size: int, num_layers: int, rank: int, num_modules: int = 4):
    """
    Estimate trainable LoRA parameters.

    For each adapted layer, LoRA adds low-rank matrices to attention projections:
    - Each LoRA adapter: 2 * hidden_size * rank parameters (A and B matrices)
    - Default targets 4 modules per layer: q_proj, k_proj, v_proj, o_proj
    """
    params_per_layer = num_modules * 2 * hidden_size * rank
    total_lora_params = params_per_layer * num_layers
    return total_lora_params

TOTAL_MODEL_PARAMS = 4_000_000_000
LORA_PARAMS = estimate_lora_params(
    hidden_size=2560,
    num_layers=CONFIG["lora_layers"],
    rank=CONFIG["lora_rank"],
)

print("\n" + "=" * 60)
print("LORA CONFIGURATION")
print("=" * 60)
print(f"Model: {CONFIG['model']}")
print(f"LoRA rank: {CONFIG['lora_rank']}, alpha: {CONFIG['lora_alpha']}")
print(f"Adapted layers: last {CONFIG['lora_layers']} of 40 (layers 24-39)")
print(f"Trainable parameters: ~{LORA_PARAMS:,} ({LORA_PARAMS/TOTAL_MODEL_PARAMS*100:.3f}%)")
print(f"Adapter save path: {ADAPTER_DIR.absolute()}")
print("=" * 60 + "\n")

# =============================================================================
# DATASET PREPARATION
# =============================================================================

print("Loading finance-alpaca dataset...")
finance_alpaca = load_dataset(CONFIG["dataset"], split="train")

def format_instruction(example):
    instruction = example['instruction']
    input_text = example.get('input', '')
    output = example['output']

    if input_text:
        user_message = f"{instruction}\n\nContext: {input_text}"
    else:
        user_message = instruction

    return {
        "messages": [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": output}
        ]
    }

formatted_data = [format_instruction(ex) for ex in finance_alpaca]
train_size = int(0.95 * len(formatted_data))

data_dir = Path("./data")
data_dir.mkdir(exist_ok=True)

with open(data_dir / "train.jsonl", 'w') as f:
    for item in formatted_data[:train_size]:
        f.write(json.dumps(item) + '\n')

with open(data_dir / "valid.jsonl", 'w') as f:
    for item in formatted_data[train_size:]:
        f.write(json.dumps(item) + '\n')

print(f"Dataset prepared: {train_size:,} train, {len(formatted_data) - train_size:,} val samples")

steps_per_epoch = train_size // CONFIG["batch_size"]
num_iterations = 10000
print(f"Epochs: {CONFIG['epochs']} × {steps_per_epoch:,} steps/epoch = {num_iterations:,} iterations")

lora_config = {
    "lora_parameters": {
        "rank": CONFIG["lora_rank"],
        "alpha": CONFIG["lora_alpha"],
        "dropout": 0.05,
        "scale": CONFIG["lora_alpha"] / CONFIG["lora_rank"],
    },

    "lr_schedule": {
        "name": "cosine_decay",
        "warmup": 100,
        "warmup_init": 1e-7,
        "arguments": [2e-5, num_iterations, 1e-6],
    }
}
with open(ADAPTER_DIR / "lora_config.yaml", 'w') as f:
    yaml.dump(lora_config, f)

print(f"LR schedule: warmup 100 steps → 2e-5 → cosine decay to 1e-6")

# Save training config for reproducibility
with open(ADAPTER_DIR / "training_config.json", 'w') as f:
    json.dump(CONFIG, f, indent=2)

# =============================================================================
# TRAINING
# =============================================================================

subprocess.run([
    "python", "-m", "mlx_lm", "lora",
    "--model", CONFIG["model"],
    "--data", "./data",
    "--train",
    "--iters", str(num_iterations),
    "--batch-size", str(CONFIG["batch_size"]),
    "--learning-rate", str(CONFIG["learning_rate"]),
    "--num-layers", str(CONFIG["lora_layers"]),
    "--max-seq-length", str(CONFIG["max_seq_length"]),
    "--steps-per-eval", "100",
    "--val-batches", "25",
    "--adapter-path", str(ADAPTER_DIR),
    "--save-every", str(CONFIG["save_every"]),
    "--config", str(ADAPTER_DIR / "lora_config.yaml"),
    "--report-to", "wandb",
    "--project-name", "qwen3-finance-test",
    "--optimizer", "adamw",
])

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print(f"Adapters saved to: {ADAPTER_DIR.absolute()}")
print("\nTo use:")
print(f'  mlx_lm.generate --model {CONFIG["model"]} --adapter-path {ADAPTER_DIR}')
print("=" * 60)