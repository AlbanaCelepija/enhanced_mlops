"""
Fine-tune an open LLM for grounded Italian QA (public-administration demo).

Dataset : crux82/squad_it  (Italian question/answer pairs with a source passage)
Method  : QLoRA (4-bit base model + LoRA adapters) via TRL's SFTTrainer
Hardware: runs on a single 16 GB GPU (e.g. Colab T4) with the defaults below

Install:
    pip install -U transformers datasets trl peft bitsandbytes accelerate
"""

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
import torch

# ----------------------------------------------------------------------------
# 1. Config
# ----------------------------------------------------------------------------

MODEL_ID = "sapienzanlp/Minerva-7B-instruct-v1.0"
# Faster alternatives for a first run:
#   "Qwen/Qwen2.5-1.5B-Instruct"   (small, multilingual, trains in minutes)
#   "google/gemma-2-2b-it"

OUTPUT_DIR = "minerva-squad-it-lora"
N_TRAIN = 4000   # subset for a quick demo; set to None to use all ~54k examples
N_EVAL = 200

SYSTEM_PROMPT = (
    "Sei un assistente della pubblica amministrazione italiana. "
    "Rispondi alla domanda dell'utente usando esclusivamente le informazioni "
    "contenute nel CONTESTO fornito. Se il contesto non contiene la risposta, "
    "rispondi: 'Non trovo questa informazione nei documenti disponibili.' "
    "Rispondi in modo breve e chiaro."
)

# ----------------------------------------------------------------------------
# 2. Data: convert SQuAD-it into chat format
# ----------------------------------------------------------------------------

def to_chat(example):
    """SQuAD-it row -> {'messages': [...]}, the format SFTTrainer expects."""
    answer = example["answers"]["text"][0]
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"CONTESTO:\n{example['context']}\n\nDOMANDA: {example['question']}",
            },
            {"role": "assistant", "content": answer},
        ]
    }


raw = load_dataset("crux82/squad_it")
columns = raw["train"].column_names

train_ds = raw["train"]
if N_TRAIN:
    train_ds = train_ds.shuffle(seed=42).select(range(N_TRAIN))
train_ds = train_ds.map(to_chat, remove_columns=columns)

eval_ds = raw["test"].shuffle(seed=42).select(range(N_EVAL)).map(to_chat, remove_columns=columns)

print(train_ds[0]["messages"][1]["content"][:300])

# ----------------------------------------------------------------------------
# 3. Model + LoRA
# ----------------------------------------------------------------------------

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

# ----------------------------------------------------------------------------
# 4. Train
# ----------------------------------------------------------------------------

args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_length=1024,          # SQuAD contexts are long; raise if you truncate too much
    bf16=True,
    logging_steps=25,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="epoch",
    gradient_checkpointing=True,
    report_to="none",
    model_init_kwargs={"quantization_config": quant_config, "dtype": torch.bfloat16},
)

trainer = SFTTrainer(
    model=MODEL_ID,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    peft_config=peft_config,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)

# ----------------------------------------------------------------------------
# 5. Quick sanity check
# ----------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = trainer.model
model.eval()

context = (
    "La carta d'identita' elettronica (CIE) si richiede presso il Comune di residenza. "
    "Il costo e' di 16,79 euro piu' i diritti fissi e di segreteria. "
    "Il documento viene spedito entro 6 giorni lavorativi dalla richiesta."
)
question = "Quanto costa la carta d'identita' elettronica?"

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": f"CONTESTO:\n{context}\n\nDOMANDA: {question}"},
]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)

out = model.generate(inputs, max_new_tokens=64, do_sample=False)
print(tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True))
