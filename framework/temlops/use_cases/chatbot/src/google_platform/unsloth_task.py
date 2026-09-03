"""Unsloth LoRA fine-tuning as one function, runnable as a Vertex CustomJob.

    python -m src.training.unsloth_task --train-uri gs://bucket/data/train.jsonl

The output is a plain HuggingFace folder (merged 16-bit weights + tokenizer),
which is exactly what `GuardedLLMPredictor.load()` expects — no peft or
unsloth dependency in the serving container.
"""
from __future__ import annotations

# Unsloth MUST be imported before transformers/trl/peft. It monkey-patches
# their internals on import; the reverse order silently disables the fast
# kernels and you lose most of the speedup.
from unsloth import FastLanguageModel  # noqa: E402  isort:skip

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("unsloth_task")


def finetune_with_unsloth(
    train_uri: str,
    output_dir: str | None = None,
    *,
    base_model: str = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
    lora_r: int = 64,
    lora_alpha: int | None = None,
    lora_dropout: float = 0.0,
    epochs: float = 3.0,
    max_steps: int = -1,
    learning_rate: float = 2e-4,
    per_device_batch_size: int = 2,
    grad_accum: int = 4,
    warmup_steps: int = 10,
    seed: int = 3407,
    train_on_responses_only: bool = True,
    save_method: str = "merged_16bit",
    local_workdir: str = "/tmp/unsloth",
) -> str:
    """Fine-tune an open LLM with Unsloth and publish a deployable model.

    Returns the GCS URI of the saved model directory.

    `save_method` matters more than anything else here:
      merged_16bit  -> standalone HF model, loads with AutoModelForCausalLM.
                       This is what the CPR predictor needs.
      merged_4bit   -> standalone but quantized; needs bitsandbytes at serve time.
      lora          -> ADAPTER ONLY. Small and fast to save, but useless to a
                       serving container that doesn't also load the base model.
    """
    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq
    from trl import SFTConfig, SFTTrainer

    from src.data import gcs_io

    work = Path(local_workdir)
    work.mkdir(parents=True, exist_ok=True)

    # Vertex injects AIP_MODEL_DIR and treats whatever lands there as the
    # job's model artifact. Honour it over the CLI default.
    output_dir = (os.environ.get("AIP_MODEL_DIR") or output_dir or "").rstrip("/")
    if not output_dir:
        raise ValueError("Pass --output-dir or run under Vertex (AIP_MODEL_DIR).")
    log.info("Artifacts will be published to %s", output_dir)

    # ---------------------------------------------------------- 1. model --
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,            # None -> bf16 on Ampere+, fp16 otherwise
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha if lora_alpha is not None else lora_r * 2,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,     # the notebook omitted this — runs weren't reproducible
    )

    # ----------------------------------------------------------- 2. data --
    records = _read_records(train_uri)
    log.info("Loaded %d training examples", len(records))

    def to_text(example: dict[str, Any]) -> dict[str, str]:
        messages = _as_messages(example)
        return {
            "text": tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        }

    ds = Dataset.from_list(records)
    dataset = ds.map(to_text, remove_columns=ds.column_names)
    log.info("Example rendered prompt:\n%s", dataset[0]["text"][:600])

    # ---------------------------------------------------------- 3. train --
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,      # `tokenizer=` is deprecated in recent TRL
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=grad_accum,
            warmup_steps=warmup_steps,
            num_train_epochs=epochs,
            max_steps=max_steps,         # -1 = disabled; any positive value WINS over epochs
            learning_rate=learning_rate,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=seed,
            output_dir=str(work / "checkpoints"),
            save_strategy="no",          # Vertex VMs are ephemeral; only the final save matters
            report_to=[],
        ),
    )

    if train_on_responses_only:
        trainer = _mask_prompt_tokens(trainer, tokenizer, base_model)

    stats = trainer.train()
    log.info("Training finished: %s", stats.metrics)

    # ----------------------------------------------------------- 4. save --
    local_model = work / "model"
    if local_model.exists():
        shutil.rmtree(local_model)

    model.save_pretrained_merged(str(local_model), tokenizer, save_method=save_method)
    _assert_deployable(local_model, save_method)

    if output_dir.startswith("gs://"):
        gcs_io.upload_dir(local_model, output_dir)
    else:
        shutil.copytree(local_model, output_dir, dirs_exist_ok=True)

    log.info("Model published to %s", output_dir)
    return output_dir


# ------------------------------------------------------------- helpers ----
def _read_records(uri: str) -> list[dict[str, Any]]:
    """Accepts .jsonl (one object per line) or .json (a single list)."""
    from src.data import gcs_io

    if uri.startswith("gs://"):
        if uri.endswith(".jsonl"):
            return list(gcs_io.read_jsonl(uri))
        local = gcs_io.download_file(uri, "/tmp/raw.json")
        return json.loads(local.read_text(encoding="utf-8"))

    text = Path(uri).read_text(encoding="utf-8")
    if uri.endswith(".jsonl"):
        return [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    return json.loads(text)


def _as_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    """Bridges both shapes: the notebook's {prompt, response} and the
    {messages: [...]} format used elsewhere in this project."""
    if "messages" in example:
        return example["messages"]

    response = example["response"]
    if not isinstance(response, str):
        # the notebook's people_data.json stores structured JSON responses
        response = json.dumps(response, ensure_ascii=False)

    messages = []
    if example.get("system"):
        messages.append({"role": "system", "content": example["system"]})
    messages += [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": response},
    ]
    return messages


def _mask_prompt_tokens(trainer, tokenizer, base_model: str):
    """Compute loss only on assistant turns. Training on the user's text too
    teaches the model to generate questions, which is rarely what you want."""
    from unsloth.chat_templates import train_on_responses_only

    name = base_model.lower()
    if "phi-3" in name or "phi3" in name:
        instruction_part, response_part = "<|user|>\n", "<|assistant|>\n"
    elif "llama-3" in name or "llama3" in name:
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n"
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    elif "qwen" in name:
        instruction_part, response_part = "<|im_start|>user\n", "<|im_start|>assistant\n"
    else:
        log.warning("Unknown chat template for %s — training on full sequence", base_model)
        return trainer

    return train_on_responses_only(
        trainer, instruction_part=instruction_part, response_part=response_part
    )


def _assert_deployable(path: Path, save_method: str) -> None:
    """Fail here, in the training job, rather than 20 minutes into a deploy."""
    if save_method == "lora":
        log.warning("save_method='lora' writes an adapter only — not directly deployable.")
        return

    files = {p.name for p in path.iterdir()}
    if not any(f.startswith("model") and f.endswith(".safetensors") for f in files) \
            and "pytorch_model.bin" not in files:
        raise RuntimeError(f"No weight files in {path}: {sorted(files)}")
    for required in ("config.json", "tokenizer_config.json"):
        if required not in files:
            raise RuntimeError(f"Missing {required} in {path}: {sorted(files)}")
    log.info("Artifact check passed: %s", sorted(files))


# ---------------------------------------------------------------- CLI -----
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-uri", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--base-model", default="unsloth/Phi-3-mini-4k-instruct-bnb-4bit")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--save-method", default="merged_16bit",
                   choices=["merged_16bit", "merged_4bit", "lora"])
    p.add_argument("--full-sequence-loss", action="store_true",
                   help="Disable response-only masking.")
    a = p.parse_args()

    finetune_with_unsloth(
        train_uri=a.train_uri,
        output_dir=a.output_dir,
        base_model=a.base_model,
        epochs=a.epochs,
        max_steps=a.max_steps,
        lora_r=a.lora_r,
        learning_rate=a.learning_rate,
        max_seq_length=a.max_seq_length,
        save_method=a.save_method,
        train_on_responses_only=not a.full_sequence_loss,
    )


if __name__ == "__main__":
    main()
