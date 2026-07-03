import argparse
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

def main():
    set_seed(42) # Ensure reproducibility

    # -------------------------------------------------------------------------
    # 1. Argument Parsing
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-3B-Instruct using QLoRA for Vedaz")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the JSONL dataset (e.g., train.jsonl)")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save the LoRA adapter")
    args = parser.parse_args()

    model_id = "Qwen/Qwen2.5-3B-Instruct"

    print(f"Loading dataset from: {args.dataset}")
    # Load the local JSONL dataset
    dataset = load_dataset("json", data_files={"train": args.dataset})["train"]
    
    # Validate that every dataset record contains a non-empty "messages" list
    dataset = dataset.filter(lambda x: "messages" in x and isinstance(x["messages"], list) and len(x["messages"]) > 0)

    # -------------------------------------------------------------------------
    # 2. Tokenizer & Chat Template Application
    # -------------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Qwen models usually don't have a pad token defined by default, so we set it to eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Crucial for causal language modeling training to avoid masking issues
    tokenizer.padding_side = "right"

    # How the chat template is applied:
    # Our dataset contains a 'messages' list (system, user, assistant).
    # We use the tokenizer's built-in `apply_chat_template` to convert this standard
    # conversational JSON array into Qwen's specific ChatML string format.
    # This ensures the model sees the exact same prompt structure during training
    # that it was pre-trained to understand.
    def apply_chat_template(example):
        example["text"] = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        return example

    dataset = dataset.map(apply_chat_template, desc="Applying chat template")

    # -------------------------------------------------------------------------
    # 3. Model Loading & 4-bit Quantization (BitsAndBytes)
    # -------------------------------------------------------------------------
    # Why QLoRA is used:
    # Full precision (16-bit/32-bit) training of a 3B parameter model requires massive VRAM 
    # (often >16GB just for optimizer states and gradients).
    # QLoRA (Quantized LoRA) uses BitsAndBytes to load the base model weights in 4-bit precision,
    # drastically reducing the memory footprint (to ~3-4GB for a 3B model), making it possible
    # to train on consumer hardware (like an RTX 3060/3090 or T4 GPU).
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",       # NormalFloat4 - optimal data type for weights
        bnb_4bit_compute_dtype=torch.float16, # Compute gradients in fp16
        bnb_4bit_use_double_quant=True   # Further reduces memory by quantizing the quantization constants
    )

    print(f"Loading base model {model_id} in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto", # Accelerate will automatically dispatch layers to available GPUs
        trust_remote_code=True
    )

    # Prepare model for parameter-efficient fine-tuning (freezes base weights, enables gradient checkpointing)
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False # Disable cache to prevent warnings during gradient checkpointing

    # -------------------------------------------------------------------------
    # 4. LoRA Configuration (PEFT)
    # -------------------------------------------------------------------------
    # Why LoRA is chosen instead of full fine-tuning:
    # Full fine-tuning updates all 3 billion parameters, which is computationally expensive,
    # prone to catastrophic forgetting, and produces massive checkpoint files.
    # LoRA (Low-Rank Adaptation) freezes the base model and injects small, trainable
    # low-rank matrices into specific attention layers (q_proj, v_proj, etc.). 
    # This means we only train < 1% of the total parameters, making it fast, stable, and 
    # resulting in a tiny, portable adapter file (usually < 100MB) that can be merged later.
    lora_config = LoraConfig(
        r=16, # Rank of the update matrices
        lora_alpha=32, # Scaling factor
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Target Qwen's attention and MLP layers
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # -------------------------------------------------------------------------
    # 5. Training Setup (SFTTrainer)
    # -------------------------------------------------------------------------
    # Why SFTTrainer is used:
    # SFTTrainer (from the TRL library - Transformer Reinforcement Learning) is a high-level wrapper
    # around Hugging Face's standard Trainer. It simplifies Supervised Fine-Tuning (SFT) by natively
    # handling PEFT integration, dataset formatting (mapping 'text' column), max sequence length truncation, 
    # and packing, avoiding hundreds of lines of boilerplate PyTorch code.
    
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        learning_rate=2e-4,
        fp16=True, # Use mixed precision (consider bf16=True if using Ampere+ GPU)
        logging_strategy="steps",
        logging_steps=5,
        max_seq_length=1024, # Maximum context length to train on
        save_strategy="epoch",
        save_total_limit=2, # Keep only the 2 most recent checkpoints
        optim="paged_adamw_8bit", # Memory efficient optimizer
        dataset_text_field="text", # Tell SFTTrainer to use our formatted text column
        gradient_checkpointing=True, # Explicitly enable gradient checkpointing for VRAM savings
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=lora_config,
        tokenizer=tokenizer,
    )

    # -------------------------------------------------------------------------
    # 6. Execute Training and Save
    # -------------------------------------------------------------------------
    print("Starting fine-tuning...")
    trainer.train()

    print(f"Training complete. Saving adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done!")

if __name__ == "__main__":
    main()
