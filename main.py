import logging
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_from_disk
from constants import MODEL, DATASET_DIR, OUTPUT_DIR, ADAPTER_DIR

logging.basicConfig(level=logging.INFO)
logging.info("Running")
logging.info("Fine Tuning with LoRA")

# Define Constants
LORA_R          = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05
TARGET_MODULES  = ["c_attn"]
EPOCHS          = 3
BATCH_SIZE      = 4
LEARNING_RATE   = 2e-4
SAVE_STEPS      = 500
LOGGING_STEPS   = 100

# Load Base Model and Tokenizer
logging.info(f"Loading base model: {MODEL}")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(MODEL)


# Apply LoRA
lora_config = LoraConfig(
    task_type       = TaskType.CAUSAL_LM,
    r               = LORA_R,
    lora_alpha      = LORA_ALPHA,
    lora_dropout    = LORA_DROPOUT,
    target_modules  = TARGET_MODULES,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load dataset & tokenize
logging.info(f"Loading dataset from: {DATASET_DIR}")
dataset = load_from_disk(DATASET_DIR)
dataset.set_format("torch")
tokenized_dataset = dataset["train"]

# Train arguments
training_args = TrainingArguments(
    output_dir              = OUTPUT_DIR,
    num_train_epochs        = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    learning_rate           = LEARNING_RATE,
    save_steps              = SAVE_STEPS,
    logging_steps           = LOGGING_STEPS,
    fp16                    = False,
    use_mps_device          = True,
)

trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = tokenized_dataset,
    data_collator   = DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Train & save
logging.info("Starting training...")
trainer.train()

logging.info(f"Saving LoRA adapter to: {ADAPTER_DIR}")
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
logging.info("Done!")