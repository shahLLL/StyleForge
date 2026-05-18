import math
import torch
import logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel
from constants import AUTHOR_TAG_1, AUTHOR_TAG_2, AUTHOR_TAG_3, AUTHOR_TAG_4, ADAPTER_DIR, MODEL

# Define Constants
AUTHORS = [AUTHOR_TAG_1, AUTHOR_TAG_2, AUTHOR_TAG_3, AUTHOR_TAG_4]
STYLE_PROMPT = "The sky looked so beautiful, "

# Setup
logging.basicConfig(level=logging.INFO)

# Load model & tokenizer
logging.info("Loading model...")
tokenizer = GPT2Tokenizer.from_pretrained(ADAPTER_DIR)
tokenizer.pad_token = tokenizer.eos_token
base_model = GPT2LMHeadModel.from_pretrained(MODEL)
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()

# Perplexity
def perplexity(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss
    return math.exp(loss.item())

# Generation
def generate(author, prompt, max_new_tokens=150):
    full_prompt = f"[AUTHOR: {author}] {prompt}"
    inputs = tokenizer(full_prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.85,
            top_p=0.92,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Style consistency check
logging.info("\n" + "="*60)
logging.info("STYLEFORGE EVALUATION")
logging.info("="*60)

# Main Evaluation Function
for author in AUTHORS:
    logging.info(f"\n── Author: {author} ──")

    # Perplexity on a tagged sample
    sample = f"[AUTHOR: {author}] {STYLE_PROMPT}"
    ppl = perplexity(sample)
    logging.info(f"Perplexity: {ppl:.2f}")

    # Generated output
    logging.info("Generated:")
    output = generate(author, STYLE_PROMPT)
    logging.info(output)