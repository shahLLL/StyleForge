import os
import logging
from datasets import Dataset, DatasetDict
from transformers import GPT2Tokenizer
from constants import AUTHOR_TAG_1, AUTHOR_TAG_2, AUTHOR_TAG_3, AUTHOR_TAG_4, CHUNK_SIZE, PROCESSED_DIR, DATASET_DIR, MODEL

# Define Constants
TRAIN_SPLIT    = 0.9
AUTHOR_TAGS = [AUTHOR_TAG_1, AUTHOR_TAG_2, AUTHOR_TAG_3, AUTHOR_TAG_4]

# Setup
os.makedirs(DATASET_DIR, exist_ok=True)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL)
logging.basicConfig(level=logging.INFO)
tokenizer.pad_token = tokenizer.eos_token


# Load all Chunks
def load_all_chunks():
    """
    Read every author's processed .txt file and collect
    all tagged chunks into a single flat list.
    Each item in the list is one training example (one chunk).
    """

    all_chunks = []

    for author in AUTHOR_TAGS:
        filepath = os.path.join(PROCESSED_DIR, f"{author.lower()}_chunks.txt")

        if not os.path.exists(filepath):
            logging.info(f"  ⚠️  Missing file for {author}: {filepath} — skipping.")
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        logging.info(f"  📖 {author}: {len(lines):,} chunks loaded")
        all_chunks.extend(lines)

    logging.info(f"\n  Total chunks across all authors: {len(all_chunks):,}")
    return all_chunks


# Build Dataset
def build_dataset(chunks):
    """
    Convert the flat list of text chunks into a HuggingFace Dataset.
    The Dataset expects a dictionary where each key is a column name
    and each value is a list of entries for that column.
    We only need one column: 'text'.
    """

    data = {"text": chunks}

    dataset = Dataset.from_dict(data)
    logging.info(f"\n  ✅ Dataset created: {len(dataset)} rows")
    logging.info(f"  Columns: {dataset.column_names}")

    return dataset


# Tokenize
def tokenize_dataset(dataset):
    """
    Convert raw text strings into token IDs that the model can read.
    This is done in batches for efficiency.

    truncation=True  — cuts chunks that slightly exceed MAX_LENGTH
    padding=True     — pads shorter chunks to MAX_LENGTH
    max_length       — enforces the token limit

    The result adds two new columns to the dataset:
    - input_ids      — the token ID sequences
    - attention_mask — 1 for real tokens, 0 for padding
    """

    def tokenize_function(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=CHUNK_SIZE,
        )

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing dataset"
    )

    tokenized = tokenized.remove_columns(["text"])
    tokenized.set_format("torch")

    logging.info(f"\n  ✅ Tokenization complete")
    logging.info(f"  Columns after tokenization: {tokenized.column_names}")

    return tokenized


# Split Dataset into training and validation
def split_dataset(tokenized):
    """
    Split the tokenized dataset into training and validation sets.
    - Training set (90%): used to update model weights during fine-tuning
    - Validation set (10%): used to monitor performance and catch overfitting

    seed=42 ensures the split is reproducible — you'll get the same
    train/val split every time you run this script.
    """

    split = tokenized.train_test_split(
        test_size=1 - TRAIN_SPLIT,
        seed=42
    )

    dataset_dict = DatasetDict({
        "train":      split["train"],
        "validation": split["test"]
    })

    train_size = len(dataset_dict["train"])
    val_size   = len(dataset_dict["validation"])

    logging.info(f"\n  ✅ Split complete")
    logging.info(f"  Training examples:   {train_size:,}")
    logging.info(f"  Validation examples: {val_size:,}")

    return dataset_dict


# Save Dataset
def save_dataset(dataset_dict):
    """
    Save the final DatasetDict to disk in Arrow format.
    This makes it fast to reload in Step 3 without re-processing
    the raw text files every time.
    """

    dataset_dict.save_to_disk(DATASET_DIR)
    logging.info(f"\n  ✅ Dataset saved to: {DATASET_DIR}/")
    logging.info(f"  Reload later with: DatasetDict.load_from_disk(\"{DATASET_DIR}\")")


# Main Script
if __name__ == "__main__":
    logging.info("=" * 55)
    logging.info("  StyleForge — HuggingFace Dataset Creation")
    logging.info("=" * 55)

    # 1. Load all tagged chunks from processed .txt files
    logging.info("\n[1/4] Loading chunks...")
    chunks = load_all_chunks()

    # 2. Build the raw Dataset object
    logging.info("\n[2/4] Building dataset...")
    dataset = build_dataset(chunks)

    # 3. Tokenize the dataset
    logging.info("\n[3/4] Tokenizing...")
    tokenized = tokenize_dataset(dataset)

    # 4. Split into train / validation
    logging.info("\n[4/4] Splitting into train/validation...")
    dataset_dict = split_dataset(tokenized)

    # 5. Save to disk
    save_dataset(dataset_dict)

    # ── Final summary ──────────────────────────────────────────
    logging.info("=" * 55)
    logging.info("All Done!")

    # Quick sanity check — peek at one training example
    logging.info("\n📝 Sample training example (first row):\n")
    sample = dataset_dict["train"][0]
    logging.info(f"  input_ids shape:      {sample['input_ids'].shape}")
    logging.info(f"  attention_mask shape: {sample['attention_mask'].shape}")
    logging.info(f"  First 10 token IDs:   {sample['input_ids'][:10].tolist()}")
    decoded = tokenizer.decode(sample['input_ids'][:30], skip_special_tokens=True)
    logging.info(f"  Decoded preview:      {decoded}...")