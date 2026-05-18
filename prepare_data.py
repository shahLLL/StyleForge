import os
import re
import logging
from transformers import GPT2Tokenizer
from constants import AUTHOR_TAG_1, AUTHOR_TAG_2, AUTHOR_TAG_3, AUTHOR_TAG_4, CHUNK_SIZE, RAW_DIR, PROCESSED_DIR, MODEL

# Define Constants
MIN_CHUNK     = 50
AUTHOR_FILES = {
    "acd1.txt": AUTHOR_TAG_1,
    "acd2.txt": AUTHOR_TAG_1,
    "acd3.txt": AUTHOR_TAG_1,
    "cd1.txt": AUTHOR_TAG_2,
    "cd2.txt": AUTHOR_TAG_2,
    "hm1.txt": AUTHOR_TAG_3,
    "hm2.txt": AUTHOR_TAG_3,
    "sh1.txt": AUTHOR_TAG_4,
    "sh2.txt": AUTHOR_TAG_4,
    "sh3.txt": AUTHOR_TAG_4,
    "sh4.txt": AUTHOR_TAG_4
}

# Setup
os.makedirs(PROCESSED_DIR, exist_ok=True)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL)
logging.basicConfig(level=logging.INFO)


# Clean raw text
def clean_gutenberg_text(text):
    """
    Remove Project Gutenberg headers, footers, and other boilerplate.
    Gutenberg files typically have a long preamble before the actual
    text begins, and a footer after it ends. This strips both out.
    """
    start_pattern = r"\*\*\* ?START OF TH(E|IS) PROJECT GUTENBERG.*?\*\*\*"
    match = re.search(start_pattern, text, re.IGNORECASE)
    if match:
        text = text[match.end():]

    end_pattern = r"\*\*\* ?END OF TH(E|IS) PROJECT GUTENBERG.*?\*\*\*"
    match = re.search(end_pattern, text, re.IGNORECASE)
    if match:
        text = text[:match.start()]

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


# Chunk Text
def chunk_text(text, chunk_size, min_chunk):
    """
    Split text into chunks of approximately `chunk_size` tokens.
    We tokenize the full text first, then slice it into even pieces.
    This ensures consistent chunk sizes regardless of word length.
    """

    token_ids = tokenizer.encode(text)
    logging.info(f"    Total tokens in text: {len(token_ids):,}")

    chunks = []
    for i in range(0, len(token_ids), chunk_size):
        chunk_ids = token_ids[i : i + chunk_size]
        if len(chunk_ids) < min_chunk:
            continue
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)

    return chunks


# Tag Files to Authors
def process_author(filename, author_tag):
    """
    Full pipeline for one author:
    1. Read raw file
    2. Clean Gutenberg boilerplate
    3. Chunk into token segments
    4. Prepend author tag to each chunk
    5. Save to processed directory
    """

    filepath = os.path.join(RAW_DIR, filename)
    if not os.path.exists(filepath):
        logging.info(f"  ⚠️  File not found: {filepath} — skipping.")
        return 0

    logging.info(f"\n📖 Processing: {filename} → [{author_tag}]")

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()
    logging.info(f"    Raw file size: {len(raw_text):,} characters")

    cleaned = clean_gutenberg_text(raw_text)
    logging.info(f"    After cleaning: {len(cleaned):,} characters")

    chunks = chunk_text(cleaned, CHUNK_SIZE, MIN_CHUNK)
    logging.info(f"    Chunks created: {len(chunks):,}")

    output_path = os.path.join(PROCESSED_DIR, f"{author_tag.lower()}_chunks.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            tagged_line = f"[AUTHOR: {author_tag}] {chunk.strip()}\n"
            f.write(tagged_line)

    logging.info(f"    ✅ Saved to: {output_path}")
    return len(chunks)

# Main Script
if __name__ == "__main__":
    logging.info("=" * 55)
    logging.info("  StyleForge — Dataset Preparation")
    logging.info("=" * 55)

    total_chunks = 0

    # Process each author file defined in AUTHOR_FILES
    for filename, author_tag in AUTHOR_FILES.items():
        count = process_author(filename, author_tag)
        total_chunks += count

    logging.info("\n" + "=" * 55)
    logging.info(f"  Done! Total chunks across all authors: {total_chunks:,}")
    logging.info(f"  Processed files saved to: {PROCESSED_DIR}/")
    logging.info("=" * 55)

    # Quick sanity check — show a sample chunk from each author
    logging.info("\n📝 Sample output (first chunk from each author):\n")
    for filename, author_tag in AUTHOR_FILES.items():
        output_path = os.path.join(PROCESSED_DIR, f"{author_tag.lower()}_chunks.txt")
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                first_line = f.readline()
            logging.info(f"[{author_tag}]: {first_line[:120]}...")
            logging.info("Done")