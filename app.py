import math
import torch
import logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel
from flask import Flask, request, jsonify, render_template_string
from constants import AUTHOR_TAG_1, AUTHOR_TAG_2, AUTHOR_TAG_3, AUTHOR_TAG_4, MODEL, ADAPTER_DIR

# Define Constants
AUTHORS = [AUTHOR_TAG_1, AUTHOR_TAG_2, AUTHOR_TAG_3, AUTHOR_TAG_4]
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>StyleForge</title>
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; background: #f9f9f9; }
        h1 { color: #333; }
        label { font-weight: bold; display: block; margin-top: 16px; }
        input[type=text], select { width: 100%; padding: 8px; margin-top: 4px; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; }
        input[type=range] { width: 100%; margin-top: 4px; }
        button { margin-top: 20px; padding: 10px 24px; background: #4a90d9; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
        button:hover { background: #357abd; }
        #output { margin-top: 24px; background: white; padding: 16px; border-radius: 4px; border: 1px solid #ddd; white-space: pre-wrap; min-height: 100px; }
        #perplexity { margin-top: 8px; color: #666; font-size: 14px; }
        .slider-row { display: flex; justify-content: space-between; font-size: 13px; color: #888; }
    </style>
</head>
<body>
    <h1>✍️ StyleForge — Author Style Mimicry</h1>
    <p>Generate text in the style of classic authors using a LoRA fine-tuned GPT-2 model.</p>

    <label>Your Prompt</label>
    <input type="text" id="prompt" placeholder="Enter a sentence to continue..." />

    <label>Author Style</label>
    <select id="author">
        {% for a in authors %}
        <option value="{{ a }}">{{ a }}</option>
        {% endfor %}
    </select>

    <label>Max Tokens: <span id="tokens-val">150</span></label>
    <input type="range" id="max_tokens" min="50" max="300" value="150" step="10"
           oninput="document.getElementById('tokens-val').innerText = this.value" />
    <div class="slider-row"><span>50</span><span>300</span></div>

    <label>Temperature: <span id="temp-val">0.85</span></label>
    <input type="range" id="temperature" min="0.5" max="1.2" value="0.85" step="0.05"
           oninput="document.getElementById('temp-val').innerText = parseFloat(this.value).toFixed(2)" />
    <div class="slider-row"><span>0.5</span><span>1.2</span></div>

    <button onclick="generate()">Generate</button>

    <div id="output">Output will appear here...</div>
    <div id="perplexity"></div>

    <script>
        async function generate() {
            const btn = document.querySelector('button');
            btn.textContent = 'Generating...';
            btn.disabled = true;
            document.getElementById('output').textContent = 'Generating...';

            const response = await fetch('/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: document.getElementById('prompt').value,
                    author: document.getElementById('author').value,
                    max_tokens: document.getElementById('max_tokens').value,
                    temperature: document.getElementById('temperature').value,
                })
            });

            const data = await response.json();
            document.getElementById('output').textContent = data.text;
            document.getElementById('perplexity').textContent = 'Perplexity: ' + data.perplexity;
            btn.textContent = 'Generate';
            btn.disabled = false;
        }
    </script>
</body>
</html>
"""

# Setup
logging.basicConfig(level=logging.INFO)

# Load model & tokenizer
logging.info("Loading model...")
tokenizer = GPT2Tokenizer.from_pretrained(ADAPTER_DIR)
tokenizer.pad_token = tokenizer.eos_token

base_model = GPT2LMHeadModel.from_pretrained(MODEL)
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()
logging.info("Model ready!")


# Perplexity
def perplexity(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss
    return math.exp(loss.item())


# Generation
def generate(prompt, author, max_tokens, temperature):
    full_prompt = f"[AUTHOR: {author}] {prompt}"
    inputs = tokenizer(full_prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=0.92,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    ppl = perplexity(full_prompt)
    return generated_text, round(ppl, 2)

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string(HTML, authors=AUTHORS)

@app.route('/generate', methods=['POST'])
def generate_route():
    data = request.get_json()
    text, ppl = generate(
        data['prompt'],
        data['author'],
        data['max_tokens'],
        data['temperature'],
    )
    return jsonify({'text': text, 'perplexity': ppl})


if __name__ == '__main__':
    app.run(debug=False, port=7860)