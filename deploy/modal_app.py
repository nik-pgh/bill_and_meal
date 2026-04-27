"""Modal deployment: serves the Bill & Meal model behind a simple web page.

Deploy with:
    modal deploy deploy/modal_app.py

Required Modal secret named "huggingface-secret" with HF_TOKEN key.
"""

import modal

app = modal.App("bill-and-meal")

BASE_MODEL = "google/gemma-4-E2B-it"
ADAPTER_REPO = "nik-pgh/bill-and-meal-gemma4-lora"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")  # required for `pip install git+https://...` URLs
    .pip_install(
        "torch",
        "torchvision",
        "transformers @ git+https://github.com/huggingface/transformers.git",
        "peft @ git+https://github.com/huggingface/peft.git",
        "accelerate",
        "bitsandbytes",
        "Pillow",
        "fastapi[standard]",
        "python-multipart",
    )
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.cls(
    image=image,
    gpu="A10G",
    secrets=[hf_secret],
    volumes={"/cache": hf_cache},
    scaledown_window=300,  # keep container warm 5 min after last request
    timeout=600,
)
class Model:
    @modal.enter()
    def setup(self) -> None:
        import os

        os.environ["HF_HOME"] = "/cache/huggingface"

        import torch
        from peft import PeftModel
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            BitsAndBytesConfig,
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.processor = AutoProcessor.from_pretrained(ADAPTER_REPO)
        base = AutoModelForImageTextToText.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.model = PeftModel.from_pretrained(base, ADAPTER_REPO)
        self.model.eval()

    @modal.method()
    def generate(self, image_bytes: bytes) -> str:
        import io

        import torch
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "What recipes can I make from this grocery receipt?"},
            ],
        }]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=600,
                do_sample=False,  # greedy: deterministic, less hallucination
            )

        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(new_tokens, skip_special_tokens=True)


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bill &amp; Meal — Receipt to Recipe</title>
<style>
  :root {
    --bg: #fafaf7;
    --card: #ffffff;
    --text: #1a1a1a;
    --muted: #6b7280;
    --accent: #ea580c;
    --accent-soft: #fff7ed;
    --border: #e5e5e5;
    --error: #dc2626;
    color-scheme: light dark;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0a0a0a;
      --card: #161616;
      --text: #f5f5f5;
      --muted: #9ca3af;
      --accent: #fb923c;
      --accent-soft: #2a1505;
      --border: #2a2a2a;
    }
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body {
    font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI",
                 "Helvetica Neue", Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    line-height: 1.6;
    padding: 24px 20px 80px;
  }
  .container { max-width: 680px; margin: 0 auto; }

  header { text-align: center; padding: 32px 0 24px; }
  .logo {
    font-size: 56px;
    line-height: 1;
    display: inline-block;
    margin-bottom: 8px;
  }
  h1 {
    font-size: 32px;
    margin: 0 0 6px;
    font-weight: 700;
    letter-spacing: -0.02em;
  }
  .tagline { color: var(--muted); font-size: 15px; margin: 0; }

  .card {
    background: var(--card);
    border-radius: 20px;
    padding: 28px;
    border: 1px solid var(--border);
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }

  .dropzone {
    display: block;
    border: 2px dashed var(--border);
    border-radius: 14px;
    padding: 40px 20px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s ease;
    user-select: none;
  }
  .dropzone:hover, .dropzone.dragover {
    border-color: var(--accent);
    background: var(--accent-soft);
  }
  .dropzone input { display: none; }
  .dz-icon {
    font-size: 40px;
    margin-bottom: 12px;
    display: block;
  }
  .dz-text { font-weight: 500; margin: 4px 0; }
  .dz-hint { color: var(--muted); font-size: 13px; margin: 4px 0 0; }

  #preview { display: none; }
  #preview-img {
    max-width: 100%;
    max-height: 280px;
    border-radius: 10px;
    display: block;
    margin: 0 auto;
  }

  button.primary {
    background: var(--accent);
    color: #fff;
    border: none;
    padding: 14px 28px;
    border-radius: 10px;
    font-size: 16px;
    font-weight: 600;
    cursor: pointer;
    width: 100%;
    margin-top: 20px;
    transition: transform 0.1s, opacity 0.2s;
  }
  button.primary:hover:not(:disabled) {
    transform: translateY(-1px);
    opacity: 0.95;
  }
  button.primary:disabled { opacity: 0.4; cursor: not-allowed; }

  /* Loading state */
  .loading {
    display: none;
    text-align: center;
    padding: 32px 16px;
  }
  .pot-wrap {
    position: relative;
    display: inline-block;
    width: 120px;
    height: 100px;
  }
  .pot {
    font-size: 72px;
    display: inline-block;
    animation: wobble 1.2s ease-in-out infinite;
    transform-origin: 50% 90%;
  }
  @keyframes wobble {
    0%, 100% { transform: rotate(-4deg); }
    50% { transform: rotate(4deg); }
  }
  .bubble {
    position: absolute;
    background: var(--accent);
    border-radius: 50%;
    opacity: 0;
    animation: rise 2s ease-in infinite;
  }
  .bubble:nth-child(2) {
    width: 10px; height: 10px;
    left: 30%; bottom: 65px;
    animation-delay: 0s;
  }
  .bubble:nth-child(3) {
    width: 8px; height: 8px;
    left: 55%; bottom: 70px;
    animation-delay: 0.6s;
  }
  .bubble:nth-child(4) {
    width: 12px; height: 12px;
    left: 45%; bottom: 60px;
    animation-delay: 1.2s;
  }
  @keyframes rise {
    0%   { opacity: 0; transform: translateY(0) scale(0.4); }
    20%  { opacity: 0.8; }
    100% { opacity: 0; transform: translateY(-80px) scale(1.2); }
  }

  .stage {
    margin-top: 16px;
    color: var(--text);
    font-size: 16px;
    font-weight: 500;
    min-height: 24px;
    transition: opacity 0.3s;
  }
  .substage {
    color: var(--muted);
    font-size: 13px;
    margin-top: 4px;
    min-height: 18px;
  }

  .progress {
    margin-top: 24px;
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
  }
  .progress-bar {
    height: 100%;
    width: 0;
    background: linear-gradient(90deg, var(--accent), #fb923c);
    border-radius: 3px;
    transition: width 0.4s ease-out;
  }

  /* Result */
  .result {
    display: none;
    margin-top: 24px;
    padding: 24px;
    background: var(--accent-soft);
    border-radius: 14px;
    border: 1px solid var(--border);
    line-height: 1.7;
  }
  .result h2 { font-size: 20px; margin: 20px 0 8px; color: var(--accent); }
  .result h3 { font-size: 17px; margin: 16px 0 6px; }
  .result strong { color: var(--accent); font-weight: 600; }
  .result ul { padding-left: 20px; margin: 8px 0; }
  .result.error {
    background: rgba(220, 38, 38, 0.08);
    color: var(--error);
  }

  footer {
    text-align: center;
    color: var(--muted);
    font-size: 12px;
    margin-top: 32px;
  }
  footer a { color: var(--muted); }
</style>
</head>
<body>
  <div class="container">
    <header>
      <div class="logo">🧾</div>
      <h1>Bill &amp; Meal</h1>
      <p class="tagline">Turn your grocery receipt into recipe ideas.</p>
    </header>

    <div class="card">
      <label class="dropzone" id="dropzone">
        <input type="file" id="file" accept="image/*">
        <div id="placeholder">
          <span class="dz-icon">📸</span>
          <p class="dz-text">Drop your receipt here or click to upload</p>
          <p class="dz-hint">JPG, PNG · max 10 MB</p>
        </div>
        <div id="preview">
          <img id="preview-img" alt="Receipt preview">
          <p class="dz-hint" style="margin-top:12px;">Click to change image</p>
        </div>
      </label>

      <button id="submit" class="primary" disabled>Get Recipe Ideas</button>

      <div class="loading" id="loading">
        <div class="pot-wrap">
          <span class="pot">🍲</span>
          <span class="bubble"></span>
          <span class="bubble"></span>
          <span class="bubble"></span>
        </div>
        <div class="stage" id="stage">Warming up the kitchen</div>
        <div class="substage" id="substage">First request takes ~60 seconds</div>
        <div class="progress"><div class="progress-bar" id="progress-bar"></div></div>
      </div>

      <div class="result" id="result"></div>
    </div>

    <footer>
      Powered by Gemma 4 fine-tuned with QLoRA · ·
      <a href="https://github.com/nik-pgh/bill_and_meal" target="_blank">GitHub</a>
    </footer>
  </div>

<script>
const STAGES = [
  { msg: "Warming up the kitchen",      sub: "Loading the chef (cold start ~30s)" },
  { msg: "Reading your receipt",        sub: "Identifying ingredients from the image" },
  { msg: "Brainstorming recipes",       sub: "Combining ingredients into dishes" },
  { msg: "Writing instructions",        sub: "Step by step, almost there" },
  { msg: "Adding the final seasoning",  sub: "Polishing the suggestions" },
];

const fileInput = document.getElementById('file');
const dropzone = document.getElementById('dropzone');
const placeholder = document.getElementById('placeholder');
const preview = document.getElementById('preview');
const previewImg = document.getElementById('preview-img');
const submit = document.getElementById('submit');
const loading = document.getElementById('loading');
const result = document.getElementById('result');
const stageEl = document.getElementById('stage');
const substageEl = document.getElementById('substage');
const progressBar = document.getElementById('progress-bar');

['dragenter', 'dragover'].forEach(evt => {
  dropzone.addEventListener(evt, e => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });
});
['dragleave', 'drop'].forEach(evt => {
  dropzone.addEventListener(evt, e => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
  });
});
dropzone.addEventListener('drop', e => {
  if (e.dataTransfer.files.length) {
    fileInput.files = e.dataTransfer.files;
    handleFile();
  }
});
fileInput.addEventListener('change', handleFile);

function handleFile() {
  const f = fileInput.files[0];
  if (!f) return;
  if (f.size > 10 * 1024 * 1024) {
    alert('Image must be under 10 MB');
    return;
  }
  previewImg.src = URL.createObjectURL(f);
  placeholder.style.display = 'none';
  preview.style.display = 'block';
  submit.disabled = false;
  result.style.display = 'none';
  result.classList.remove('error');
}

let stageInterval, progressInterval;

function startLoadingAnimation() {
  let stageIdx = 0;
  let elapsed = 0;
  const totalEstimate = 60; // seconds

  const setStage = () => {
    const s = STAGES[stageIdx % STAGES.length];
    stageEl.style.opacity = 0;
    setTimeout(() => {
      stageEl.textContent = s.msg;
      substageEl.textContent = s.sub;
      stageEl.style.opacity = 1;
    }, 200);
    stageIdx++;
  };

  setStage();
  stageInterval = setInterval(setStage, 7000);

  progressInterval = setInterval(() => {
    elapsed += 0.4;
    // Asymptotic progress: never quite reach 100%
    const pct = (1 - Math.exp(-elapsed / totalEstimate * 2)) * 95;
    progressBar.style.width = pct + '%';
  }, 400);
}

function stopLoadingAnimation() {
  clearInterval(stageInterval);
  clearInterval(progressInterval);
  progressBar.style.width = '100%';
  setTimeout(() => { progressBar.style.width = '0%'; }, 500);
}

function renderMarkdown(text) {
  // Minimal markdown: escape HTML, then handle bold, headings, list bullets
  const esc = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  return esc
    .replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h2>$1</h2>')
    .replace(/^\\* (.+)$/gm, '<li>$1</li>')
    .replace(/(<li>.*?<\\/li>)(\\n<li>)/g, '$1$2')
    .replace(/(<li>.*?<\\/li>)+/gs, m => '<ul>' + m + '</ul>')
    .replace(/\\n\\n/g, '<br><br>')
    .replace(/\\n/g, '<br>');
}

submit.addEventListener('click', async () => {
  submit.disabled = true;
  submit.textContent = 'Cooking…';
  loading.style.display = 'block';
  result.style.display = 'none';
  result.classList.remove('error');
  startLoadingAnimation();

  const fd = new FormData();
  fd.append('file', fileInput.files[0]);

  try {
    const res = await fetch('/predict', { method: 'POST', body: fd });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    result.innerHTML = renderMarkdown(data.recipes);
    result.style.display = 'block';
  } catch (err) {
    result.classList.add('error');
    result.innerHTML = '❌ ' + err.message + ' — try again in a moment.';
    result.style.display = 'block';
  } finally {
    stopLoadingAnimation();
    loading.style.display = 'none';
    submit.disabled = false;
    submit.textContent = 'Get Recipe Ideas';
  }
});
</script>
</body>
</html>
"""


@app.function(image=image, timeout=600)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, File, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse

    web_app = FastAPI()

    @web_app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return INDEX_HTML

    @web_app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        image_bytes = await file.read()
        recipes = Model().generate.remote(image_bytes)
        return JSONResponse({"recipes": recipes})

    return web_app
