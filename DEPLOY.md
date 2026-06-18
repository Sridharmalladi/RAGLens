# Deploying RAGLens to HuggingFace Spaces

Follow these steps in order. Total time: ~30 minutes (most of it is the Colab notebook running).

---

## Step 1 — Build the Corpus (Google Colab, one-time)

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload `build_corpus.ipynb` from this repo (File → Upload notebook)
3. Run all cells top to bottom
   - It downloads 50 arXiv papers (~300 MB total)
   - Embeds them with BGE-small (~5 min on CPU, ~2 min on T4)
   - Saves two files: `corpus/index.faiss` and `corpus/processed/chunks.json`
4. The last cell downloads both files to your computer automatically
5. Put them in the correct paths in this repo:
   ```
   corpus/index.faiss
   corpus/processed/chunks.json
   ```

---

## Step 2 — Get a Free Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (free — no credit card required)
3. Create an API key
4. Copy it — you'll need it in Step 4

---

## Step 3 — Create a HuggingFace Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Fill in:
   - **Space name:** `raglens` (or any name you want)
   - **License:** Apache 2.0
   - **SDK:** Gradio
   - **Hardware:** CPU Basic ← free tier, no GPU needed
3. Click **Create Space**

---

## Step 4 — Add Your API Key as a Secret

1. In your new Space, go to **Settings → Variables and secrets**
2. Click **New secret**
3. Name: `GROQ_API_KEY`
4. Value: paste your Groq API key
5. Click **Save**

---

## Step 5 — Push the Code

In your terminal:

```bash
# Clone the HuggingFace Space repo (replace YOUR_USERNAME and YOUR_SPACE_NAME)
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
cd YOUR_SPACE_NAME

# Copy all RAGLens files into it
cp -r /path/to/RAGLens/. .

# Make sure the corpus files are included
ls corpus/index.faiss          # must exist
ls corpus/processed/chunks.json  # must exist

# Commit and push
git add .
git commit -m "Initial RAGLens deployment"
git push
```

The Space will automatically build and launch. First build takes ~5-10 minutes as it installs dependencies.

---

## Step 6 — Verify It's Working

1. Open your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`
2. Click the **🧪 Try It Yourself** tab
3. Click one of the suggested query buttons
4. Click **▶ Run all 4 configs**
5. You should see all 4 result cards populate within ~60-90 seconds

If you see "Corpus not built yet" — the corpus files weren't committed. Re-check Step 1 and Step 5.

---

## Step 7 — Monitoring (automatic)

The scheduler starts automatically when the Space boots. It runs the 3 fixed queries every hour and writes scores to SQLite.

- First monitoring data appears ~1 hour after deployment
- Check the **📊 Live Monitoring** tab after that

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Corpus not built yet" | Commit `corpus/index.faiss` and `corpus/processed/chunks.json` |
| Scores show "Scoring unavailable" | Check that `GROQ_API_KEY` secret is set in Space settings |
| Models grayed out (configs 3 & 4) | QLoRA adapter not committed — train it first or accept base model fallback |
| Space crashes on boot | Check Space logs for OOM — try a smaller model first (Qwen2.5-1.5B is default) |
| Generation very slow | CPU inference on large models is slow — Qwen2.5-1.5B is the fastest option |

---

## Optional: Custom Domain

1. Buy a domain (e.g., `raglens.dev` ~$12/year)
2. In HuggingFace Space settings → **Custom domains** → add your domain
3. Add a CNAME record in your DNS provider pointing to your Space URL
4. Resume link becomes `raglens.dev` instead of the long HF URL
