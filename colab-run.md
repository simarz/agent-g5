# Running this project in Google Colab

Local training isn't viable (AMD GPU on Windows can't run the LoRA stack), so we
run on a free Colab **T4 GPU**. This is the repeatable setup — do it at the start
of each Colab session.

## 0. Push your latest code first (from your local machine)

Colab pulls from GitHub, so commit and push before each session:

```bash
git add -A
git commit -m "your message"
git push origin main
```

## 1. New notebook + enable the GPU

1. Open <https://colab.research.google.com> → **New notebook**.
2. **Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save.**

## 2. Confirm the GPU is live

```python
import torch
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

Expect `True` and `Tesla T4`. If it prints `False`, redo step 1.

## 3. Clone the repo

```python
!git clone https://github.com/simarz/agent-g5.git
%cd agent-g5
```

If the repo is **private**, use a GitHub personal access token:

```python
!git clone https://<YOUR_TOKEN>@github.com/simarz/agent-g5.git
```

(Better: store the token via Colab **Secrets** 🔑 rather than pasting it inline.)

## 4. Install dependencies

Colab already ships a CUDA-enabled `torch` — **do not reinstall torch**. Install
only the extras:

```python
!pip install -q -U transformers datasets accelerate
# training days also need: peft bitsandbytes
```

## 5. Run

```python
!python model.py
```

First run downloads the model weights (~8 GB for Qwen3-4B) into the session —
a few minutes, one time per session.

## 6. Pull updates during a session

After you push new commits from local, refresh Colab's copy:

```python
%cd /content/agent-g5
!git pull origin main
```

---

## Things to know about Colab

- **Sessions are ephemeral.** When the runtime disconnects, cloned files *and*
  downloaded model weights are wiped. Re-run steps 1–4 next time.
- **Avoid re-downloading the model every session:** mount Google Drive and point
  the Hugging Face cache at it:

  ```python
  from google.colab import drive; drive.mount('/content/drive')
  import os; os.environ['HF_HOME'] = '/content/drive/MyDrive/hf_cache'
  ```

- **Save your outputs** (adapters, logs, result files) to Drive or push them to
  GitHub before the session ends, or you'll lose them.
- **Free-tier limits:** GPU time is capped and idle sessions disconnect. Keep the
  tab active during long runs.

## Recommended workflow

Edit locally in VS Code → commit & push → `git pull` in Colab → run on the GPU.
You get your editor for writing and Colab only for the GPU.
