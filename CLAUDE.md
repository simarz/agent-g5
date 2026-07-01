# Handoff / operating contract

Read at the start of every session. The top sections are the operating contract
(apply every time). "Background" at the bottom is reference you can trim once
internalized.

---

## Operating contract #0 — Mentor, NOT implementer  (highest priority)

**This is a long-term learning project. I (the user) build it myself, day by day.**
Do not hand me finished implementations — that defeats the purpose.

- **I write the code.** You guide, review, explain, and unblock.
- **Do not write full features, modules, or "the whole thing"** even if it seems
  helpful or fast. If tempted, stop and ask first.
- When I'm stuck, help **narrowly**: one function, one error, one concept — not
  the surrounding solution.
- Explaining concepts, sketching structure, and reviewing my code is welcome.
- Boilerplate/config (e.g. `pyproject.toml`, install commands) is fine to help
  with; ask if unsure whether something crosses into "the project itself."
- Prefer **guidance, checklists, and pointers** over code. I have my own weekly
  roadmap — follow my lead on scope; don't jump ahead.
- **When in doubt: ask whether I want guidance or code before writing code.**

---

## What this project is

A from-scratch **recursive self-improvement engine**: takes a small, pre-trained
open-weight LM and bootstraps its own reasoning by repeatedly solving verifiable
problems, keeping only attempts that pass an automatic checker, and fine-tuning
on those. Each loop yields a better model that generates better training data, so
capability compounds across iterations.

Academic names: **STaR** (Self-Taught Reasoner) and **expert iteration**.
Starting task: grade-school math (**GSM8K**) — correctness is trivially checkable.

**Final deliverable = the working loop *plus the evidence*:** held-out accuracy
climbing across iterations, ablations showing which techniques drove the gains,
and a small model that measurably out-reasons the starting one.

This is deliberately an **ML-majority** project. The only place weights change is
the fine-tuning step; generate/verify/filter just decide *what* to teach. The
difficulty is in the training dynamics; data is self-generated (minimal plumbing).

---

## Tech stack

- **Language:** Python (non-negotiable — the fine-tuning ecosystem is Python-first).
- **Core libraries:** `transformers`, `peft` (LoRA), `datasets`, `accelerate`, `torch`.
- **Fine-tuning method:** LoRA adapters — **don't merge**; keep updating the
  adapter checkpoint across iterations.
- MLX (`mlx-lm`) is an Apple-Silicon option in the handoff but **N/A for us**
  (see environment below).

## Our environment / infra

- **Training runs on a cloud NVIDIA GPU (Colab/Kaggle).** Local machine is an
  **AMD RX 6700 XT on Windows**, which *cannot* run the LoRA training stack
  (no CUDA; ROCm is Linux-only and this card is unofficial). Local GPU is
  inference-only at best.
- This is consistent with the "weights not GGUF" rule below: fine-tuning is the
  whole point, so Ollama/GGUF is out and a real training GPU is required.
- Local Python: global **Python 3.11** (`py -3.11`), NOT a virtualenv. Avoid the
  `hermes-agent` venv that bare `python` sometimes resolves to.
- On Colab: do **not** reinstall `torch` (it ships a CUDA build); install only
  the missing extras.

---

## Model choice

**Primary pick: `Qwen3-4B` (Instruct)**, or `Qwen3.5-4B` if using the newer
generation. Confirm the exact HF repo id on the model card before pulling.

Why this model for *this* project:
- **Apache 2.0 license** — fine-tune, checkpoint, share adapters freely.
- **Ungated** — no HF token/click-through; `from_pretrained` just works.
- **Right capability band** — solves a meaningful fraction of GSM8K, with headroom.
- **Tooling** — first-class in transformers, peft, MLX.

Hardware tiers:
- **Comfortable (good GPU / M-series 24GB+):** `Qwen3-4B`.
- **Tight (8–16GB):** `Qwen3-1.7B`, or `Phi-4-mini` (3.8B, MIT), or load the 4B
  in 4-bit via `bitsandbytes`.

### CRITICAL model rule
**Do NOT use reasoning-specialized / math-distilled models** (DeepSeek-R1
distills, Phi-4-reasoning, etc.). They already near-max GSM8K, which kills the
project — nothing to elicit, no improvement curve. We want a *general* small
model that's right **sometimes**: target a raw GSM8K solve rate of ~**40–70%**.

### Weights, not GGUF
Download standard **safetensors** from the HF repo. Do **not** use `ollama pull`
/ GGUF — those are inference-only and can't be LoRA-fine-tuned, and fine-tuning
is the entire point.

---

## Working conventions

- **One self-contained improvement per commit. Commit daily.** Every commit
  either adds a capability or moves a measured number.
- **Build the eval harness early.** Easy to fool yourself here (the model grades
  its own homework). A trusted held-out accuracy metric separates ML from
  self-delusion.
- **Keep loops fast.** A full generate→filter→train cycle in minutes beats an
  hour-long one. Run long generation/training in the background; the *commit* is
  the code change, not the wait.
- **The verifier is sacred.** It's the signal source for the whole project.
  Over-test it.

---

## Six-week plan (weekly goals)

- [ ] **Week 1 — The verifier.** Hand a problem to the base model, get an answer,
  automatically decide if it's right.
- [ ] **Week 2 — The data engine.** Point it at a batch of problems; get back a
  clean dataset of self-generated, verified-correct traces. Track baseline solve rate.
- [ ] **Week 3 — Close the loop.** One full cycle: generate → filter → LoRA
  fine-tune → improved model generates the next round. Milestone: recursion closes.
- [ ] **Week 4 — Prove it's real.** Trusted held-out accuracy chart per iteration.
  Add an overfitting detector (train solve-rate up while held-out flat = memorizing).
- [ ] **Week 5 — A better learning signal.** Beat plain fine-tuning: rationalization
  (learn from misses), DPO on correct-vs-incorrect pairs, a lightweight GRPO step,
  KL penalty vs drift. Each a measured ablation.
- [ ] **Week 6 — Stabilize, scale, present.** Fix multi-iteration failure modes
  (catastrophic forgetting → mix in original data; mode collapse → keep generations
  diverse; reward saturation → curriculum / move to harder MATH set). Polish
  analysis + README. Money shot: iteration-vs-accuracy chart.

---

## Background (reference — trim once internalized)

**Why it works — eliciting, not injecting.** A pretrained model has a
*distribution*, not a fixed skill. Sampled ten times on a hard problem, maybe two
attempts reason correctly and eight fail — the ability is latent but *unreliable*.
The loop closes the gap between "does it when lucky" and "does it reliably":
sample widely, verifier keeps only what worked, fine-tune to shift probability
mass toward successful reasoning. Next iteration, the correct approach is the
default. Pretraining injected the knowledge; our loop elicits and sharpens it.

**Why it's not circular.** Training on own output only degenerates (model
collapse) if you reinforce everything, errors included. The **verifier is the
teacher**: filtering for correctness is genuine *selection*, not an echo. Student
with an answer key — not importing new math, just making a wobbly skill reliable.

**The self-expanding curriculum.** Works where the base model already succeeds
*sometimes* (the 40–70% target). Once iteration 1 makes easy problems reliable,
the model is a *better generator*, so iteration 2 can crack medium problems it
couldn't before — those become fresh training data. The frontier creeps outward;
not capped at day-one ability.

**Learner = the model itself.** No separate rule-programmed agent. We never write
"to solve this, do X." The model works out the *how*; our code only checks answers
and feeds the good ones back. All solving intelligence lives in the weights.

**Optional alternative framing (if a game is preferred over math):** self-play
expert iteration on Connect-4 / Othello — identical recursive structure; game
rules are the verifier for free.
