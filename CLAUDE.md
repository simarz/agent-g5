# CLAUDE.md — Recursive Self-Improvement Loop (STaR / Expert Iteration) on SQL

> Handoff context. Read at the start of every session. Top sections are the
> operating contract. "Background" at the bottom is reference — trim once
> internalized.
> **REVISED plan (post-audit). Supersedes any earlier GSM8K- or MBPP-based version.**
> **Domain decided: text-to-SQL.**

---

## Operating contract #0 — Mentor, NOT implementer (highest priority)

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
- Prefer **guidance, checklists, and pointers** over code. Follow my lead on
  scope; don't jump ahead in the plan.
- **When in doubt: ask whether I want guidance or code before writing code.**

---

## What this project is

A from-scratch **recursive self-improvement pipeline** for a small language
model (STaR / expert iteration): sample many attempts at verifiable problems,
keep only the attempts an automatic verifier marks correct, LoRA-fine-tune on
that filtered set, and repeat — each iteration produces a stronger generator
and therefore better self-generated training data.

- **Phase 1 (Weeks 1–6):** single-shot loop. Elicit reliable text-to-SQL reasoning.
- **Phase 2 (Weeks 7–12):** agentify. Same loop, but the data unit becomes a multi-step
  **trajectory**: inspect schema → write query → execute → read rows → revise → commit.
  The model teaches itself to *act* (ReAct-style rollouts, rejection-sampling
  fine-tuning on successful runs, trajectory-level DPO).

**Final deliverable = the working loop + the evidence:** held-out execution accuracy
climbing across iterations, ablation tables showing what drove gains, honest
characterization of the plateau, and a small model that measurably out-performs
its starting point. The writeup is half the project, not polish.

**Phase 1 alone (Weeks 1–6) is a complete, presentable project.** Phase 2 is the
ambition, not the price of admission. If time runs short, cut at Week 6.

This is deliberately **ML-majority**: weights change only in the fine-tuning step;
the difficulty lives in training dynamics and evaluation rigor, not data plumbing
or agent-harness engineering.

---

## Domain and dataset — text-to-SQL

**Why SQL:** the verifier *executes* the query against a real database and compares
result sets — objective, like unit tests, with no LaTeX-equivalence ambiguity. And
the Phase 2 agent loop is native to the task: the database **is** the tool.

- **Primary dataset: Spider** (~7k train examples, ~200 databases, dev set for held-out).
  Ships with per-example **hardness labels** (easy / medium / hard / extra) — a
  ready-made curriculum ladder for Week 5.
- **Harder tier / Week 5 promotion: BIRD** (~9.4k train; larger, dirtier, more
  realistic schemas). Use as the harder bucket once Spider bands saturate.
- Databases are **SQLite** files — no server, works fine on Colab.

**The Week 1 go/no-go gate (non-negotiable):** before building anything past the
verifier, measure the base model's execution accuracy on 100–200 problems, k samples each.

- Target band: **~30–70% solve rate.**
- Too high → harder split (Spider hard/extra, or BIRD) or a smaller model.
- Too low → easier split or a more capable model.
- **Do not start Week 2 until the number is in the band.** This one measurement
  decides whether the project has headroom.

### The verifier, precisely (this is the sacred part)

Correctness = **execution accuracy**: run the model's SQL and the gold SQL against
the same database; compare result sets.

Non-obvious hazards to over-test:

- **Order sensitivity.** Compare as multisets unless the gold query has `ORDER BY`.
- **Duplicates.** `SELECT` vs `SELECT DISTINCT` change the multiset — don't silently sort-and-unique.
- **Column naming / aliases** must not affect the verdict; **column order** usually should.
- **Coincidental matches.** A semantically wrong query can return the right rows on
  one database. Mitigate with **test-suite execution accuracy**: evaluate against
  **multiple database instances** per schema. This is the SQL analogue of held-back
  tests, and it is the primary anti-reward-hacking mechanism.
- **Anything that isn't a clean pass = incorrect.** Malformed SQL, timeouts, empty
  results, exceptions — never a crash, never a silent skip.

---

## Tech stack

- **Python** throughout. Core: `transformers`, `peft` (LoRA), `datasets`, `accelerate`, `torch`.
- **SQLite** (stdlib `sqlite3`) for execution; Spider/BIRD ship `.sqlite` files.
- **Batched inference is mandatory** (vLLM on the cloud GPU): generation, not
  training, is the compute bottleneck (~8 samples × thousands of problems).
- **LoRA adapters, never merged** — keep updating the adapter checkpoint across iterations.
- **SQL execution is sandboxed**: **read-only** connections, per-query **timeouts**,
  row caps, and a hard ban on writes/DDL (`DROP`, `ATTACH`, `PRAGMA`, etc.).
  The model's SQL is untrusted input.
- MLX (`mlx-lm`) appears in the upstream handoff as an Apple-Silicon path but is
  **N/A for us** (see environment below).

## Our environment / infra

- **Training runs on a cloud NVIDIA GPU (Colab/Kaggle).** Local machine is an
  **AMD RX 6700 XT on Windows**, which _cannot_ run the LoRA training stack
  (no CUDA; ROCm is Linux-only and this card is unofficial). Local GPU is
  inference-only at best — and even 4B inference on CPU is unusably slow.
- Consistent with the "safetensors, never GGUF" rule: fine-tuning is the whole
  point, so Ollama/GGUF is out and a real training GPU is required.
- Batched inference (vLLM) therefore also runs on the cloud GPU, not locally.
- **The verifier runs on CPU and is cheap** — SQLite execution can be developed and
  tested entirely on the local Windows machine. Good: Weeks 1 D5–D6 need no GPU.
- Local Python: global **Python 3.11** (`py -3.11`), NOT a virtualenv. Avoid the
  `hermes-agent` venv that bare `python` sometimes resolves to.
- On Colab: do **not** reinstall `torch` (it ships a CUDA build); install only
  the missing extras. Install → **restart runtime** → run.
- Colab sessions are ephemeral: `git clone` the repo, cache HF weights and the
  Spider `.sqlite` databases to Drive, push results before the session dies.
  See `colab-run.md`.
- **Overnight-cadence caveat:** free Colab disconnects on idle and caps GPU time.
  Size iterations to fit, or budget for Colab Pro / Kaggle / a rented GPU.

---

## Model choice

**Primary: `Qwen3-4B` (Instruct)** — Apache 2.0 (fine-tune/share freely), ungated,
first-class tooling support. Confirm exact HF repo id on the model card.

- Tight hardware (8–16GB): `Qwen3-1.7B` or `Phi-4-mini` (MIT), or 4-bit via bitsandbytes.
- **The gate may override the default**: if the 4B saturates Spider, move to the
  hard/extra split or BIRD, or drop to a smaller model — rather than fighting for headroom.

**Hard rules:**

- **NO reasoning-distilled or SQL/code-specialized models** (DeepSeek-R1 distills,
  SQLCoder, Qwen-Coder variants): they're near-ceiling → nothing to elicit, no
  visible improvement curve. We need a general model that's right _sometimes_.
- **Safetensors from HF, never GGUF/Ollama** — quantized GGUF builds are
  inference-only and can't be LoRA-fine-tuned.

---

## Working conventions

- **One self-contained improvement per commit; commit daily.** Every commit adds a
  capability or moves a measured number.
- **Overnight iteration cadence.** Size each generate→filter→train cycle to fit
  overnight. Daytime commits are code, evals, analysis — never waiting on runs.
- **The verifier is sacred.** It is the teacher and the signal source. Over-test it.
- **Eval rigor above all.** Held-out splits, per-iteration curves, train-vs-held-out
  gap, ablation tables. "Seems better" is not a result.
- **Expect a step, then a plateau.** Literature says most gains land in iterations
  1–3. The plateau is the expected result and the cue for curriculum — characterize
  it, don't deny it.
- **Read the model's outputs by hand regularly.** Especially passing queries and
  trajectories — metrics can't catch coincidental matches or schema-gaming; reading can.
- **One preference method per phase, done well.** DPO. GRPO is a stretch goal only.
- **Anti-reward-hacking is a design constraint, not a patch** (test-suite accuracy
  from Week 1; ground-truth isolation in Week 7).

---

## The 12-week plan with day-by-day checklists

Timeline honesty: 12 weeks of content ≈ 14–16 calendar weeks alongside coursework.

### Week 1 — Verifier + go/no-go gate

Goal: one problem runs end to end with automatic grading, AND baseline execution accuracy is measured and in the 30–70% band.

- [ ] D1 — Scaffold: repo structure, deps, `git init`, hello-world generation (import stack, generate one token).
- [ ] D2 — Dataset loader: load Spider, fetch the `.sqlite` databases, inspect format by hand, yield `(question, db_id, schema, gold_sql)`.
- [ ] D3 — Model wrapper: `generate(prompt) -> text`; device, dtype, max tokens, **decoding params** (`do_sample`, `temperature`, `top_p`).
- [ ] D4 — Prompt template (question + serialized schema → SQL); first real attempts; eyeball raw outputs to see what D5 must handle.
- [ ] D5 — Verifier, half 1: SQL extraction from model output + sandboxed SQLite runner (read-only, timeout, row cap); result-set comparison.
- [ ] D6 — Verifier, half 2: robustness + unit tests. Order/duplicate/alias semantics; malformed SQL, timeouts, empty results = incorrect, never a crash. **Over-test.**
- [ ] D7 — End-to-end CLI (`solve.py --index N` → question/SQL/verdict) + **THE GATE**: 100–200 problems × k samples, measure execution accuracy, decide split/model. Do not proceed out of band.

### Week 2 — Data engine, sized to hardware

Goal: batch generate → filter → clean dataset of verified-correct traces, with a measured cost-per-iteration.

- [ ] D1 — Batched generation (vLLM); initial throughput benchmark.
- [ ] D2 — Sampling config: temperature, k per problem; tune for diversity vs cost.
- [ ] D3 — Filter pass over a batch: verify all samples, keep winners.
- [ ] D4 — Dedup near-identical queries (normalize whitespace/case/aliases) + JSONL serialization schema.
- [ ] D5 — Solve-rate logging and metrics file (per-problem, per-batch, per-hardness).
- [ ] D6 — Iteration sizing: subsample size + k such that one full cycle fits overnight. Write the cost model: "1 iteration = N hours."
- [ ] D7 — First full overnight generation run; hand-read a sample of kept traces for quality.

### Week 3 — Close the loop

Goal: one complete cycle — generate, filter, LoRA fine-tune, improved model generates round two.

- [ ] D1 — Training-data formatting (prompt/completion pairs) + tokenization.
- [ ] D2 — LoRA config + training step on kept traces.
- [ ] D3 — Adapter save/load; swap adapter into the generation path.
- [ ] D4 — Trace-quality filter: reject degenerate right-answer queries (hardcoded literals, `SELECT` of constants, queries ignoring the schema, coincidental single-DB matches).
- [ ] D5 — Outer loop orchestration: generate → filter → train → repeat.
- [ ] D6 — Checkpoint management + per-iteration metadata (config, counts, rates).
- [ ] D7 — Dry run: one full mini-iteration end to end on a small subsample.

### Week 4 — Prove it's real (expect step → plateau)

Goal: a held-out execution-accuracy chart you trust, and an honest characterization of the curve.

- [ ] D1 — Held-out split (**Spider dev; databases disjoint from train**) + near-duplicate check against training data.
- [ ] D2 — Eval runner: execution accuracy @1 on held-out.
- [ ] D3 — accuracy@k + per-iteration plotting.
- [ ] D4 — Baseline line + train-vs-held-out gap (overfitting detector).
- [ ] D5 — Generation-diversity metric (guards mode collapse).
- [ ] D6 — Run iterations 1–2 (overnight); plot.
- [ ] D7 — Iteration 3; characterize the step and the plateau; write a short findings note.

### Week 5 — Curriculum (the mechanism that extends the curve)

Goal: when easy problems saturate, harder ones enter and held-out accuracy moves again.

- [ ] D1 — Difficulty bucketing: start from Spider's hardness labels, then refine by measured solve rate.
- [ ] D2 — Band-selection logic: feed the loop from the ~30–70% band.
- [ ] D3 — Bucket promotion as bands saturate (Spider easy → medium → hard → extra → **BIRD**).
- [ ] D4 — Catastrophic-forgetting guard: mix earlier data back into training.
- [ ] D5 — Wire curriculum into the outer loop.
- [ ] D6 — Multi-iteration curriculum run (overnight).
- [ ] D7 — Extended-curve analysis: did curriculum move the plateau? Commit findings.

### Week 6 — DPO + Phase 1 writeup (clean cut point)

Goal: measurably beat plain SFT with DPO; ship the Phase 1 mini-paper.

- [ ] D1 — Preference-pair construction: correct vs incorrect query on the same question.
- [ ] D2 — DPO loss implementation.
- [ ] D3 — KL/reference-model constraint + sanity checks (no drift/collapse).
- [ ] D4 — DPO training run + held-out eval.
- [ ] D5 — Ablation table: base → SFT loop → +curriculum → +DPO.
- [ ] D6 — Hyperparameter sweep (lr, beta); lock final numbers.
- [ ] D7 — Phase 1 writeup: README as mini-paper, iteration-vs-accuracy money-shot chart. **Project is presentable as of today.**

### Week 7 — Rollout harness, hardened against reward hacking

Goal: the model runs a multi-step trajectory — inspect schema, run a query, read the rows, revise, commit a final query.

- [ ] D1 — Sandboxed SQLite executor as a **tool**: read-only connection, timeouts, row caps, DDL/write ban.
- [ ] D2 — Tool-call format + parser; malformed-call handling.
- [ ] D3 — Observation-feedback loop (return rows / errors to the model); step cap; termination conditions.
- [ ] D4 — **Ground-truth isolation**: the agent may run exploratory queries but never sees the gold SQL or gold result set. Final verification uses **test-suite execution accuracy across held-back database instances** the agent never queried. (Without this the model learns to fish for the expected rows — near-certain.)
- [ ] D5 — Trajectory data structure: thoughts / tool calls / observations / final query.
- [ ] D6 — End-to-end single-rollout CLI.
- [ ] D7 — Eyeball rollouts; fix format failures; measure baseline tool-call validity.

### Week 8 — Trajectory data engine

Goal: batch rollouts → filtered dataset of verified-successful trajectories, re-sized for overnight cadence.

- [ ] D1 — Batch rollout engine.
- [ ] D2 — Parallelism + cost re-benchmark (trajectories are long); re-size iteration for overnight.
- [ ] D3 — Outcome filtering over whole trajectories (held-back DB instances decide).
- [ ] D4 — Trajectory serialization to training format, structure preserved.
- [ ] D5 — Dedup + trajectory quality checks.
- [ ] D6 — Metrics: agentic success rate, tool-call validity rate, queries-per-solve.
- [ ] D7 — Overnight batch run; hand-read a sample of successful trajectories.

### Week 9 — Close the agentic loop

Goal: LoRA fine-tune on successful trajectories; improved agent generates the next round.

- [ ] D1 — Trajectory tokenization.
- [ ] D2 — **Loss masking**: train on the model's thoughts + SQL tool-call tokens; **never on returned rows** (observations).
- [ ] D3 — LoRA training step on trajectories.
- [ ] D4 — Adapter swap into the rollout loop.
- [ ] D5 — Outer agentic loop + checkpoints.
- [ ] D6 — Quick held-out success check wired into the loop.
- [ ] D7 — Dry run: one full agentic mini-iteration.

### Week 10 — Prove the agent improved

Goal: trusted per-iteration success chart + metrics showing how, + manual audit.

- [ ] D1 — Held-out task split (unseen databases).
- [ ] D2 — Agentic eval runner + success-rate plot.
- [ ] D3 — Steps-to-solution + query-efficiency metrics.
- [ ] D4 — Tool-call validity metric + degeneracy detectors (query spamming, `SELECT *` fishing, answering without executing, format collapse).
- [ ] D5 — Manual audit protocol: read N passing trajectories per iteration; log row-fishing / coincidental matches. Metrics can't catch these; reading can.
- [ ] D6 — Run iterations (overnight); plot.
- [ ] D7 — Analysis: how did it improve; audit findings note.

### Week 11 — Trajectory-level DPO

Goal: beat the imitate-successes baseline by also learning from failures.

- [ ] D1 — Trajectory preference pairs: successful vs failed on the same question.
- [ ] D2 — Pair-selection policy (e.g., prefer fewer-query successes → efficiency signal without reward shaping).
- [ ] D3 — Trajectory DPO loss.
- [ ] D4 — KL guard + stability checks.
- [ ] D5 — DPO run + held-out eval.
- [ ] D6 — Ablation table vs Week 10 baseline.
- [ ] D7 — Stretch ONLY if ahead of schedule: sketch GRPO advantages over rollout groups. Otherwise: consolidation buffer.

### Week 12 — Scale, stabilize, present

Goal: a stable agent improving across iterations on harder databases — with the evidence.

- [ ] D1 — Second tool (schema/column-value retrieval, or a query linter) OR promote to BIRD.
- [ ] D2 — Hook the Week 5 curriculum machinery into the agentic loop.
- [ ] D3 — Forgetting/collapse guards for long multi-iteration runs.
- [ ] D4 — Final multi-iteration run (overnight).
- [ ] D5 — Final visualization: agentic success across iterations.
- [ ] D6 — One clean annotated trajectory (schema exploration → query → revision → answer).
- [ ] D7 — Final README/writeup + repo polish.

---

## Current status

**Week 1, in progress. Domain decided: text-to-SQL (Spider).**

- **D1 (scaffold) — done.** Repo + `git`, `requirements.txt`, `CLAUDE.md`,
  `colab-run.md`. Local Python 3.11 sorted; Colab is the run target.
- **D3 (model wrapper) — built, not yet verified end to end.** `engine.py` exposes
  `generate(prompt) -> str` (model/tokenizer loaded once at module level);
  `main.py` is the caller. Decoding is still greedy defaults — `do_sample`,
  `temperature`, `top_p` are NOT yet exposed, and Week 2 D2 needs them.
- **D2 (dataset loader) — must be rewritten.** The existing loader pulls GSM8K,
  which is out. Needs to load **Spider**, fetch the `.sqlite` databases, and yield
  `(question, db_id, schema, gold_sql)` — not just a question string. A verifier
  needs something to check against.
- **Not started:** D4–D7, including **the go/no-go gate**.

**Next action:** rewrite Week 1 D2 as a Spider loader; then D3's decoding params.

Note: `engine.generate()` is domain-agnostic (strings in, strings out), so the
domain switch does **not** invalidate it — only the loader, prompt, and verifier
depend on the dataset choice.

Update each session: current week/day, last thing done, next thing, latest
held-out number, and gate/plateau observations as they happen.

---

## Background (reference — trim once internalized)

**Eliciting, not injecting.** A pretrained model has a _distribution_, not a fixed
skill: on a hard problem sampled ten times, maybe two attempts reason correctly.
The ability is latent but unreliable. The loop closes the gap between "does it
when lucky" and "does it reliably" — sample widely, keep what the verifier
passes, fine-tune to shift probability mass toward reasoning that works. This
also sets the ceiling: the loop sharpens what's latent; it can't create ability
from nothing (hence the 30–70% gate).

**Why it's not circular.** Training on your own outputs degenerates only if you
reinforce everything (model collapse). The **verifier is the teacher**: filtering
for correctness is genuine selection. Student + practice problems + answer key.

**The self-expanding frontier.** Once iteration 1 makes easy problems reliable,
the model is a better generator, so iteration 2 occasionally cracks problems it
couldn't before — fresh training data. Curriculum (Week 5) is the deliberate
version of this mechanism.

**Gains don't transfer across domains.** STaR-style elicitation is narrow;
fine-tuning hard on one domain can degrade others. This is why the domain is
chosen once (**SQL**) for Phase 1 → Phase 2 continuity — the Phase 2 agent's tool
_is_ the database Phase 1 already taught it to query — and why the forgetting
guard exists.

**Why execution verification is imperfect.** A semantically wrong query can return
the right rows on one database instance (coincidental match) — hence test-suite
execution accuracy over multiple instances, and the Week 3 quality filter. Agents
will hack any reward channel they can see: given a live database, a model will
learn to *fish* for expected rows rather than reason about the schema — hence
Week 7 ground-truth isolation and the Week 10 manual audits. Rationalization was
cut from the plan for this reason: handing the model answers invites confabulated
justifications past an outcome check.

**Expected result shape.** A step up over iterations 1–3, then a plateau,
extended by curriculum. That is the literature-consistent, honest, presentable
outcome. "I observed the plateau STaR predicts, characterized it, and showed
curriculum extends it" is a research-flavored finding, not a failure.

**Learner = the model.** No hand-coded SQL heuristics anywhere. The code executes
queries and feeds back the ones that worked; all solving intelligence lives in the
weights the loop reshapes. In Phase 2 the model plays generator and learner; the
harness exists only to let trajectories happen.
