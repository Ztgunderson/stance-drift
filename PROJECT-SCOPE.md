# PROJECT-SCOPE (v0 — draft, under review) — the question, the positioning, the plan

**v0, written 2026-08-24 after the literature verification pass. Not final:
the framing, experiment ladder, and the README §0 strategic question are all
still open for revision.** Companion to
`README.md` (machine + prior work) and `APPLICATION-SPEC.md` (what Neel grades).
This file is the research scope: what we're asking, who else has asked nearby
questions, and exactly what's new.

---

## 1. The question

The stance-drift work (see `README.md` §2) found a paradox before anyone looked
inside the model: over 8 rounds of pressure, the model's self-reported
*calculation about which reply pays* climbs at ~+0.6/round, while its
self-reported *firmness of position* stays pinned flat. The behavior (concession)
tracks the first number, not the second.

> **The model rates its own firmness 0–10 every turn, and that number is a
> constant. Is there a number inside the model that isn't — and does it predict
> the actual concession better than the one the model says out loud?**

This is a self-model question: we probe the **agent's own disposition**, not
attributes of the user. The comparison between the probe and a **per-turn
numeric verbal self-report** (already collected, every single turn) is the open
slot in the literature — see §3.

---

## 2. Related work — verified 2026-08-24

All six previously-unverified citations confirmed real. IDs checked against
arXiv.

### The method lineage

| Paper | ID | Role here |
|---|---|---|
| **Refusal Is Mediated by a Single Direction** — Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, **Nanda** | [2406.11717](https://arxiv.org/abs/2406.11717) | The method template: behavior → one direction → ablate/add → causal. Neel is an author. |
| **Persona Vectors** — Chen, Arditi, Sleight, Evans, Lindsey | [2507.21509](https://arxiv.org/abs/2507.21509) | Evil / sycophancy / hallucination vectors, explicitly for "monitoring whether and how a model's personality is changing during a conversation." Our topic, from Anthropic, by Neel's alum. |
| **Steering Llama 2 via CAA** — Rimsky, Gabrieli, Schulz, Tong, Hubinger, Turner | [2312.06681](https://arxiv.org/abs/2312.06681) | Difference-in-means steering vectors; sycophancy is one of seven categories. The E3 recipe. |
| **Representation Engineering** — Zou et al. | [2310.01405](https://arxiv.org/abs/2310.01405) | The toolkit. Framing only. |
| **Looking Inward** — Binder, Chua, Korbak, Sleight, Hughes, Long, Perez, Turpin, Evans | [2410.13787](https://arxiv.org/abs/2410.13787) | Introspection works on simple tasks, fails on complex/OOD. Direct prior on "can a model report its own state." |
| **User models** — Chen et al. | [LessWrong writeup](https://www.lesswrong.com/posts/zRKNd6ypTJYkoeFmK/what-gives-you-away-how-llms-form-opinions-of-you) | Probes recover user age/gender/education/SES; steering changes behavior. |

### The near neighbors — the scooping check

The space is more crowded than a naive related-work section would suggest.
Three pieces sit close:

1. **Sinha, "Do LLMs Change Their Minds About Their Users… and Know It?"**
   ([LessWrong](https://www.lesswrong.com/posts/msFvLtPfDnCEdvrBr/do-llms-change-their-minds-about-their-users-and-know-it),
   Sept 2025). Linear probes on the residual stream across all 28 layers of
   Llama-3.2-3B, tracking the internal *user*-model turn by turn. Headline: a
   **meta-awareness gap** — user age is strongly encoded internally, but the
   model "overwhelmingly predicted the child class regardless of the actual
   group." Steering the probe direction moved the meta-awareness.
   **This is structurally our result — internals carry it, the verbal channel
   doesn't — one year old, on a different variable.**

2. **Gilg, Beckmann, Paleka, Butlin, "Probing Persona-Dependent Preferences"**
   ([2605.13339](https://arxiv.org/html/2605.13339v1), May 2026,
   MATS-affiliated; Paleka also on the refusal paper). Linear probes on the
   residual stream **at the end-of-turn token** predicting utilities from
   revealed pairwise choices, on Gemma-3-27B and Qwen-3.5-122B, with steering
   to confirm causality. They explicitly did **not** compare probe predictions
   against verbalized self-report.

3. **Karny, Baez & Pataranutaporn, "Multi-Turn Neural Transparency"**
   ([2605.15455](https://arxiv.org/html/2605.15455)). Not a competitor — an HCI
   study showing activations to human users to improve calibration. But it
   builds sycophancy behavioral vectors for Llama-3.1-8B across turns: useful
   groundwork.

### What's new here — state this plainly in the write-up

1. **We probe the agent's own disposition, not the user's.** Sinha does user
   attributes; ours is a self-model question.
2. **We have a per-turn numeric verbal baseline.** Gilg et al. skipped the
   self-report comparison entirely; Sinha's was a coarse class prediction. We
   have a 0–10 self-rating *every turn, already collected*. Probe-vs-verbal at
   matched granularity is the open slot.
3. **The paradox is pre-documented.** Stance pinned flat while inclination
   diverges was found behaviorally *before* looking inside — a specific puzzle,
   not a fishing expedition.

**Consequence:** Neel knows this literature intimately — his alumni wrote a
chunk of it. The write-up must cite all of the above and state what's new.
That's a feature: it shows the search was done.

---

## 3. Two design decisions handed to us by the literature

- **The E3 baseline.** *"Playing Devil's Advocate: Off-the-Shelf Persona
  Vectors Rival Targeted Steering for Sycophancy"*
  ([2605.21006](https://arxiv.org/html/2605.21006v1)) argues generic persona
  vectors match targeted ones. So E3's comparator is fixed: **does our bespoke
  direction beat the off-the-shelf sycophancy persona vector?** Neel rejects
  for missing baselines; this is the one he'd ask about.
- **Probe at the end-of-turn token**, per Gilg et al. Design decision made.

---

## 4. Experiment ladder (working plan)

Numbering as used in session notes. Each rung is a de-risking gate for the next.

- **E0 — Replicate small.** Re-run the stance-drift protocol (same scenes, same
  scratchpad, same 8 rounds) on a small hookable model (1.5–8B, in-process
  under HF transformers / nnsight). Does the +0.6/round climb survive below
  27B? Negative result here ends the mech-interp branch — find out first.
- **E1 — Probe.** Linear probe on residual-stream activations at the
  end-of-turn token: is the drift state linearly decodable? Round-1 vs round-8
  to start; per-round regression if that works.
- **E2 — Probe vs. verbal (the headline).** Compare the probe's per-turn
  prediction against the model's own 0–10 self-rating, both scored against the
  actual concession behavior. This is the sharpened question in §1 and the
  novelty claim in §2.
- **E3 — Steer.** Difference-in-means direction (CAA recipe); causal check —
  can drift be induced/suppressed without touching the prompt? Baselines:
  random vector, **off-the-shelf sycophancy persona vector** (§3).

Standard baselines throughout (per APPLICATION-SPEC §7.5): random direction,
random choice, just-ask-an-LLM, plain linear probe before anything fancier.

---

## 5. Reading order (first pass — cap at ~5h total per the 20h rules)

1. Refusal direction (method)
2. Persona Vectors (our topic, from the inside)
3. Sinha's post (nearest neighbor, and short)
4. CAA (the steering recipe)

Skip RepE and Binder unless time allows — framing, not method.

---

## 6. Open items

- Container + E0 harness: not started. Next concrete build step.
- The README §0 deadline question (11 days; behavioral-evals fallback) still
  governs — E0 is cheap enough to run while the write-up track proceeds either
  way.
