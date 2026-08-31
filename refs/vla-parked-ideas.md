# VLA / physical-AI — parked ideas & debrief

**Written 2026-08-25, closing the VLA exploration.** Decision: the Sep-4
application is the stance-drift continuation (PROJECT-SCOPE E0–E3). This file
is the parking lot so nothing is lost and nothing distracts. Field snapshot
lives in `vla-interp-field-survey.md`; don't reopen either file until a
revisit trigger below fires.

---

## 1. Debrief — what the exploration established (one paragraph)

VLA interp (Aug 2026) is a ~10-paper, 12-month-old field still in its
toolkit-transfer phase; the crowded corner is failure-detection probes, the
empty corners are the second-generation questions (diffing across the VLM→VLA
fine-tune, post-training science, anything alignment-flavored — e.g. does the
refusal direction survive action fine-tuning). Nothing robotics-flavored
appears in Neel's stated interests or starter-project lists, and his
pet-interest warning applies directly, so a VLA project is wrong for THIS
application — but the field explicitly has no talent pipeline (Stocking & Häon
LW white paper) and the empty corners need exactly the instruments the E-ladder
teaches. **Deferred, not abandoned.**

---

## 2. The skill bridge — why E0–E3 is training for the parked scope

The stance-drift ladder teaches three instruments plus a discipline. Each maps
one-to-one onto a second-generation question and onto a parked VLA idea:

| E-rung | Instrument learned | Second-generation question it powers | Parked VLA analog |
|---|---|---|---|
| **E0** (replicate small) | Behavioral harness rigor; "check the phenomenon exists in *your* setting before building on it" | Post-training science — every claim about what a training stage did needs the behavior verified at your scale first | Verify OpenVLA actually fails semantic-safety tasks (LIBERO-Safety) before probing why |
| **E1** (linear probe) | Probing internal state per turn; layer/token-position choices; probe baselines | Elicitation of latent knowledge; monitoring; eval awareness | ProbeAct-style state probes; "does the backbone still represent harmfulness?" |
| **E2** (probe vs verbal report) | **The general instrument: compare the model's verbal channel against its internals, scored on behavior.** | CoT faithfulness; introspection; meta-awareness gaps — Neel's model-biology core | The "silent instruction channel": A3 (2603.19233) showed VLAs often ignore language while behavior follows vision — same shape, verbal channel vs actual policy |
| **E3** (diff-in-means steering + persona-vector baseline) | **A diff vector between conditions is the atomic unit of model diffing.** Bespoke vs off-the-shelf direction comparison | Model diffing & post-training science: "is the fine-tuning diff just a bias term or something deeper?" is the same math across *checkpoints* instead of across *turns* | Extract refusal direction from Llama-2-chat, test survival in OpenVLA (same backbone) — pure E3 skill transfer |
| (throughout) | Baselines, skepticism, random-vector controls, read-your-data | What separates accepted from rejected applications | What separates a real VLA-safety result from the 5th toolkit-transfer paper |

**The one-line version:** stance-drift studies *character drift across turns*;
post-training science studies *character drift across training steps*; VLA
diffing studies *character drift across a modality fine-tune*. Same linear
toolkit, different axis. E0–E3 is the cheapest place to learn it because the
phenomenon, data, and harness already exist here.

Two deeper continuities worth remembering:

- **If E1 finds a drift direction**, the natural next project is: is it the
  same direction that moves under sycophancy *fine-tuning*? That's a literal
  in-context-vs-in-weights diffing experiment — square in Neel's interests,
  and a bridge from this project to post-training science with no robotics
  needed.
- **E2's probe-vs-verbal design is portable to any system with a report
  channel and a behavior channel** — CoT vs answer, instruction vs action,
  stated principle vs choice. Learning it on stance-drift is learning it for
  everything on this list.

---

## 3. Parked idea queue (with revisit triggers)

| # | Idea | Shape | Revisit when |
|---|---|---|---|
| P1 | **Refusal-survival diffing**: does the Arditi et al. refusal direction survive OpenVLA's action fine-tune of Llama-2-7B? Black-box pilot ≈2h (chat-prompt the backbone); white-box via E3 skills | Model diffing wearing a robot | (a) E0 fails and the sprint needs a pivot target, or (b) research-phase proposal / MATS 13 idea list |
| P2 | **VLM→VLA diff characterization**: Prismatic-7B vs OpenVLA — is the diff a "you are a robot now" bias term or deeper? (Neel's narrow-fine-tuning question verbatim) | Model diffing / post-training science | 3–6 month project slot; needs only cached activations, no simulator |
| P3 | **Silent instruction channel**: when does a VLA's behavior detach from its instruction while looking compliant? (extends A3's visual-dominance finding; prompt-injection-adjacent) | Model biology / applied interp, embodied | After P1/P2 give internals fluency on VLAs; needs rollout infra |
| P4 | **Stage-wise post-training on VLAs**: extend A2's "world model emerges over training" across pretrain→SFT→RL checkpoints | Post-training science | Access to checkpoint series + real compute budget |
| P5 | **Gemma-Scope→π0 transfer**: do Gemma-family SAEs/persona vectors transfer into PaliGemma-based π0? ⚠ verify Gemma-version compatibility before spending anything | Improved methods | Only if P1/P2 already working; version check first |
| P6 | **Physical-AI safety field entry**: the Stocking–Häon white paper says no MATS-equivalent pipeline exists; Jetson + robotics-adjacent skills are a real differentiator there | Career positioning, not a paper | After MATS outcome known (Nov 2026); or if applying to robotics-safety-flavored programs |

**Do-not-do list** (crowded or wrong-shaped): failure-detection probes
(4 papers already), another SAE-dictionary-on-a-VLA paper, anything requiring
a physical robot before the internals story exists.

---

## 4. Rule for the sprint

Until the application is submitted: **no VLA reading, no VLA code, no
simulator installs.** If a robotics idea intrudes mid-sprint, append one line
to §3 and go back to work. The E-ladder *is* the robotics prep.
