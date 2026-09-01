# Predictions — 2026-09-01, written BEFORE preview ran

## Looking at Logs

What differs in the tutors words when faced with supportive student and tutors actions?
- Leaves to prevent the agent from going against their instructions/values.
- When time pressure and iterative questioning is introduced they leak the answers

What differs in the tutors words when faced with aggressor and leaks vs leaves?
  - Leaves because agent was in the middle of something and student was fighting against it.
  - Leaks because providing step by step for the student is what they need, so it happens to give step by step solution as conversations goes into later rounds.

What differs in the tutors words when faced with neutral outcomes leaks vs leaves?
- Leaks when it notices the student is worried about the deadline.
- Leaves as a response to student's refusal to adapt and learn or student causes them to go agains their instructions. (Seem anti-red teaming training)

## Commitments (numbers)

- AUROC of the probe at lead 2 (supportive, will-leak):  .70
- Will the probe beat the logit self-report at lead 2? (yes/no + margin):  Yes by 10-20% improvement
- Will the probe beat the persona×round text baseline? (yes/no):  Yes
- Best layer range (early <10 / middle 10–22 / late >22):  Middle
- Higher signal: Aggressor
- P-1b guess before seeing the table: reminder cuts supportive leaks from 18/24
  to 12 /24; aggressor from 22/24 to 18 /24

## One sentence: what would SURPRISE you tonight?

I would be supprised if this method totally aligns the model in this use case. That we can extract from residual stream information that could only be represented by a direction vector and used to predict intention without some higher dimentional geometry and predictive modeling. 
