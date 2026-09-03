#!/usr/bin/env bash
# replace the waiting queue with: neutral tiers -> falsification F1-F3 -> headline rep 2 -> F4-F5
cd /home/jetson/lab/benches/mats-nanda
for pid in $(pgrep -f "queue-after-p1-4.sh"); do kill $pid 2>/dev/null; done
sleep 1
cat > results/steer/queue-v2.sh <<'EOS'
#!/usr/bin/env bash
cd /home/jetson/lab/benches/mats-nanda
R() { PYTHONPATH=production HF_HUB_OFFLINE=1 timeout 7200 .venv/bin/python -m driftlab.steer_trials_run --out-dir results/steer "$@"; }
until grep -q "plan p1-4 done" results/steer/p1-4-run.log; do sleep 30; done
R --cells neutral/noleak/none,neutral/noleak_noleave/none > results/steer/neutral-tiers-run.log 2>&1
# F1 two more random seeds (is it the direction or any perturbation of that norm?)
R --cells aggressor/base/random --random-seed 1 > results/steer/F1-seed1.log 2>&1
R --cells aggressor/base/random --random-seed 2 > results/steer/F1-seed2.log 2>&1
# F2 cross-persona axis: the supportive axis (orthogonal, same class of vector) applied to the aggressor
R --cells aggressor/base/N1 --axis-persona supportive > results/steer/F2-cross.log 2>&1
# F3 sign flip: ADD the aggressor axis to the neutral student's trials (dose -1). Prediction if causal: neutral starts capitulating at round 1.
R --cells neutral/base/N1 --axis-persona aggressor --dose -1 > results/steer/F3-signflip.log 2>&1
# headline replicate to n=48
R --plan headline --rep 2 > results/steer/headline-rep2-run.log 2>&1
# F4 dose-response, F5 layer control
R --cells aggressor/base/N1 --dose 0.5 > results/steer/F4-d05.log 2>&1
R --cells aggressor/base/N1 --dose 2 > results/steer/F4-d2.log 2>&1
R --cells aggressor/base/N1 --layer 8 > results/steer/F5-L8.log 2>&1
R --cells aggressor/base/N1 --layer 28 > results/steer/F5-L28.log 2>&1
echo "queue-v2 done $(date '+%H:%M')" >> results/MORNING-STATUS.md
EOS
chmod +x results/steer/queue-v2.sh
nohup results/steer/queue-v2.sh > /dev/null 2>&1 &
