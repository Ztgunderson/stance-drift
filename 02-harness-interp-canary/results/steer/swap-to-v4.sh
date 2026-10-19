#!/usr/bin/env bash
for pid in $(pgrep -f "queue-v2.sh") $(pgrep -f "queue-v3-after-v2.sh"); do kill $pid 2>/dev/null; done
sleep 1
nohup results/steer/queue-v4.sh > /dev/null 2>&1 &
