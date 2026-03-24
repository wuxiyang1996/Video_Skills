#!/bin/bash
CKPT_DIR="/workspace/game_agent/Game-AI-Agent/runs/Qwen3-8B_2048_20260322_071227/checkpoints/step_0011"
MAIN_PID=1759870
POLL_INTERVAL=30

echo "[watcher] Waiting for checkpoint at $CKPT_DIR ..."
echo "[watcher] Will send SIGTERM to PID $MAIN_PID once metadata.json appears."

while true; do
    if [ -f "$CKPT_DIR/metadata.json" ]; then
        echo "[watcher] $(date): Checkpoint step_0011 detected!"
        sleep 5  # let any final writes flush
        echo "[watcher] Sending SIGTERM to PID $MAIN_PID ..."
        kill -TERM "$MAIN_PID" 2>/dev/null
        sleep 10
        if kill -0 "$MAIN_PID" 2>/dev/null; then
            echo "[watcher] Process still alive, sending SIGKILL ..."
            kill -KILL "$MAIN_PID" 2>/dev/null
        fi
        echo "[watcher] Done. Training stopped after step 11 checkpoint."
        exit 0
    fi
    sleep "$POLL_INTERVAL"
done
