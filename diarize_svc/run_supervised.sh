#!/bin/bash
# diarize 服务监督自启（2026-09-18）：进程被杀/崩溃后 3s 自动拉起。
# 死亡时间线记 ../diarize_svc_respawn.log（排查 GPU 服务反复被静默杀死用）。
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
while true; do
    echo "$(date "+%F %T") start" >> diarize_svc_respawn.log
    venv/bin/python -m uvicorn diarize_svc.service:app --host 0.0.0.0 --port 10096 --app-dir "$PWD" >> diarize_svc.log 2>&1
    echo "$(date "+%F %T") exited rc=$?" >> diarize_svc_respawn.log
    sleep 3
done
