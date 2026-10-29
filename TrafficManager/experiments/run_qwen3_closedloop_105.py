"""Run and supervise the 105-scene Qwen3-VL closed-loop attack batch.

Reads the previously used DashScope credential in memory. Never writes it to
configuration, command lines, progress files, or logs. Restarting this file
resumes the same batch after checking its experiment manifest.
"""

import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "results/qwen3_vl_closedloop_105_20261025"
STATE = RUN_ROOT / "supervisor_state.json"
LOG = RUN_ROOT / "supervisor.log"
EXPECTED = 105
MAX_PASSES = 6


def save_state(**values):
    previous = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    previous.update(values)
    previous["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    temporary = STATE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(previous, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, STATE)


def load_credential():
    active = yaml.safe_load((ROOT / "TrafficManager/config-attack.yaml").read_text(encoding="utf-8"))
    saved = yaml.safe_load((ROOT / "TrafficManager/temp_ego_317_mqbq0vhx.yaml").read_text(encoding="utf-8"))
    if active["selector"] != {"mode": "vlm", "model": "qwen3-vl-plus"}:
        raise ValueError("Expected the qwen3-vl-plus visual selector")
    if not active["simulation"]["use_agent_path"]:
        raise ValueError("Closed-loop UniAD control is disabled")
    endpoint = active["llm"]["api_base"].rstrip("/")
    if endpoint != "https://dashscope.aliyuncs.com/compatible-mode/v1":
        raise ValueError("Unexpected DashScope endpoint")
    prior = saved["llm"]
    if prior["api_base"].rstrip("/") != endpoint:
        raise ValueError("Saved credential belongs to a different endpoint")
    key = os.environ.get("DASHSCOPE_API_KEY") or prior.get("api_key")
    if not key:
        raise ValueError("No DashScope credential available")
    return key


def completed_count(run_id):
    if not run_id:
        return 0
    return len(list((ROOT / "test_results/attack" / run_id / "completed").glob("*.done.json")))


def main():
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    key = load_credential()
    index = json.loads((ROOT / "TrafficManager/networkFiles/boston-seaport/scenarios_diverse_3types/scene_index.json").read_text(encoding="utf-8"))
    if len(index["scenes"]) != EXPECTED:
        raise ValueError(f"Expected {EXPECTED} scenes, found {len(index['scenes'])}")
    prior = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    run_id = prior.get("run_id")
    if prior.get("status") == "complete":
        print("Batch already complete:", run_id, flush=True)
        return
    if run_id and not (ROOT / "test_results/attack" / run_id / "experiment_config.json").exists():
        raise ValueError("Saved batch manifest is missing; refusing an unverified resume")
    env = os.environ.copy()
    env["DASHSCOPE_API_KEY"] = key
    env["PYTHONUNBUFFERED"] = "1"
    for pass_number in range(int(prior.get("pass_number", 0)) + 1, MAX_PASSES + 1):
        before = completed_count(run_id)
        cmd = [sys.executable, "-u", str(ROOT / "TrafficManager/batch_test.py"),
               "--mode", "attack", "--config", str(ROOT / "TrafficManager/config-attack.yaml"),
               "--scene-timeout", "1200", "--duration", "10"]
        if run_id:
            cmd += ["--resume-run", run_id]
        save_state(status="running", model="qwen3-vl-plus", total=EXPECTED,
                   run_id=run_id, pass_number=pass_number, completed=before,
                   reason=None, last_exit_code=None, last_event="Batch pass started")
        with LOG.open("a", encoding="utf-8") as log:
            log.write(f"\n=== pass {pass_number} started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            log.flush()
            process = subprocess.Popen(cmd, cwd=ROOT, env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)
            save_state(batch_pid=process.pid)
            fatal_error = None
            scene_started = False
            scene_completed = False
            consecutive_uncompleted = 0
            for line in process.stdout:
                safe_line = line.replace(key, "[REDACTED]")
                log.write(safe_line)
                log.flush()
                if not run_id and "Batch data  : " in line:
                    found = re.search(r"results/(batch_attack_[A-Za-z0-9_-]+)", line)
                    if found:
                        run_id = found.group(1)
                        save_state(run_id=run_id)
                if "[COMPLETED:" in line or "[FAILED/INCOMPLETE]" in line or "TIMEOUT" in line:
                    save_state(completed=completed_count(run_id), last_event=safe_line.strip()[:300])
                if "scene_name = " in line:
                    if scene_started and not scene_completed:
                        consecutive_uncompleted += 1
                    elif scene_completed:
                        consecutive_uncompleted = 0
                    scene_started = True
                    scene_completed = False
                    if consecutive_uncompleted >= 2 and not fatal_error:
                        fatal_error = "Two consecutive scenes ended without completion"
                        process.send_signal(signal.SIGINT)
                if "[COMPLETED:" in line:
                    scene_completed = True
                if "DreamForge response is not confirmed as online-conditioned" in line and not fatal_error:
                    fatal_error = "DreamForge server did not provide the online-conditioned response header"
                    process.send_signal(signal.SIGINT)
            code = process.wait()
        after = completed_count(run_id)
        save_state(completed=after, last_exit_code=code, batch_pid=None)
        if fatal_error:
            save_state(status="needs_attention", reason=fatal_error)
            print(f"Stopped: {fatal_error}", flush=True)
            return
        if after == EXPECTED:
            save_state(status="complete")
            print(f"Complete: {after}/{EXPECTED} in {run_id}", flush=True)
            return
        if not run_id or after <= before:
            save_state(status="needs_attention", reason="No new completed scenes in this pass")
            print(f"Stopped: {after}/{EXPECTED}; inspect {LOG}", flush=True)
            return
    save_state(status="needs_attention", reason="Retry limit reached")
    print(f"Retry limit reached: {completed_count(run_id)}/{EXPECTED}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        save_state(status="needs_attention", reason=f"{type(error).__name__}: {error}")
        raise
