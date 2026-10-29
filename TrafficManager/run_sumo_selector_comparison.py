"""Run three selector variants on matched SUMO scenes, without GPU services."""
import argparse
from datetime import datetime, timezone, timedelta
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from TrafficManager.experiments.sumo_selector_protocol import VARIANTS, PROMPT_VERSION, select_scene_ids
from TrafficManager.experiments.sumo_batch_resume import load_resume, selection_rejected, technical_failure

INDEX = ROOT / "TrafficManager/networkFiles/boston-seaport/scenarios_diverse_3types/scene_index.json"


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    return value


def save(path, value):
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2), encoding="utf-8")


def summarize(output, outcomes, planned_ids):
    from TrafficManager.attack_metric_calculator import load_pkl, ScoreCalculator, summarize_scene_results
    rows = {v: {} for v in VARIANTS}
    failures = []
    audits = {v: {} for v in VARIANTS}
    for status in outcomes:
        variant, scene_id = status["variant"], status["scene_id"]
        if not status["complete"]:
            failures.append(status)
            continue
        result_dir = ROOT / status["result_path"]
        try:
            records = load_pkl(str(result_dir / "drive_arena.pkl"))
            if len(records) != 1:
                raise ValueError("Expected one episode per scene")
            result = ScoreCalculator(records[0]).calculate(
                ttc_cap=10.0, collision_area_thr=0.0, danger_ttc_threshold=3.0, prd_cap=20.0,
            )
            if result is None:
                raise ValueError("No evaluable frames")
            rows[variant][scene_id] = result
            audits[variant][scene_id] = json.loads((result_dir / "selection_audit.json").read_text(encoding="utf-8"))
        except Exception as error:
            failures.append({"variant": variant, "scene_id": scene_id, "evaluation_error": str(error)})
    common = set.intersection(*(set(v) for v in rows.values()))
    mismatches = []
    for scene_id in sorted(common):
        hashes = {audits[v][scene_id]["text_sha256"] for v in VARIANTS}
        candidates = {json.dumps(audits[v][scene_id]["candidates"], sort_keys=True) for v in VARIANTS}
        if len(hashes) != 1 or len(candidates) != 1:
            mismatches.append(scene_id)
    paired = sorted(common - set(mismatches))
    summaries = {}
    per_type = {}
    report = ["# SUMO selector pilot", "", "Ego: SUMO nominal control. DreamForge and UniAD disabled.",
              ("Identical prompts and candidate inputs; frozen goal/path functions shared across all runs."
               if (output.parent / 'trajectory_templates').is_dir() else
               "Identical text prompts and candidate inputs; planner responses cached by exact request."),
              "One selection sample per model and scene. Exploratory results, not UniAD closed-loop results.", "",
              f"Planned scenes per variant: {len(planned_ids)}; paired valid scenes: {len(paired)}.", "",
              "| Variant | Valid / planned | Paired N | CR (%) | MMTTC (s) | DER (%) | CPRD (m/s²) | CRV (m/s) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    fields = ("collision_rate", "mean_min_ttc", "mean_danger_exposure_rate",
              "mean_peak_required_deceleration", "mean_collision_normal_relative_velocity")
    for variant in VARIANTS:
        summary = summarize_scene_results([rows[variant][s] for s in paired], do_print=False)
        summaries[variant] = summary
        vals = []
        for key in fields:
            val = summary[key] if summary else float("nan")
            if key in ("collision_rate", "mean_danger_exposure_rate"):
                val *= 100
            vals.append(f"{val:.3f}" if math.isfinite(val) else "N/A")
        report.append(f"| {variant} | {len(rows[variant])} / {len(planned_ids)} | {len(paired)} | " + " | ".join(vals) + " |")
        per_type[variant] = {
            attack: summarize_scene_results([rows[variant][s] for s in paired if s.startswith(attack + "_")], do_print=False)
            for attack in ("brake_check", "cut_in", "opposite_direction_intrusion")
        }
    report += ["", f"Failed/evaluation-error runs: {len(failures)}.",
               f"Selection rejections retained without retries: {sum(selection_rejected(s) for s in outcomes)}. These have no executed attack metrics and are not assigned zero collision rates.",
               "Input mismatches excluded from paired table: " + (", ".join(mismatches) or "none") + ".",
               "", "CRV is conditional on collisions; N/A means no measurable collision velocity."]
    save(output / "comparison.json", {"paired_scene_ids": paired, "summaries": summaries,
         "per_type": per_type, "scene_results": rows, "failures": failures, "input_mismatches": mismatches})
    (output / "comparison.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    with (output / "scene_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["variant", "scene_id", "paired", "attacker_id", "attack_type",
              "topology_corrected", "CR", "MMTTC", "DER", "CPRD", "CRV"])
        writer.writeheader()
        for variant in VARIANTS:
            for scene_id, row in sorted(rows[variant].items()):
                audit = audits[variant][scene_id]["selection_details"]
                status = next(s for s in outcomes if s["variant"] == variant and s["scene_id"] == scene_id)
                writer.writerow(clean({"variant": variant, "scene_id": scene_id, "paired": scene_id in paired,
                    "attacker_id": status["attacker_id"], "attack_type": status["attack_type"],
                    "topology_corrected": audit["topology_corrected"], "CR": row["scene_collision_flag"],
                    "MMTTC": row["scene_min_ttc_for_avg"], "DER": row["danger_exposure_rate"],
                    "CPRD": row["peak_required_deceleration"], "CRV": row["collision_normal_relative_velocity"]}))
    return summaries, len(paired)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-type-limit", type=int, default=5)
    parser.add_argument("--scene-timeout", type=int, default=600)
    parser.add_argument("--vision-model", default="qwen3-vl-plus")
    parser.add_argument("--two-stage", action="store_true", help="Separate selection/metrics and reuse fixed planner functions")
    parser.add_argument("--run-id")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--reuse-from", help="Import completed trials and the same frozen trajectory functions")
    parser.add_argument("--resume", action="store_true", help="Continue a stopped batch, preserving all attempted decisions")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for variant in ("vlm_image", "vlm_text"):
        VARIANTS[variant]["model"] = args.vision_model
    os.chdir(ROOT)
    ids = select_scene_ids(json.loads(INDEX.read_text(encoding="utf-8")), args.per_type_limit)
    outcomes = []
    previous = None
    if args.resume and (not args.two_stage or not args.run_id or args.reuse_from):
        raise ValueError('--resume requires --two-stage and --run-id, without --reuse-from')
    if args.reuse_from:
        if not args.two_stage:
            raise ValueError('--reuse-from requires --two-stage')
        from TrafficManager.experiments.reuse_sumo_batch import load_completed
        previous = (ROOT / args.reuse_from).resolve()
        outcomes = load_completed(ROOT, previous, ids, VARIANTS)
    plan = {"variants": VARIANTS, "scene_ids": ids, "total_runs": len(ids) * len(VARIANTS),
            "prompt_version": PROMPT_VERSION, "duration": 10.0, "attack_time": 2.0,
            "ego_control": "SUMO nominal", "dreamforge_enabled": False, "uniad_enabled": False,
            "scene_timeout_seconds": args.scene_timeout}
    plan.update(reused_runs=len(outcomes), new_runs=plan['total_runs'] - len(outcomes),
                reused_from=str(previous) if previous else None)
    if args.resume:
        if Path(args.run_id).name != args.run_id:
            raise ValueError('run-id must be a directory name')
        plan, outcomes = load_resume(ROOT, ROOT / 'results' / args.run_id, ids, VARIANTS)
        plan = dict(plan, resume_preserved=len(outcomes), resume_remaining=len(ids) * len(VARIANTS) - len(outcomes))
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    run_id = args.run_id or ("sumo_selector_pilot_" + datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S"))
    if Path(run_id).name != run_id:
        raise ValueError("run-id must be a directory name")
    output = ROOT / "results" / run_id
    if args.detach:
        log_path = ROOT / 'results' / (run_id + ('.resume_' + datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S') if args.resume else '') + '.log')
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--run-id', run_id,
                   '--per-type-limit', str(args.per_type_limit), '--scene-timeout', str(args.scene_timeout),
                   '--vision-model', args.vision_model]
        if args.two_stage:
            command.append('--two-stage')
        if previous:
            command += ['--reuse-from', str(previous)]
        if args.resume:
            command.append('--resume')
        with log_path.open('x', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, PYTHONHASHSEED='0'))
        print(json.dumps({'pid': process.pid, 'output': str(output), 'log': str(log_path)}))
        return
    output.mkdir(parents=True, exist_ok=args.resume)
    (output / "logs").mkdir(exist_ok=args.resume)
    snapshot_root = output
    if args.resume:
        snapshot_root = output / 'resume_history' / datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
        snapshot_root.mkdir(parents=True, exist_ok=False)
        (snapshot_root / 'previous_progress.json').write_bytes((output / 'progress.json').read_bytes())
    source_paths = [Path(__file__).resolve(), ROOT / "TrafficManager/experiments/sumo_selector_scene.py",
                    ROOT / "TrafficManager/experiments/sumo_selector_protocol.py", ROOT / "TrafficManager/sim_manager_only.py",
                    ROOT / "TrafficManager/sim_manager_t_attack.py", ROOT / "TrafficManager/attack_metric_calculator.py"]
    source_paths += sorted((ROOT / "TrafficManager/attack_framework").glob("*.py"))
    if args.two_stage:
        source_paths += [ROOT / 'TrafficManager/experiments/reusable_trajectory_templates.py',
                         ROOT / 'TrafficManager/run_selector_only_comparison.py',
                         ROOT / 'TrafficManager/experiments/reuse_sumo_batch.py']
        plan.update(stage1='stage1_selection', stage2='stage2_metrics',
                    trajectory_policy='qwen-coder-plus-1106 generates one reusable function pair per attack type',
                    selection_policy='fresh SUMO inputs; raw choice retained; no attack-type correction',
                    python_hash_seed=0)
    source_paths.append(ROOT / 'TrafficManager/experiments/sumo_batch_resume.py')
    for src in source_paths:
        dst = snapshot_root / "code_snapshot" / src.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
    plan["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    plan["scene_sha256"] = {s: {ext: hashlib.sha256((INDEX.parent / s / (s + ext)).read_bytes()).hexdigest()
                                      for ext in (".json", ".rou.xml", ".sumocfg")} for s in ids}
    save(snapshot_root / "experiment_config.json", plan)
    print("RUN_ROOT=" + str(output), flush=True)
    metrics_output = output
    if args.two_stage:
        from TrafficManager.experiments.reusable_trajectory_templates import prepare_templates
        from TrafficManager.run_selector_only_comparison import export_report
        phase1 = output / 'stage1_selection'
        (phase1 / 'responses').mkdir(parents=True, exist_ok=args.resume)
        metrics_output = output / 'stage2_metrics'
        metrics_output.mkdir(exist_ok=args.resume)
        selection_manifest = {'samples': ids, 'expected_calls': len(ids) * 3, 'combined_sumo_run': True}
        save(phase1 / 'manifest.json', dict(selection_manifest, variants=VARIANTS, prompt_version=PROMPT_VERSION))
        if args.resume:
            print(f'Resume: preserving {len(outcomes)} attempted trials; {plan["total_runs"] - len(outcomes)} remaining. No reselection of rejected decisions.', flush=True)
            export_report(phase1, selection_manifest)
            summarize(metrics_output, outcomes, ids)
        elif previous:
            from TrafficManager.experiments.reuse_sumo_batch import copy_artifacts
            copy_artifacts(previous, output)
            print(f'Reusing {len(outcomes)} completed trials and six frozen trajectory functions; no model calls for these.', flush=True)
            export_report(phase1, selection_manifest)
            summarize(metrics_output, outcomes, ids)
        else:
            print('Preparing six shared trajectory functions before all simulations', flush=True)
            prepare_templates(output / 'trajectory_templates')
    completed_keys = {(s['scene_id'], s['variant']) for s in outcomes}
    active = None
    started = time.monotonic()

    def terminate_child(signum=None, frame=None):
        if active is not None:
            try:
                os.killpg(active.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if signum is not None:
            save(output / "interrupted.json", {"signal": signum, "attempted": len(outcomes),
                 "reason": "Batch interrupted; inspect progress and logs. No incomplete run is counted as success."})
            raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate_child)
    signal.signal(signal.SIGINT, terminate_child)
    for scene_id in ids:
        if all((scene_id, variant) in completed_keys for variant in VARIANTS):
            continue
        for variant in VARIANTS:
            if (scene_id, variant) in completed_keys:
                continue
            ordinal = len(outcomes) + 1
            child_run_id = run_id + "_" + variant
            print(f"START {ordinal}/{plan['total_runs']} {scene_id} {variant}", flush=True)
            save(output / "progress.json", {"run_id": run_id, "total": plan["total_runs"], "attempted": len(outcomes),
                 "active": {"scene_id": scene_id, "variant": variant}, "outcomes": outcomes})
            log_path = output / "logs" / (scene_id + "_" + variant + ".log")
            command = [sys.executable, "-u", str(ROOT / "TrafficManager/experiments/sumo_selector_scene.py"),
                       "--scene-id", scene_id, "--run-id", child_run_id, "--variant", variant,
                       "--vision-model", args.vision_model,
                       "--cache-dir", str(output / "planner_cache")]
            if args.two_stage:
                command += ['--templates-dir', str(output / 'trajectory_templates'),
                            '--selection-dir', str(phase1)]
            timed_out = False
            tick = time.monotonic()
            with log_path.open("x", encoding="utf-8") as log:
                active = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                          stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                          env=dict(os.environ, PYTHONHASHSEED='0'))
                try:
                    active.wait(timeout=args.scene_timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    terminate_child()
                    try:
                        active.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(active.pid, signal.SIGKILL)
                        active.wait()
                finally:
                    terminate_child()
            status_path = ROOT / "results" / child_run_id / "_status" / scene_id / "run_status.json"
            status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
            status.update(variant=variant, scene_id=scene_id, returncode=active.returncode, timed_out=timed_out,
                          wall_seconds=round(time.monotonic() - tick, 2), log=str(log_path))
            status["complete"] = bool(status.get("complete") and status.get("attack_triggered") and active.returncode == 0 and not timed_out)
            if selection_rejected(status):
                status['failure_kind'] = 'selection_rejected'
            outcomes.append(status)
            active = None
            save(output / "progress.json", {"run_id": run_id, "total": plan["total_runs"], "attempted": len(outcomes),
                 "active": None, "elapsed_seconds": round(time.monotonic() - started, 1), "outcomes": outcomes})
            print(f"END {ordinal}/{plan['total_runs']} complete={status['complete']} seconds={status['wall_seconds']}", flush=True)
            if args.two_stage:
                export_report(phase1, selection_manifest)
            if len(outcomes) >= 3 and all(technical_failure(s) for s in outcomes[-3:]):
                summarize(metrics_output, outcomes, ids)
                raise RuntimeError("Three consecutive failures; inspect logs before further API calls")
        summaries, paired_n = summarize(metrics_output, outcomes, ids)
        print("PAIRED_PROGRESS=" + json.dumps(clean({"n": paired_n, "summaries": summaries})), flush=True)
    print("COMPLETE=" + str(metrics_output / "comparison.md"), flush=True)


if __name__ == "__main__":
    main()
