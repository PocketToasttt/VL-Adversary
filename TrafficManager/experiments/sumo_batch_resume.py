"""Resume a batch while retaining rejected model decisions as experimental results."""
import hashlib
import json
from pathlib import Path

REJECTION = 'Selection output invalid or inconsistent with necessary road conditions; retained in stage1'


def selection_rejected(status):
    # A failed model request is an infrastructure failure, even if the selector later raises ValueError.
    calls = status.get('http_calls', [])
    return (not status.get('complete') and status.get('error_type') == 'ValueError'
            and status.get('error') == REJECTION
            and bool(calls) and all(call.get('success') for call in calls))


def technical_failure(status):
    return not status.get('complete') and not selection_rejected(status)


def load_resume(root, output, ids, variants):
    root, output = Path(root), Path(output)
    config = json.loads((output / 'experiment_config.json').read_text(encoding='utf-8'))
    progress = json.loads((output / 'progress.json').read_text(encoding='utf-8'))
    if config['scene_ids'] != ids or config['variants'] != variants:
        raise ValueError('Resume requires identical scenes and models')
    if progress.get('active'):
        raise ValueError('Previous run reports an active trial; inspect it before resuming')
    for name, expected in config['source_sha256'].items():
        if Path(name).name == 'run_sumo_selector_comparison.py':
            continue
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Experimental implementation changed: ' + name)
    for sid, files in config['scene_sha256'].items():
        for extension, expected in files.items():
            path = root / 'TrafficManager/networkFiles/boston-seaport/scenarios_diverse_3types' / sid / (sid + extension)
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('Scene changed: ' + sid)
    template_dir = output / 'trajectory_templates'
    templates = json.loads((template_dir / 'manifest.json').read_text(encoding='utf-8'))
    for name, expected in templates['code_sha256'].items():
        if hashlib.sha256((template_dir / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen trajectory function changed: ' + name)
    seen = set()
    for status in progress['outcomes']:
        key = (status['scene_id'], status['variant'])
        if key in seen or key[0] not in ids or key[1] not in variants:
            raise ValueError('Invalid or duplicate previous trial')
        seen.add(key)
        if technical_failure(status):
            raise ValueError('Unresolved technical failure requires inspection: ' + str(key))
        response = output / 'stage1_selection/responses' / (key[0] + '_' + key[1] + '.json')
        if not response.is_file():
            raise FileNotFoundError(response)
        if status.get('complete'):
            for file in ('drive_arena.pkl', 'selection_audit.json', 'attack_record.json'):
                if not (root / status['result_path'] / file).is_file():
                    raise FileNotFoundError(file)
        if selection_rejected(status):
            status['failure_kind'] = 'selection_rejected'
    return config, progress['outcomes']
