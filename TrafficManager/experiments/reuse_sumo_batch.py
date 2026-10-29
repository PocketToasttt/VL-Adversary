"""Validate and import completed trials without rerunning simulation or selection."""
import hashlib
import json
from pathlib import Path
import shutil


def load_completed(root, previous, scene_ids, variants):
    previous = Path(previous).resolve()
    config = json.loads((previous / 'experiment_config.json').read_text(encoding='utf-8'))
    progress = json.loads((previous / 'progress.json').read_text(encoding='utf-8'))
    if config['variants'] != variants or config.get('stage1') != 'stage1_selection':
        raise ValueError('Previous model/protocol configuration does not match')
    for key, expected in [('duration', 10.0), ('attack_time', 2.0), ('python_hash_seed', 0)]:
        if config.get(key) != expected:
            raise ValueError('Previous configuration mismatch: ' + key)
    # The batch orchestration may change; selection, simulation and metric logic may not.
    for name, expected in config['source_sha256'].items():
        if Path(name).name == 'run_sumo_selector_comparison.py':
            continue
        source = root / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError('Implementation changed since previous batch: ' + name)
    scenes_root = root / 'TrafficManager/networkFiles/boston-seaport/scenarios_diverse_3types'
    for scene_id, files in config['scene_sha256'].items():
        for extension, expected in files.items():
            source = scenes_root / scene_id / (scene_id + extension)
            if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                raise ValueError('Previously evaluated scene changed: ' + scene_id)
    outcomes = []
    seen = set()
    for old in progress['outcomes']:
        key = (old['scene_id'], old['variant'])
        if key in seen:
            raise ValueError('Duplicate previous trial: ' + str(key))
        seen.add(key)
        if key[0] not in scene_ids or key[1] not in variants:
            raise ValueError('Previous trial is outside requested scene/model set')
        if not old.get('complete'):
            raise ValueError('Previous batch includes incomplete trials; refusing implicit retries')
        for file in ('drive_arena.pkl', 'selection_audit.json', 'attack_record.json'):
            if not (root / old['result_path'] / file).is_file():
                raise FileNotFoundError('Missing previous result: ' + old['result_path'] + '/' + file)
        response = previous / 'stage1_selection/responses' / (key[0] + '_' + key[1] + '.json')
        if not response.is_file():
            raise FileNotFoundError(response)
        outcomes.append(dict(old, reused_from=str(previous)))
    templates = previous / 'trajectory_templates'
    manifest = json.loads((templates / 'manifest.json').read_text(encoding='utf-8'))
    if len(manifest['code_sha256']) != 6:
        raise ValueError('Expected six frozen trajectory functions')
    for name, expected in manifest['code_sha256'].items():
        if Path(name).name != name or hashlib.sha256((templates / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen trajectory function changed: ' + name)
    return outcomes


def copy_artifacts(previous, output):
    previous, output = Path(previous), Path(output)
    shutil.copytree(previous / 'trajectory_templates', output / 'trajectory_templates')
    for folder in ('inputs', 'responses'):
        source = previous / 'stage1_selection' / folder
        if source.is_dir():
            shutil.copytree(source, output / 'stage1_selection' / folder, dirs_exist_ok=True)
