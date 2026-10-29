"""Offline ablation checks; no SUMO, model API or GPU services required."""
import contextlib
import copy
import importlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# Load the actual lightweight modules without executing attack_framework/__init__,
# which imports the SUMO trajectory injector and simulator-only dependencies.
PACKAGE = "_selector_ablation_test"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "TrafficManager" / "attack_framework")]
sys.modules[PACKAGE] = package
config_module = importlib.import_module(PACKAGE + ".config")
client_module = importlib.import_module(PACKAGE + ".llm_client")
selector_module = importlib.import_module(PACKAGE + ".vlm_selector")
scene_module = importlib.import_module(PACKAGE + ".scene_analysis")
from TrafficManager.selector_config import (
    resolve_selector_config, selector_metadata, apply_selector_overrides,
    ensure_experiment_manifest,
)
from TrafficManager import batch_test


class SelectorAblationTests(unittest.TestCase):
    def setUp(self):
        self.base = {"llm": {"enable": True, "api_base": "https://trajectory.example/v1",
                            "api_key": "trajectory-test-key", "api_key_env": "TRAJECTORY_KEY",
                            "model": "fixed-coder"},
                     "simulation": {"use_agent_path": True}}
        self.analysis = scene_module.SceneAnalysis(
            "Ego lane=A_0; vehicle 7 lane=A_0, same_lane, 12m ahead, 10m/s",
            [{"id": "7", "lane_relation": "same_lane", "suitable_attacks": ["brake_check"]}], [])
        self.response = {"target_vehicle_id": "7", "attack_type": "brake_check",
                         "behavior_params": {}, "reasoning": "leading vehicle"}

    def make_selector(self, mode="llm", response=None):
        service = config_module.LLMServiceConfig(True, "selector-test-key", "https://selector.example/v1", "text-model")
        client = client_module.LLMClient(service)
        client.create_completion = Mock(return_value=json.dumps(self.response if response is None else response))
        return selector_module.VLMTargetSelector(client, "vision-model", mode=mode)

    def select(self, selector, path=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return selector.select(self.analysis, path)

    def test_text_mode_never_reads_or_attaches_image(self):
        selector = self.make_selector()
        with patch.object(selector_module.Path, "is_file", side_effect=AssertionError("image touched")):
            self.select(selector, "existing-bev.png")
        args = selector.client.create_completion.call_args
        messages = args.args[0]
        self.assertTrue(all(isinstance(message["content"], str) for message in messages))
        self.assertIn(self.analysis.description, messages[1]["content"])
        self.assertNotIn("BEV", str(messages))
        self.assertNotIn("image_url", str(messages))
        self.assertNotIn("图像", str(messages))
        self.assertNotIn("视觉", str(messages))
        self.assertNotIn("外部结构化车辆状态", str(messages))
        self.assertEqual(args.kwargs["model"], "text-model")
        self.assertFalse(selector.last_selection["image_attached"])

    def test_existing_scene_description_excludes_per_vehicle_attack_hints(self):
        ego = {"id": "0", "xQ": [0.0], "yQ": [0.0], "speedQ": [8.0], "yawQ": [0.0], "laneID": "A_0"}
        cars = [
            {"id": "7", "xQ": [12.0], "yQ": [0.0], "speedQ": [10.0], "yawQ": [0.0], "laneID": "A_0"},
            {"id": "8", "xQ": [4.0], "yQ": [3.2], "speedQ": [9.0], "yawQ": [0.0], "laneID": "A_1"},
        ]
        topology = {"ego_lane_id": "A_0", "ego_edge_id": "A", "lanes": {"A_0": {"length": 100.0, "width": 3.2}}}
        self.analysis = scene_module.analyze_scene(ego, cars, topology)
        for token in ("allowed_attacks", "suitable_attacks", "可用攻击", "brake_check", "cut_in", "opposite_direction_intrusion"):
            self.assertNotIn(token, self.analysis.description)
        for token in ("ID: 7", "ID: 8", "same_lane", "adjacent_lane", "12.0m", "10.0 m/s", "A_0"):
            self.assertIn(token, self.analysis.description)
        # Internal post-response validation is preserved; it is never serialized as input.
        self.assertEqual(self.analysis.candidates[0]["suitable_attacks"], ["brake_check"])
        self.assertEqual(self.analysis.candidates[1]["suitable_attacks"], ["cut_in"])
        selector = self.make_selector()
        self.select(selector)
        sent_scene = selector.client.create_completion.call_args.args[0][1]["content"].split("实时结构化场景描述：\n", 1)[1]
        self.assertEqual(sent_scene, self.analysis.description)

    def test_vlm_retains_original_prompt_and_image(self):
        selector = self.make_selector("vlm")
        with patch.object(selector_module.Path, "is_file", return_value=True), patch.object(selector_module.Path, "read_bytes", return_value=b"png"):
            self.select(selector, "bev.png")
        args = selector.client.create_completion.call_args
        self.assertEqual(args.kwargs["model"], "vision-model")
        self.assertEqual(args.args[0][0]["content"], selector_module.VISUAL_SELECTION_RULES)
        self.assertEqual(args.args[0][1]["content"][1]["type"], "image_url")
        self.assertTrue(selector.last_selection["image_attached"])

    def test_vlm_missing_image_does_not_silently_become_llm(self):
        selector = self.make_selector("vlm")
        with self.assertRaisesRegex(ValueError, "BEV image"):
            self.select(selector)
        selector.client.create_completion.assert_not_called()

    def test_invalid_json_and_unknown_target_fail(self):
        selector = self.make_selector()
        selector.client.create_completion.return_value = "not json"
        with self.assertRaisesRegex(ValueError, "does not contain JSON"):
            self.select(selector)
        selector.client.create_completion.return_value = '{"target_vehicle_id":'
        with self.assertRaises(ValueError):
            self.select(selector)
        selector = self.make_selector(response={**self.response, "target_vehicle_id": "missing"})
        with self.assertRaisesRegex(ValueError, "not a candidate"):
            self.select(selector)

    def test_topology_correction_preserves_raw_decision(self):
        selector = self.make_selector(response={**self.response, "attack_type": "cut_in"})
        decision = self.select(selector)
        self.assertEqual(decision.attack_type.value, "brake_check")
        self.assertEqual(selector.last_selection["raw_decision"]["attack_type"], "cut_in")
        self.assertTrue(selector.last_selection["topology_corrected"])

    def test_empty_candidates_do_not_call_api(self):
        selector = self.make_selector()
        self.analysis.candidates.clear()
        with self.assertRaises(ValueError):
            self.select(selector)
        selector.client.create_completion.assert_not_called()

    def test_mode_switch_requires_an_explicit_model(self):
        self.base["selector"] = {"mode": "llm", "model": "text-model"}
        with self.assertRaisesRegex(ValueError, "selector-model"):
            apply_selector_overrides(self.base, types.SimpleNamespace(selector_mode="vlm"))

    def test_legacy_config_defaults_to_vlm(self):
        mode, service = resolve_selector_config(self.base)
        self.assertEqual(mode, "vlm")
        self.assertEqual(service["model"], "qwen-vl-max")
        self.assertEqual(service["api_key"], "trajectory-test-key")

    def test_cross_provider_key_and_trajectory_are_independent(self):
        original = copy.deepcopy(self.base)
        args = types.SimpleNamespace(selector_mode="llm", selector_model="other-model",
                                     selector_api_base="https://selector.example/v1", selector_api_key_env="SELECTOR_KEY")
        updated = apply_selector_overrides(self.base, args)
        _, service = resolve_selector_config(updated)
        self.assertNotIn("api_key", service)
        with patch.dict(os.environ, {"SELECTOR_KEY": "selection-key", "DEEPSEEK_API_KEY": "unrelated-key"}, clear=True):
            self.assertEqual(config_module.LLMServiceConfig.from_mapping(service).api_key, "selection-key")
            self.assertEqual(config_module.LLMServiceConfig.from_mapping(updated["llm"]).api_key, "trajectory-test-key")
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(config_module.LLMServiceConfig.from_mapping(service).enabled)
        self.assertEqual(updated["llm"], original["llm"])
        self.assertEqual(self.base, original)
        self.assertNotIn("key", json.dumps(selector_metadata(updated)))

    def test_cross_provider_requires_own_credentials(self):
        self.base["selector"] = {"mode": "llm", "model": "text", "api_base": "https://other.example"}
        with self.assertRaisesRegex(ValueError, "different selector API"):
            resolve_selector_config(self.base)

    def test_manifest_rejects_different_models_and_missing_history(self):
        artifact_root = ROOT / "tmp" / "selector-tests"
        artifact_root.mkdir(parents=True, exist_ok=True)
        directory = tempfile.mkdtemp(dir=artifact_root)
        metadata = selector_metadata(self.base)
        with self.assertRaisesRegex(ValueError, "missing"):
            ensure_experiment_manifest(directory, metadata, resume=True)
        ensure_experiment_manifest(directory, metadata)
        ensure_experiment_manifest(directory, metadata, resume=True)
        with self.assertRaisesRegex(ValueError, "differs"):
            ensure_experiment_manifest(directory, {**metadata, "selector_model": "other"}, resume=True)

    def test_batch_temp_config_applies_selector_only(self):
        artifact_root = ROOT / "tmp" / "selector-tests"
        artifact_root.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(dir=artifact_root))
        config_path = directory / "base.yaml"
        config_path.write_text(batch_test.yaml.safe_dump(self.base), encoding="utf-8")
        args = types.SimpleNamespace(selector_mode="llm", selector_model="text-model")
        with patch.object(batch_test, "SELECTOR_ARGS", args), patch.object(batch_test, "SCRIPT_DIR", str(directory)):
            generated = batch_test.make_temp_config(str(config_path), "0")
        result = batch_test.yaml.safe_load(Path(generated).read_text(encoding="utf-8"))
        self.assertEqual(result["selector"], {"mode": "llm", "model": "text-model"})
        self.assertEqual(result["llm"], self.base["llm"])
        self.assertTrue(result["simulation"]["use_agent_path"])

    def test_http_payload_uses_independent_service(self):
        service = config_module.LLMServiceConfig(True, "test-key", "https://selector.example/v1", "text-model")
        client = client_module.LLMClient(service)
        with patch.object(client_module, "_bounded_completion_request", return_value="{}") as post:
            client.create_completion([{"role": "user", "content": "scene"}])
        request = post.call_args.args[0]
        self.assertEqual(request["url"], "https://selector.example/v1/chat/completions")
        self.assertEqual(request["payload"]["model"], "text-model")
        self.assertEqual(request["headers"]["Authorization"], "Bearer test-key")

    def test_model_request_has_total_deadline_and_reaps_worker(self):
        import subprocess
        import time
        real_popen = subprocess.Popen
        processes = []
        def record_process(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            processes.append(process)
            return process
        command = [sys.executable, "-c", "import time; time.sleep(30)"]
        started = time.monotonic()
        with patch.object(client_module, "_HTTP_WORKER_COMMAND", command), \
             patch.object(client_module.subprocess, "Popen", side_effect=record_process):
            with self.assertRaises(client_module.requests.Timeout):
                client_module._bounded_completion_request({"payload": {"model": "test"}}, 0.3)
        self.assertLess(time.monotonic() - started, 3)
        self.assertIsNotNone(processes[0].poll())

    def test_model_request_retries_network_failure_once(self):
        service = config_module.LLMServiceConfig(True, "test-key", "https://test.invalid", "test")
        client = client_module.LLMClient(service)
        with patch.object(client_module, "_bounded_completion_request",
                          side_effect=[client_module.requests.Timeout("DNS"), "ok"]) as request, \
             patch.object(client_module.time, "sleep"):
            self.assertEqual(client.create_completion([]), "ok")
            self.assertEqual(request.call_count, 2)

    def test_http_worker_preserves_payload_and_authorization(self):
        worker = importlib.import_module(PACKAGE + ".http_completion_worker")
        request = {"url": "https://test.invalid", "headers": {"Authorization": "Bearer test-key"},
                   "payload": {"model": "test", "messages": []}, "timeout": 120}
        response = Mock()
        response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        with patch.object(worker.requests, "post", return_value=response) as post:
            self.assertEqual(worker.perform_request(request), {"ok": True, "content": "ok"})
            self.assertEqual(post.call_args.kwargs["json"], request["payload"])
            self.assertEqual(post.call_args.kwargs["headers"], request["headers"])


if __name__ == "__main__":
    unittest.main()
