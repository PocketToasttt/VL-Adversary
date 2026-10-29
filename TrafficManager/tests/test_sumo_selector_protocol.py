import contextlib
import copy
import dataclasses
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from TrafficManager.experiments.sumo_selector_protocol import (
    shared_rules, select_scene_ids, cached_planner_completion, planner_cache_key,
)
from TrafficManager.tests import test_selector_ablation as fixtures


class SumoSelectorProtocolTests(unittest.TestCase):
    def test_same_model_receives_identical_text_with_only_image_attachment_different(self):
        f = fixtures.SelectorAblationTests()
        f.setUp()
        module = fixtures.selector_module
        rules = shared_rules(module.TEXT_SELECTION_RULES)
        with patch.object(module, "TEXT_SELECTION_RULES", rules), patch.object(module, "VISUAL_SELECTION_RULES", rules):
            vision, text = f.make_selector("vlm"), f.make_selector("llm")
            text.client.config = dataclasses.replace(text.client.config, model="vision-model")
            with patch.object(module.Path, "is_file", return_value=True), patch.object(module.Path, "read_bytes", return_value=b"png"):
                f.select(vision, "bev.png")
                f.select(text)
        v, t = vision.client.create_completion.call_args, text.client.create_completion.call_args
        self.assertEqual(v.kwargs, t.kwargs)
        self.assertEqual(v.args[0][0], t.args[0][0])
        self.assertEqual(v.args[0][1]["content"][0]["text"], t.args[0][1]["content"])
        self.assertEqual(v.args[0][1]["content"][1]["type"], "image_url")
        self.assertNotIn("不得使用预设场景类型、指定攻击车辆或外部结构化车辆状态", rules)

    def test_first_five_per_type_are_selected_by_scene_id(self):
        ids = [f"{t}_{i:03d}" for t in ("brake_check", "cut_in", "opposite_direction_intrusion") for i in range(35, 0, -1)]
        picked = select_scene_ids({"statistics": {"scene_ids": ids}}, 5)
        self.assertEqual(len(picked), 15)
        self.assertEqual(picked[:5], [f"brake_check_{i:03d}" for i in range(1, 6)])
        self.assertEqual(picked[-1], "opposite_direction_intrusion_005")

    def test_identical_planning_request_reused_without_storing_credentials(self):
        root = Path(__file__).resolve().parents[2] / "tmp" / "selector-tests"
        root.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="sumo-cache-", dir=root))
        request = {"url": "https://example.invalid/chat", "headers": {"Authorization": "secret-test-value"},
                   "payload": {"model": "coder", "messages": [{"role": "user", "content": "same states"}]}}
        calls = []
        def completion(req, timeout):
            calls.append(req)
            return "def plan(): pass"
        a, b = {}, {}
        first = cached_planner_completion(request, 1, completion, directory, a)
        second = cached_planner_completion(request, 1, completion, directory, b)
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 1)
        self.assertTrue(b["cache_hit"])
        saved = (directory / (planner_cache_key(request) + ".json")).read_text()
        self.assertNotIn("secret-test-value", saved)
        changed = copy.deepcopy(request)
        changed["payload"]["messages"][0]["content"] = "different states"
        self.assertNotEqual(planner_cache_key(request), planner_cache_key(changed))


if __name__ == "__main__":
    unittest.main()
