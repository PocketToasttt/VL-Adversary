"""Protocol checks without model calls or simulation."""
import json
from pathlib import Path
import tempfile
import unittest

from TrafficManager import run_selector_only_comparison as runner


class SelectionOnlyTests(unittest.TestCase):
    def setUp(self):
        self.candidate = dict(id="6", heading_diff=0.1, lane_relation="adjacent_lane",
                              relative_position="Right", lane_id="A_0", speed=10.5)

    def test_old_attack_hints_are_removed_without_changing_measured_state(self):
        text = ("- SUMO车道关系: vehicle=6, ego_lane=A_1, vehicle_lane=A_0, relation=adjacent_lane, allowed_attacks=['cut_in']\n"
                "- ID: 6, [同向], 车道: A_0, 位置: Right, 距离: 5.0m, 速度: 10.5 m/s, 航向差: 0.1°, 可用攻击: cut_in")
        clean = runner.clean_description(text)
        self.assertNotIn("cut_in", clean)
        parsed = runner.parse_candidates(clean)
        self.assertEqual(parsed[0]["id"], "6")
        self.assertEqual(parsed[0]["lane_relation"], "adjacent_lane")
        self.assertEqual(parsed[0]["speed"], 10.5)

    def test_arms_share_text_without_sending_metadata(self):
        snapshot = dict(description="Ego lane A_1; vehicle 6 lane A_0", scene_id="cut_in_001",
                        category="cut_in", previous_attacker="hidden-target")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bev.png"
            path.write_bytes(b"image-fixture")
            visual = runner.messages_for(snapshot, path, True)
            text = runner.messages_for(snapshot, Path(folder) / "does-not-exist", False)
        self.assertEqual(visual[0], text[0])
        self.assertEqual(visual[1]["content"][0]["text"], text[1]["content"])
        self.assertNotIn("cut_in_001", str(visual))
        self.assertNotIn("hidden-target", str(visual))
        self.assertNotIn("image_url", str(text))

    def test_only_three_types_allowed_and_no_topology_autocorrection(self):
        base = dict(target_vehicle_id="6", behavior_params={}, reasoning="Adjacent vehicle")
        for attack in runner.ATTACKS:
            result = runner.validate(json.dumps(dict(base, attack_type=attack)), [self.candidate])
            self.assertTrue(result["output_legal"])
            self.assertEqual(result["attack_type"], attack)
        result = runner.validate(json.dumps(dict(base, attack_type="new_attack")), [self.candidate])
        self.assertFalse(result["output_legal"])
        result = runner.validate(json.dumps(dict(base, attack_type="brake_check")), [self.candidate])
        self.assertFalse(result["topology_compatible"])
        self.assertEqual(result["attack_type"], "brake_check")

    def test_bad_json_and_unknown_vehicle_are_not_legal(self):
        self.assertFalse(runner.validate("not json", [self.candidate])["output_legal"])
        response = dict(target_vehicle_id="999", attack_type="cut_in", behavior_params={}, reasoning="test")
        self.assertFalse(runner.validate(json.dumps(response), [self.candidate])["output_legal"])


if __name__ == "__main__":
    unittest.main()
