import unittest
from TrafficManager.experiments.sumo_batch_resume import REJECTION, selection_rejected, technical_failure


class ResumeFailureTests(unittest.TestCase):
    def rejection(self):
        return dict(complete=False, error_type='ValueError', error=REJECTION,
                    http_calls=[dict(success=True)])

    def test_model_rejection_is_retained_without_stopping_batch(self):
        outcome = self.rejection()
        self.assertTrue(selection_rejected(outcome))
        self.assertFalse(technical_failure(outcome))
        self.assertFalse(outcome['complete'])
        self.assertFalse(all(technical_failure(self.rejection()) for _ in range(3)))

    def test_api_failure_does_not_masquerade_as_model_rejection(self):
        outcome = self.rejection()
        outcome['http_calls'] = [dict(success=False)]
        self.assertFalse(selection_rejected(outcome))
        self.assertTrue(technical_failure(outcome))

    def test_unrelated_failure_still_stops_after_three(self):
        outcomes = [dict(complete=False, error_type='RuntimeError', error='SUMO disconnected') for _ in range(3)]
        self.assertTrue(all(technical_failure(s) for s in outcomes))
        self.assertFalse(technical_failure(dict(complete=True)))


if __name__ == '__main__':
    unittest.main()
