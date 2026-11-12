from __future__ import annotations

import unittest

from volumecv.geometry import VolumeResult
from volumecv.session import CompetitionSession


def _result(shape: str) -> VolumeResult:
    return VolumeResult(
        shape=shape,
        dimensions={"a_mm": 60.0},
        volume_mm3=1000.0,
        volume_cm3=1.0,
        sigma_mm3=1.0,
        relative_sigma=0.001,
        confidence=0.99,
    )


class SessionTests(unittest.TestCase):
    def test_bonus_requires_completed_base(self) -> None:
        now = [0.0]
        session = CompetitionSession(clock=lambda: now[0])
        session.start()
        with self.assertRaises(RuntimeError):
            session.record("bonus", _result("blind_slot"))
        for _ in range(2):
            session.record("l_block", _result("l_block"))
            session.record("cylinder", _result("cylinder"))
        session.record("bonus", _result("blind_slot"))
        self.assertEqual(session.base_completed, 4)
        self.assertEqual(session.bonus_completed, 1)

    def test_time_limit_is_enforced(self) -> None:
        now = [0.0]
        session = CompetitionSession(total_seconds=10.0, clock=lambda: now[0])
        session.start()
        now[0] = 11.0
        with self.assertRaises(RuntimeError):
            session.record("l_block", _result("l_block"))


if __name__ == "__main__":
    unittest.main()
