from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from time import monotonic

from .geometry import VolumeResult


@dataclass(frozen=True)
class CompetitionRecord:
    sequence: int
    target_key: str
    shape: str
    result: VolumeResult
    session_time_s: float
    locked: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "target_key": self.target_key,
            "shape": self.shape,
            "session_time_s": round(self.session_time_s, 6),
            "locked": self.locked,
            "result": self.result.to_dict(),
        }


@dataclass
class CompetitionSession:
    """Enforce the contest's result order, time budget and lock semantics."""

    total_seconds: float = 480.0
    base_targets: tuple[str, ...] = ("l_block", "cylinder")
    runs_per_target: int = 2
    bonus_runs: int = 2
    minimum_confidence: float = 0.65
    clock: Callable[[], float] = monotonic
    started_at: float | None = None
    records: list[CompetitionRecord] = field(default_factory=list)

    def start(self) -> None:
        if self.started_at is not None:
            raise RuntimeError("Session already started.")
        self.started_at = self.clock()

    def elapsed_s(self) -> float:
        if self.started_at is None:
            return 0.0
        return max(0.0, self.clock() - self.started_at)

    def remaining_s(self) -> float:
        return max(0.0, self.total_seconds - self.elapsed_s())

    @property
    def base_limit(self) -> int:
        return len(self.base_targets) * self.runs_per_target

    @property
    def base_completed(self) -> int:
        return min(
            len(
                [
                    record
                    for record in self.records
                    if record.target_key in self.base_targets
                ]
            ),
            self.base_limit,
        )

    @property
    def bonus_completed(self) -> int:
        return len([record for record in self.records if record.target_key == "bonus"])

    def record(
        self,
        target_key: str,
        result: VolumeResult,
        required_shape: str | None = None,
    ) -> CompetitionRecord:
        if self.started_at is None:
            raise RuntimeError("Start the session before measuring.")
        if self.remaining_s() <= 0:
            raise RuntimeError("The eight-minute measurement window has expired.")
        if result.has_errors:
            raise ValueError("A result with validation errors cannot be locked.")
        if result.confidence < self.minimum_confidence:
            raise ValueError(
                f"Result confidence {result.confidence:.3f} is below the "
                f"lock threshold {self.minimum_confidence:.3f}."
            )
        if required_shape is not None and result.shape != required_shape:
            raise ValueError(
                f"Shape mismatch: expected {required_shape}, got {result.shape}."
            )

        if target_key in self.base_targets:
            if self.base_completed >= self.base_limit:
                raise RuntimeError("All base measurements are already locked.")
            already_for_target = len(
                [record for record in self.records if record.target_key == target_key]
            )
            if already_for_target >= self.runs_per_target:
                raise RuntimeError(
                    f"Target {target_key} already has {self.runs_per_target} runs."
                )
        elif target_key == "bonus":
            if self.base_completed < self.base_limit:
                raise RuntimeError(
                    "Complete all base measurements before the optional blind slot."
                )
            if self.bonus_completed >= self.bonus_runs:
                raise RuntimeError("Both optional blind-slot runs are complete.")
        else:
            raise ValueError(f"Unknown target key: {target_key}")

        record = CompetitionRecord(
            sequence=len(self.records) + 1,
            target_key=target_key,
            shape=result.shape,
            result=result,
            session_time_s=self.elapsed_s(),
        )
        self.records.append(record)
        return record

    def summary(self) -> dict[str, object]:
        return {
            "started": self.started_at is not None,
            "elapsed_s": round(self.elapsed_s(), 6),
            "remaining_s": round(self.remaining_s(), 6),
            "base_completed": self.base_completed,
            "base_required": self.base_limit,
            "bonus_completed": self.bonus_completed,
            "bonus_required": self.bonus_runs,
            "base_complete": self.base_completed >= self.base_limit,
            "records": [record.to_dict() for record in self.records],
        }
