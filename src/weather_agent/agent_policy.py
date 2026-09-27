from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from .episode_runner import DecisionAdapter, Submission, Turn
from .runtime_records import LLMAttemptRef


@dataclass(frozen=True)
class ScenarioStep:
    """One deterministic choice expressed only in public operation language."""

    semantic_key: str
    arguments: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.semantic_key:
            raise ValueError("scenario semantic_key is required")


ScenarioRoute = Callable[[Turn, int], ScenarioStep]


class ScenarioDecisionAdapter:
    """Deterministic Adapter that resolves semantic keys from each public Turn."""

    def __init__(self, route: Sequence[ScenarioStep] | ScenarioRoute) -> None:
        if not callable(route) and not route:
            raise ValueError("scenario route must not be empty")
        self._route = route
        self._index = 0
        self.turns: list[Turn] = []

    def choose(
        self,
        turn: Turn,
        *,
        record_attempt: Callable[[LLMAttemptRef], None],
    ) -> Submission:
        self.turns.append(turn)
        step = (
            self._route(turn, self._index)
            if callable(self._route)
            else self._sequence_step()
        )
        matches = tuple(
            choice
            for choice in turn.choices
            if choice.semantic_key == step.semantic_key
        )
        if len(matches) != 1:
            raise LookupError(
                f"expected one current operation for semantic key {step.semantic_key!r}"
            )
        self._index += 1
        return Submission(choice=matches[0].choice, arguments=dict(step.arguments))

    def _sequence_step(self) -> ScenarioStep:
        assert not callable(self._route)
        if self._index >= len(self._route):
            raise LookupError("scenario route is exhausted")
        return self._route[self._index]


__all__ = [
    "DecisionAdapter",
    "ScenarioDecisionAdapter",
    "ScenarioRoute",
    "ScenarioStep",
    "Submission",
    "Turn",
]
