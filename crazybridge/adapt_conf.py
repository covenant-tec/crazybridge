"""The Adaptive EKF Bellman policy file (``adapt_policies.conf``), as an object.

Mirrors :mod:`crazybridge.pid_conf`: parses the 18 integer action IDs (`0..7`)
defining the optimal 9-state MDP 2D policy and 9-state SMDP policy, pushes them
to the Crazyflie's ``ekfPolicy`` CRTP parameter group on connect without
recompiling firmware, and serializes them into ``metrics.json`` so every flight
run records the exact decision matrices it flew with.

Both 3x3 matrix rows (``0 0 0``) and 1-value-per-line layouts are accepted;
``#`` starts a comment anywhere on a line.
"""

from __future__ import annotations

import os
from pathlib import Path

PARAM_GROUP = "ekfPolicy"
MIN_ACTION_ID = 0
MAX_ACTION_ID = 7

STATES_2D = (
    "n_down", "n_eq", "n_up",
    "m_down", "m_eq", "m_up",
    "a_down", "a_eq", "a_up",
)

MDP_SUFFIXES = tuple(f"m_{s}" for s in STATES_2D)
SMDP_SUFFIXES = tuple(f"s_{s}" for s in STATES_2D)
ALL_SUFFIXES = MDP_SUFFIXES + SMDP_SUFFIXES
EXPECTED = len(ALL_SUFFIXES)  # 18 action IDs (9 MDP + 9 SMDP)


def default_adapt_conf_path() -> str:
    """Resolve ``adapt_policies.conf`` from package share or source tree."""
    try:
        from ament_index_python.packages import get_package_share_directory

        share_candidate = os.path.join(
            get_package_share_directory("crazybridge"),
            "config",
            "adapt_policies.conf",
        )
        if os.path.isfile(share_candidate):
            return share_candidate
    except Exception:
        pass

    src_candidate = Path(__file__).resolve().parent.parent / "config" / "adapt_policies.conf"
    return str(src_candidate)


class AdaptConf:
    """Parsed ``adapt_policies.conf`` (9 MDP actions + 9 SMDP actions in ``0..7``)."""

    __slots__ = ("values", "source_path")

    def __init__(self, values: list[int], source_path: str | None = None) -> None:
        if len(values) != EXPECTED:
            raise ValueError(
                f"expected {EXPECTED} action IDs (9 MDP + 9 SMDP), got {len(values)}"
            )
        parsed: list[int] = []
        for idx, raw in enumerate(values):
            act = int(raw)
            if act < MIN_ACTION_ID or act > MAX_ACTION_ID:
                raise ValueError(
                    f"action at index {idx} ({ALL_SUFFIXES[idx]})={act} "
                    f"out of range [{MIN_ACTION_ID}, {MAX_ACTION_ID}]"
                )
            parsed.append(act)
        self.values: tuple[int, ...] = tuple(parsed)
        self.source_path = source_path

    @classmethod
    def load(cls, path: str) -> "AdaptConf":
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        values: list[int] = []
        with open(path, encoding="utf-8") as fh:
            for lineno, raw_line in enumerate(fh, 1):
                body = raw_line.split("#", 1)[0].replace(",", " ").strip()
                if not body:
                    continue
                for token in body.split():
                    try:
                        values.append(int(token))
                    except ValueError as exc:
                        raise ValueError(
                            f"{path}:{lineno}: expected integer action ID (0..7), got {token!r}"
                        ) from exc
        if len(values) != EXPECTED:
            raise ValueError(
                f"{path}: expected {EXPECTED} integer action IDs "
                f"(9 MDP + 9 SMDP), got {len(values)}"
            )
        return cls(values, source_path=os.path.abspath(path))

    @property
    def mdp_policy(self) -> tuple[int, ...]:
        return self.values[:9]

    @property
    def smdp_policy(self) -> tuple[int, ...]:
        return self.values[9:18]

    def params(self) -> list[tuple[str, int]]:
        """``(ekfPolicy.<name>, action_id)`` pairs in TOC order."""
        return [
            (f"{PARAM_GROUP}.{suffix}", val)
            for suffix, val in zip(ALL_SUFFIXES, self.values)
        ]

    def as_dict(self) -> dict[str, dict[str, int]]:
        """Structured ``{'mdp': {STATE: act}, 'smdp': {STATE: act}}`` for JSON reports."""
        return {
            "mdp": {
                st.upper(): act for st, act in zip(STATES_2D, self.mdp_policy)
            },
            "smdp": {
                st.upper(): act for st, act in zip(STATES_2D, self.smdp_policy)
            },
        }

    def describe(self) -> list[str]:
        """Human-readable log lines showing both 3x3 decision matrices."""
        m = self.mdp_policy
        s = self.smdp_policy
        return [
            f"  MDP 2D [N/M/A x DOWN,EQ,UP]: [{m[0]},{m[1]},{m[2]}] [{m[3]},{m[4]},{m[5]}] [{m[6]},{m[7]},{m[8]}]",
            f"  SMDP   [N/M/A x DOWN,EQ,UP]: [{s[0]},{s[1]},{s[2]}] [{s[3]},{s[4]},{s[5]}] [{s[6]},{s[7]},{s[8]}]",
        ]
