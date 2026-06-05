#!/usr/bin/env python3
"""Deterministic command scripts used for demo videos and public benchmark runs."""

from __future__ import annotations

from typing import Any

import numpy as np


PUBLIC_EPISODE_LABELS = (
    "straight_ramp",
    "left_oval_curve",
    "right_oval_curve",
    "curve_recovery",
)


def build_demo_segments(config: dict[str, Any]) -> list[list[float]]:
    """Return an easy-to-read command script for demo videos.

    The demo is intentionally human-readable:
    stand -> slow forward -> medium forward -> fast forward -> slow forward -> stand.
    """
    demo_cfg = config["demo_rollout"]
    if "segments" in demo_cfg and demo_cfg["segments"]:
        return [[float(x) for x in segment] for segment in demo_cfg["segments"]]
    return [
        [0.0, 0.0, 0.0],
        [0.20, 0.0, 0.0],
        [0.40, 0.0, 0.0],
        [0.60, 0.0, 0.0],
        [0.30, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]


def public_command_script(safe_ranges: dict[str, list[float]], episode_idx: int) -> list[list[float]]:
    """Return the deterministic command schedule for one public benchmark episode."""
    vx_min, vx_max = map(float, safe_ranges["vx"])
    vy_min, vy_max = map(float, safe_ranges["vy"])
    yaw_min, yaw_max = map(float, safe_ranges["yaw"])
    turn_radius = float(safe_ranges.get("turn_radius", 18.25))

    def oval_yaw(vx: float, sign: float = 1.0) -> float:
        yaw = float(np.clip(vx / turn_radius, 0.0, yaw_max))
        return float(np.clip(sign * yaw, yaw_min, yaw_max))

    slow = max(vx_min, 0.8)
    medium = min(vx_max, 2.4)
    curve_fast = min(vx_max, 3.0)

    scripts = [
        [
            [0.0, 0.0, 0.0],
            [slow, 0.0, 0.0],
            [0.70 * vx_max, 0.0, 0.0],
            [vx_max, 0.0, 0.0],
        ],
        [
            [slow, 0.0, oval_yaw(slow)],
            [medium, 0.0, oval_yaw(medium)],
            [curve_fast, 0.0, oval_yaw(curve_fast)],
            [slow, 0.0, 0.0],
        ],
        [
            [slow, 0.0, oval_yaw(slow, -1.0)],
            [medium, 0.0, oval_yaw(medium, -1.0)],
            [curve_fast, 0.0, oval_yaw(curve_fast, -1.0)],
            [slow, 0.0, 0.0],
        ],
        [
            [1.4, 0.0, 0.0],
            [1.6, vy_max, oval_yaw(1.6)],
            [1.2, vy_min, oval_yaw(1.2, -1.0)],
            [2.0, 0.0, oval_yaw(2.0)],
        ],
    ]
    return scripts[episode_idx % len(scripts)]


def public_command_episode_label(episode_idx: int) -> str:
    return PUBLIC_EPISODE_LABELS[episode_idx % len(PUBLIC_EPISODE_LABELS)]


def command_for_step(segments: list[list[float]], step_idx: int, total_steps: int) -> np.ndarray:
    """Convert a segment list into one command vector for the current step."""
    segment_length = max(1, total_steps // len(segments))
    segment_idx = min(len(segments) - 1, step_idx // segment_length)
    return np.asarray(segments[segment_idx], dtype=np.float32)


def seconds_to_steps(duration_seconds: float, ctrl_dt: float) -> int:
    return max(1, int(round(float(duration_seconds) / float(ctrl_dt))))
