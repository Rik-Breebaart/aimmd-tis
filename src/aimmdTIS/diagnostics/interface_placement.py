"""Interface placement diagnostics utilities.

This module centralizes interface placement logic that was previously in
``aimmdTIS.Tools`` and extends it to support stable-state OPS storage input.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def ceil_decimal(x, decimals=1):
    multiply = 10 ** decimals
    return np.ceil(multiply * x) / multiply


def floor_decimal(x, decimals=1):
    multiply = 10 ** decimals
    return np.floor(multiply * x) / multiply


def interfaces_q_space(q_0, overlap, direction="forward"):
    interfaces = [q_0]
    if direction == "forward":
        s = -1
    elif direction == "backward":
        s = 1
    else:
        raise ValueError("Invalid direction")
    while np.isnan(interfaces[-1]) == 0:
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)  # Suppress warnings from log of negative numbers when interfaces go out of bounds
        interfaces.append(round(s * np.log(overlap * (1 + np.exp(s * interfaces[-1])) - 1), 2))
    return np.array(interfaces[:-1])


def _as_1d_q(values):
    q = np.asarray(values)
    if q.ndim == 2 and q.shape[1] == 1:
        return q[:, 0]
    return q.reshape(-1)


def _evaluate_q_from_descriptors(model, descriptors):
    q_total = model.log_prob(descriptors, use_transform=False, batch_size=None)
    return _as_1d_q(q_total)


def _compute_min_max_from_q(q_total, stable_states, in_state):
    min_max_stable_q = np.zeros((len(stable_states), 2))
    for state in range(len(stable_states)):
        state_mask = in_state[:, state].astype(bool)
        if np.sum(state_mask) == 0:
            min_max_stable_q[state, :] = np.nan
            continue
        min_max_stable_q[state, 0] = np.min(q_total[state_mask])
        min_max_stable_q[state, 1] = np.max(q_total[state_mask])
        print(f"stable_state {stable_states[state]} data falls in q-range {min_max_stable_q[state]}")
    return min_max_stable_q


def _build_interface_result(min_max_stable_q, overlap=0.2):
    if np.sum(np.isnan(min_max_stable_q)) == 0:
        forward_interfaces = interfaces_q_space(
            ceil_decimal(min_max_stable_q[0, 1], decimals=2),
            overlap,
            direction="forward",
        )
        backward_interfaces = interfaces_q_space(
            floor_decimal(min_max_stable_q[1, 0], decimals=2),
            overlap,
            direction="backward",
        )
        print("Forward interfaces: {}".format(" ".join(map(str, forward_interfaces.flatten()))))
        print("Backward interfaces: {}".format(" ".join(map(str, backward_interfaces.flatten()))))
        return forward_interfaces, backward_interfaces
    return None, None


def _iter_stable_trajectories(traj, state_volume):
    len_traj = len(traj)
    if state_volume is not None:
        in_state_mask = np.array([bool(state_volume(snapshot)) for snapshot in traj], dtype=bool)
    else:
        in_state_mask = np.ones(len_traj, dtype=bool)

    if len(in_state_mask) != len_traj:
        raise ValueError(
            f"State mask length mismatch: "
            f"mask={len(in_state_mask)} q={len_traj}"
        )
    return in_state_mask

def _infer_state_volume(stable_states, stable_state_volumes, idx):
    if stable_state_volumes is not None:
        return stable_state_volumes[idx]
    if stable_states is not None and callable(stable_states[idx]):
        return stable_states[idx]
    return None
   


def check_interfaces(model, stable_states, descriptors, weights=None, shot_results=None, in_state=None, overlap=0.2):
    """Compute forward/backward interfaces from descriptor arrays.

    This keeps the historical API used in older scripts while delegating
    implementation to diagnostics utilities.
    """
    q_total = _evaluate_q_from_descriptors(model, descriptors)
    if in_state is None:
        if weights is not None and shot_results is not None:
            in_state = np.zeros_like(shot_results)
            for state in range(len(stable_states)):
                in_state[:, state] = (weights * shot_results[:, state]) > 0
        else:
            raise ValueError("Provide `in_state` or both `weights` and `shot_results` to calculate it.")

    min_max_stable_q = _compute_min_max_from_q(q_total, stable_states, in_state)
    return _build_interface_result(min_max_stable_q, overlap=overlap)


def check_interfaces_from_stable_storage(
    model,
    stable_storages: Sequence,
    stable_states: Optional[Sequence[str]] = None,
    stable_state_volumes: Optional[Sequence] = None,
    descriptor_transform=None,
    overlap: float = 0.2,
    n_thermalize: int = 0,
    max_steps: Optional[int] = None,
):
    """Compute interfaces directly from stable-state OPS storage.

    Parameters
    ----------
    model
        AIMMD model with ``log_prob``.
    stable_storages
        Sequence of OPS storage objects or paths, one per stable state.
    stable_states
        State labels; defaults to ``state_0, state_1, ...``. If entries are
        callables (OPS volumes), they are used as state-membership tests.
    stable_state_volumes
        Optional explicit sequence of OPS state volumes/callables, one per
        stable storage. Used to determine which configurations are inside each
        state.
    descriptor_transform
        Descriptor callable. If omitted, uses ``model.descriptor_transform``.
    overlap
        Target interface overlap.
    n_thermalize
        Number of MC steps skipped from the start of each storage.
    max_steps
        Optional cap on number of MC steps processed per storage.

    Returns
    -------
    dict
        Includes interfaces plus stable-state metadata.
    """
    if descriptor_transform is None:
        descriptor_transform = getattr(model, "descriptor_transform", None)
    if descriptor_transform is None:
        raise ValueError("descriptor_transform is required (either argument or model.descriptor_transform)")

    if stable_states is None:
        stable_states = [f"state_{i}" for i in range(len(stable_storages))]

    min_max_stable_q = np.full((len(stable_storages), 2), np.nan)
    min_max_stable_q_in_state = np.full((len(stable_storages), 2), np.nan)
    stable_metadata = []
    in_state_masks = {}
    for idx, storage_obj in enumerate(stable_storages):
        close_after = False
        storage = storage_obj
        if isinstance(storage_obj, (str, Path)):
            import openpathsampling as paths

            storage = paths.Storage(str(storage_obj), "r")
            close_after = True

        q_values_all = []
        q_values_in_state = []
        state_volume = _infer_state_volume(stable_states, stable_state_volumes, idx)

        n_frames_total = 0
        n_frames_in_state = 0
        state_label = stable_states[idx]
        if not isinstance(state_label, str):
            state_label = getattr(state_label, "name", f"state_{idx}")
        in_state_masks[state_label] = []


        for traj in storage.trajectories:
            descriptors = descriptor_transform(traj)
            q_values = _as_1d_q(model.log_prob(descriptors, use_transform=False, batch_size=None))
            q_values_all.append(q_values)
            n_frames_total += len(q_values)

            in_state_mask = _iter_stable_trajectories(traj, state_volume)
            in_state_masks[state_label].append(in_state_mask)
            if np.any(in_state_mask):
                q_values_in_state.append(q_values[in_state_mask])
                n_frames_in_state += int(np.sum(in_state_mask))

        if q_values_all:
            q_concat = np.concatenate(q_values_all)
            min_max_stable_q[idx, 0] = np.min(q_concat)
            min_max_stable_q[idx, 1] = np.max(q_concat)
        if q_values_in_state:
            q_concat_in_state = np.concatenate(q_values_in_state)
            min_max_stable_q_in_state[idx, 0] = np.min(q_concat_in_state)
            min_max_stable_q_in_state[idx, 1] = np.max(q_concat_in_state)


        print(
            f"stable_state {state_label} q-range(all)={min_max_stable_q[idx]} "
            f"q-range(in_state)={min_max_stable_q_in_state[idx]}"
        )

        stable_metadata.append(
            {
                "state": state_label,
                "n_frames_total": int(n_frames_total),
                "n_frames_in_state": int(n_frames_in_state),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        if close_after:
            storage.close()

    # Interface placement uses max(q_A in A) for forward and min(q_B in B) for backward.
    # This is encoded in _build_interface_result from per-state [min, max] rows.
    forward_interfaces, backward_interfaces = _build_interface_result(min_max_stable_q_in_state, overlap=overlap)
    return {
        "forward_interfaces": forward_interfaces,
        "backward_interfaces": backward_interfaces,
        "min_max_stable_q": min_max_stable_q,
        "min_max_stable_q_in_state": min_max_stable_q_in_state,
        "in_state_masks": in_state_masks,
        "stable_state_metadata": stable_metadata,

    }
