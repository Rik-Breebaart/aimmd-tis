"""Path extrema from an on-the-fly descriptor storage.

:meth:`aimmdTIS.AIMMD_TIS.run_TIS` can write each path's descriptors to an
ops-setup ``DescriptorStorage`` while sampling. The functions here compute the
per-path extrema of any committor model from those stored arrays, without
reading the OPS storage again. This is how extrema of later models are added
to earlier iterations' extrema files for multi-set MBAR.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .TIS_AIMMD_setup import IN_STATE_HIGH, IN_STATE_LOW


def _open_descriptor_storage(storage):
    if isinstance(storage, (str, Path)):
        from ops_setup.descriptor_storage import DescriptorStorage
        return DescriptorStorage.load(storage, mode="r")
    return storage


def _interface_trajectories(storage, set_name, interface_value):
    interfaces = storage.sets[set_name].interfaces
    key = f"{float(interface_value):.2f}" if not isinstance(interface_value, str) else interface_value
    if key not in interfaces:
        raise KeyError(f"Interface {key!r} not in set {set_name!r}: {list(interfaces)}")
    return interfaces[key].trajectories


def extrema_from_descriptor_storage(
    storage,
    set_name: str,
    interface_value: Union[float, str],
    q_functions: Dict[str, Callable[[np.ndarray], np.ndarray]],
    descriptor_names: Sequence[str] = ("descriptors",),
    in_state_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    direction: Optional[str] = None,
) -> pd.DataFrame:
    """Per-path min/max of each ``q_functions`` entry for one interface ensemble.

    Parameters
    ----------
    storage
        ``DescriptorStorage`` or its base path.
    set_name, interface_value
        Set and interface the paths were stored under.
    q_functions
        ``{column_base: f(descriptors) -> values}``; each function is applied
        once to all stored frames of the interface (batched). Columns
        ``min_<column_base>`` and ``max_<column_base>`` are returned.
    in_state_fn
        ``f(descriptors) -> bool`` marking frames inside the ensemble's initial
        state. Those frames are excluded, consistent with the interface order
        parameter used by :meth:`AIMMD_TIS.create_interface_set`.
    direction
        ``"forward"``/``"backward"``: value used for a path that never leaves
        its initial state (the interface sentinel). Without it such paths get NaN.

    Returns
    -------
    DataFrame with ``step`` (the stored ``storage_index``, i.e. the MC step) and
    the min/max columns, one row per stored path, in storage order.
    """
    storage = _open_descriptor_storage(storage)
    trajectories = _interface_trajectories(storage, set_name, interface_value)
    if not trajectories:
        return pd.DataFrame({"step": []})

    arrays = [np.asarray(storage.descriptors[name]) for name in descriptor_names]
    arrays = [a.reshape(-1, 1) if a.ndim == 1 else a for a in arrays]
    desc = np.concatenate(
        [np.concatenate([a[t.descriptor_start:t.descriptor_end] for a in arrays], axis=1)
         for t in trajectories],
        axis=0,
    )
    lengths = np.array([t.n_points for t in trajectories])
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])

    outside = np.ones(len(desc), dtype=bool)
    if in_state_fn is not None:
        outside = ~np.asarray(in_state_fn(desc), dtype=bool).reshape(-1)
    any_outside = np.add.reduceat(outside.astype(int), offsets) > 0
    sentinel = {"forward": IN_STATE_LOW, "backward": IN_STATE_HIGH}.get(direction, np.nan)

    data = {"step": [int(t.storage_index) for t in trajectories]}
    for name, fn in q_functions.items():
        values = np.asarray(fn(desc), dtype=float).reshape(-1)
        vmax = np.maximum.reduceat(np.where(outside, values, -np.inf), offsets)
        vmin = np.minimum.reduceat(np.where(outside, values, np.inf), offsets)
        data[f"min_{name}"] = np.where(any_outside, vmin, sentinel)
        data[f"max_{name}"] = np.where(any_outside, vmax, sentinel)
    return pd.DataFrame(data)


def trajectories_from_descriptor_storage(
    storage,
    set_name: str,
    interface_value: Union[float, str],
    descriptor_names: Sequence[str] = ("descriptors",),
    start: int = 0,
):
    """Yield ``(storage_index, descriptors, shot_results)`` per stored path from ``start`` on.

    ``shot_results`` is ``None`` when the storage holds none.
    """
    storage = _open_descriptor_storage(storage)
    for idx, traj in enumerate(_interface_trajectories(storage, set_name, interface_value)):
        if idx < start:
            continue
        desc = storage.get_trajectory_descriptors(set_name, float(interface_value), idx, list(descriptor_names))
        shots = storage.get_trajectory_shot_results(set_name, float(interface_value), idx)
        yield int(traj.storage_index), desc, shots


def rpe_dataset_from_descriptor_storage(
    rows: Sequence[dict],
    set_name: str,
    n_skip: Union[int, Sequence[int]] = 0,
    descriptor_names: Sequence[str] = ("descriptors",),
    shot_results_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    dataset=None,
):
    """Build (or extend) an ``rpe_mbar.RpeDataset`` from TIS descriptor storages.

    The descriptor storages written by :meth:`AIMMD_TIS.run_TIS` already hold
    every path's descriptors and shot results (storage index = MC step), so the
    OPS ``.nc`` storages are not opened.

    Parameters
    ----------
    rows
        Run summaries of one interface set, in interface order. Each row needs
        ``"descriptor_storage"`` (path) and ``"interface_value"``.
    set_name
        Set name the paths were stored under (``descriptor_set_name`` in
        ``run_TIS``) and the set name in the dataset; it must equal the origin
        label used for the extrema files, so ``get_descriptors_and_weights``
        pairs paths and MBAR weights correctly.
    n_skip
        First MC step used per interface (int for all, or one per row). Use the
        same value as for the extrema loader. Step 0 is the initial path.
    descriptor_names
        Descriptor arrays to concatenate per frame.
    shot_results_fn
        ``f(descriptors) -> (n_frames, 2)``, only used when a storage was
        written without shot results.
    dataset
        Existing ``RpeDataset`` to append this set to (multi-set RPE).

    Returns
    -------
    rpe_mbar.RpeDataset
    """
    import rpe_mbar

    if isinstance(n_skip, (int, np.integer)):
        n_skip = [int(n_skip)] * len(rows)
    if len(n_skip) != len(rows):
        raise ValueError(f"n_skip has {len(n_skip)} entries for {len(rows)} rows")

    if dataset is None:
        dataset = rpe_mbar.RpeDataset(metadata={"n_thermalization": {}, "source": "descriptor_storage"})
    descriptors = [] if dataset.descriptors is None else [dataset.descriptors]
    shot_results = [] if dataset.shot_results is None else [dataset.shot_results]
    offset = 0 if dataset.descriptors is None else len(dataset.descriptors)

    set_data = rpe_mbar.SetData(set_name=str(set_name))
    for row, skip in zip(rows, n_skip):
        iface = float(row["interface_value"])
        iface_data = rpe_mbar.InterfaceData(interface_value=iface)
        rel = 0
        for step, desc, shots in trajectories_from_descriptor_storage(
            row["descriptor_storage"], set_name, iface, descriptor_names
        ):
            if step < int(skip):
                continue
            desc = np.asarray(desc)
            if desc.ndim == 1:
                desc = desc.reshape(-1, 1)
            if shots is None:
                if shot_results_fn is None:
                    raise ValueError(
                        f"{row['descriptor_storage']} holds no shot results; pass shot_results_fn"
                    )
                shots = shot_results_fn(desc)
            descriptors.append(desc)
            shot_results.append(np.asarray(shots, dtype=float))
            iface_data.trajectories.append(rpe_mbar.TrajectoryInfo(
                storage_index=int(step), relative_index=rel, n_points=len(desc),
                descriptor_start=offset, descriptor_end=offset + len(desc)))
            offset += len(desc)
            rel += 1
        if rel == 0:
            raise RuntimeError(f"{row['descriptor_storage']}: no stored paths from step {skip} on")
        set_data.interfaces[f"{iface}"] = iface_data
        dataset.metadata.setdefault("n_thermalization", {}).setdefault(str(set_name), {})[f"{iface}"] = int(skip)

    dataset.sets[str(set_name)] = set_data
    dataset.descriptors = np.concatenate(descriptors, axis=0)
    dataset.shot_results = np.concatenate(shot_results, axis=0)
    return dataset
