''' 
This file will contain the setup script for performing TIS using the AIMMD learned committor model. 
'''
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import openpathsampling as paths
import aimmd 
import torch
import sys
import simtk.unit as unit
import openpathsampling.engines.toy as toys
from functools import reduce
from .Tools import read_config, interface_indicator
from pathlib import Path
from uuid import uuid4
import openpathsampling.engines.openmm as ops_openmm
from openpathsampling.experimental.storage import monkey_patch_all
from openpathsampling.experimental.storage.collective_variables import MDTrajFunctionCV
from openpathsampling.experimental.storage import Storage
import os 
# Monkey patch OpenPathSampling to use experimental features
import pickle
'''

What is required for TIS_AIMMD runs

* The AIMMD model (through the storage)

'''


def _coordinates_descriptor_transform(snapshot):
    return snapshot.coordinates[0]


def _predict_with_nnet(descriptors, nnet, batch_size):
    # A single snapshot gives a 1-D descriptor vector; networks with BatchNorm need a
    # (batch, features) input, so always pass at least a batch of one.
    descriptors = np.atleast_2d(np.asarray(descriptors))
    device = next(nnet.parameters()).device
    dtype = next(nnet.parameters()).dtype
    n_split = (descriptors.shape[0] // batch_size) + 1
    predictions = []
    was_training = nnet.training
    nnet.eval()
    with torch.no_grad():
        for descript_part in np.array_split(descriptors, n_split):
            descript_part = torch.as_tensor(descript_part, device=device, dtype=dtype)
            pred = nnet(descript_part).cpu().numpy()
            predictions.append(pred)
    nnet.train(was_training)
    return np.concatenate(predictions, axis=0)


# Unpickled networks keyed by file path, so the committor CV does not
# reload the network from disk for every snapshot.
_NNET_CACHE = {}


def _load_cached_nnet(nnet_location):
    nnet = _NNET_CACHE.get(nnet_location)
    if nnet is None:
        with open(nnet_location, "rb") as file_handle:
            nnet = pickle.load(file_handle)
        nnet.eval()
        _NNET_CACHE[nnet_location] = nnet
    return nnet


def _q_committor_pkl(snapshot, nnet_location, descriptor_transform, batch_size):
    descriptors = descriptor_transform(snapshot)
    nnet = _load_cached_nnet(nnet_location)
    pred = _predict_with_nnet(descriptors, nnet, batch_size)
    return float(np.asarray(pred).reshape(-1)[0])


def _q_committor_nnet(snapshot, nnet, descriptor_transform, batch_size):
    descriptors = descriptor_transform(snapshot)
    pred = _predict_with_nnet(descriptors, nnet, batch_size)
    return float(np.asarray(pred).reshape(-1)[0])


def _q_committor_temp(snapshot, descriptor_transform, weights, bias):
    descriptors = descriptor_transform(snapshot)
    pred = np.dot(descriptors, weights[:len(descriptors)]) + bias
    return float(np.asarray(pred).reshape(-1)[0])


IN_STATE_LOW = -1e10   # forward: frames in A sit below every interface
IN_STATE_HIGH = 1e10   # backward: frames in B sit above every interface


def _cv_with_state(snapshot, cv, state, in_state_value):
    """Order parameter that equals ``in_state_value`` inside ``state``
    and ``cv(snapshot)`` everywhere else."""
    if state(snapshot):
        return in_state_value
    return cv(snapshot)


def masked_extrema(values, in_state, in_state_value):
    """(min, max) of ``values`` over frames outside the origin state.

    This is the extremum of the interface order parameter built by
    :meth:`AIMMD_TIS.create_interface_set` (``in_state_value`` inside the
    state, the CV value elsewhere), so a path only counts as crossing an
    interface after it has left its initial state. A path that never leaves
    the state gets ``in_state_value`` for both.
    """
    outside = ~np.asarray(in_state, dtype=bool)
    if not outside.any():
        return float(in_state_value), float(in_state_value)
    values = np.asarray(values, dtype=float).reshape(-1)[outside]
    return float(values.min()), float(values.max())


class AIMMD_TIS:
    def __init__(
        self,
        engine,
        AIMMD_model,
        stateA,
        stateB,
        descriptor_transform=None,
        directory="",
        use_transform=False,
        model_pkl=True,
        cvq_type="model",
        cv_label="cv_q",
    ):
        self.engine = engine
        self.model = AIMMD_model
        self.stateA = stateA
        self.stateB = stateB
        self.use_transform = use_transform
        self.model_pkl = model_pkl
        self.cv_label = self._sanitize_label(cv_label) or "cv_q"
        self.directory = Path(directory) if directory else Path.cwd()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._uses_model_pickle = False
        self._cv_q_name = f"cv_q_{uuid4().hex}"
        self._model_pickle_path = self.directory / f"{self._cv_q_name}.pkl"
        if descriptor_transform==None:
            descriptor_transform = paths.FunctionCV(
                'descriptor_transform',
                _coordinates_descriptor_transform,
                cv_wrap_numpy_array=True,
            ).with_diskcache()

        self.model.descriptor_transform = descriptor_transform
        
        self.cvq_type = cvq_type
        if self.cvq_type=="pkl":
            with open(self._model_pickle_path, "wb") as f:
                pickle.dump(self.model.nnet, f)
            self._uses_model_pickle = True
            self.cv_q = paths.CoordinateFunctionCV(
                name=self._cv_q_name,
                f=_q_committor_pkl,
                nnet_location=str(self._model_pickle_path),
                descriptor_transform=descriptor_transform,
                batch_size=4096,
            ).with_diskcache()
        elif self.cvq_type=="nnet":
            self.cv_q = paths.CoordinateFunctionCV(
                name=self._cv_q_name,
                f=_q_committor_nnet,
                nnet=self.model.nnet,
                descriptor_transform=descriptor_transform,
                batch_size=4096,
            ).with_diskcache()
        elif self.cvq_type=="model":
            with open(self._model_pickle_path, "wb") as f:
                pickle.dump(self.model.nnet, f)
            self._uses_model_pickle = True
            self.cv_q = paths.CoordinateFunctionCV(
                name=self._cv_q_name,
                f=_q_committor_pkl,
                nnet_location=str(self._model_pickle_path),
                descriptor_transform=descriptor_transform,
                batch_size=4096,
            ).with_diskcache()
        elif self.cvq_type=="temp":
            weights = np.random.RandomState(42).randn(100)
            weights[0:5] = np.array([3, -0.1, 0.1, 0.1, 1.5]) * 3
            self.cv_q = paths.CoordinateFunctionCV(
                name=self._cv_q_name,
                f=_q_committor_temp,
                descriptor_transform=descriptor_transform,
                weights=weights,
                bias=-6,
            ).with_diskcache()
        else:
            raise ValueError("Incorrect cvq_type given as argument, choose from either 'pkl','nnet','model' ")

    @staticmethod
    def _sanitize_label(value):
        if value is None:
            return None
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
        cleaned = "".join(ch if ch in allowed else "_" for ch in str(value))
        while "__" in cleaned:
            cleaned = cleaned.replace("__", "_")
        return cleaned.strip("_")

    def _make_interface_storage(self, interface_value, template, direction="forward", overwrite=True, iteration=None):
        interface_ind = interface_indicator(interface_value)
        if iteration is None:
            file_name = f"aimmd_tis_storage_{direction}_i{interface_ind}.nc"
        else:
            file_name = f"aimmd_tis_storage_it{int(iteration)}_{direction}_i{interface_ind}.nc"
        storage_path = self.directory / file_name
        if overwrite and storage_path.exists():
            storage_path.unlink()
        storage = paths.Storage(str(storage_path), "w", template=template)
        storage.save(template)
        storage.save(self.engine)
        return storage, storage_path

    def create_interface_set(self, interface_value, direction="forward"):
        """
        Create an interface set for TIS.

        The interface volume is the union of the initial state and the
        region on the state side of ``interface_value``:

            forward:  stateA ∪ {cv_q ≤ interface_value}
            backward: stateB ∪ {cv_q ≥ interface_value}

        This is built from a modified order parameter that takes a
        sentinel value inside the state, so a path counts as crossing
        the interface only once it has left the state *and* passed
        ``interface_value``.

        Parameters
        ----------
        interface_value : float
            Value of cv_q at the interface.
        direction : str, optional
            Direction of crossing ('forward' or 'backward'). Default is 'forward'.

        Returns
        -------
        VolumeInterfaceSet
            Interface set for TIS simulation.
        """
        if direction == "forward":
            lambda_cv = paths.FunctionCV(
                f"{self.cv_q.name}_with_stateA",
                _cv_with_state,
                cv=self.cv_q,
                state=self.stateA,
                in_state_value=IN_STATE_LOW,
            )
            return paths.VolumeInterfaceSet(
                lambda_cv,
                minvals=-float("inf"),
                maxvals=interface_value,
            )

        elif direction == "backward":
            lambda_cv = paths.FunctionCV(
                f"{self.cv_q.name}_with_stateB",
                _cv_with_state,
                cv=self.cv_q,
                state=self.stateB,
                in_state_value=IN_STATE_HIGH,
            )
            return paths.VolumeInterfaceSet(
                lambda_cv,
                minvals=interface_value,
                maxvals=float("inf"),
            )

        else:
            raise ValueError(f"direction must be 'forward' or 'backward', got {direction!r}")

    @staticmethod
    def _as_scalar(value):
        value_array = np.asarray(value)
        return float(value_array.reshape(-1)[0])

    def _merge_cvs_dict(self, cvs_dict=None):
        merged = {self.cv_label: self.cv_q}
        if cvs_dict is not None:
            merged.update(cvs_dict)
        return merged

    def _origin_state(self, direction):
        """Initial state of the interface ensemble and its order-parameter sentinel."""
        if direction == "forward":
            return self.stateA, IN_STATE_LOW
        if direction == "backward":
            return self.stateB, IN_STATE_HIGH
        raise ValueError(f"direction must be 'forward' or 'backward', got {direction!r}")

    def compute_cv_extrema(self, storage, cvs_dict=None, n_thermalize=0, direction=None):
        """Per-path CV extrema from a stored TIS run (reads every snapshot).

        With ``direction`` given, frames inside the ensemble's initial state are
        excluded, consistent with the interface definition (see
        :func:`masked_extrema`). ``direction=None`` gives raw extrema.
        """
        cvs = self._merge_cvs_dict(cvs_dict)
        data = {"step": []}

        for cv_name in cvs:
            data[f"min_{cv_name}"] = []
            data[f"max_{cv_name}"] = []

        origin_state, sentinel = self._origin_state(direction) if direction else (None, None)
        for i, step in enumerate(storage.steps[n_thermalize:]):
            step_index = i + n_thermalize
            traj = step.active[0].trajectory
            data["step"].append(step_index)
            if origin_state is not None:
                in_state = [bool(origin_state(snapshot)) for snapshot in traj]
            else:
                in_state = np.zeros(len(traj), dtype=bool)

            for cv_name, cv_func in cvs.items():
                values = [self._as_scalar(cv_func(snapshot)) for snapshot in traj]
                vmin, vmax = masked_extrema(values, in_state, sentinel)
                data[f"min_{cv_name}"].append(vmin)
                data[f"max_{cv_name}"].append(vmax)

        return pd.DataFrame(data)

    def _q_values(self, traj, descriptors):
        """cv_q along a trajectory in one batched network pass."""
        if self.cvq_type == "temp":
            return np.array([self._as_scalar(self.cv_q(snapshot)) for snapshot in traj])
        nnet = _load_cached_nnet(str(self._model_pickle_path)) if self._uses_model_pickle else self.model.nnet
        return np.asarray(_predict_with_nnet(descriptors, nnet, 4096)).reshape(-1)

    def _path_extrema_row(self, step_index, traj, descriptors, in_state, sentinel, cvs_dict=None):
        """One extrema-CSV row for a path, with frames in the origin state masked."""
        row = {"step": int(step_index)}
        vmin, vmax = masked_extrema(self._q_values(traj, descriptors), in_state, sentinel)
        row[f"min_{self.cv_label}"], row[f"max_{self.cv_label}"] = vmin, vmax
        for cv_name, cv_func in (cvs_dict or {}).items():
            values = [self._as_scalar(cv_func(snapshot)) for snapshot in traj]
            row[f"min_{cv_name}"], row[f"max_{cv_name}"] = masked_extrema(values, in_state, sentinel)
        return row

    def store_max_min_q(
        self,
        storage,
        direction,
        interface_value,
        directory: Path ="",
        cvs_dict=None,
        n_thermalize=0,
        origin_label=None,
        iteration=None,
        extrema_df=None,
    ):
        """Write the per-path extrema CSV for one interface ensemble.

        ``extrema_df`` (from the on-the-fly recording in :meth:`run_TIS`) is
        written as is; otherwise the extrema are computed from ``storage``.
        """
        interface_ind = interface_indicator(interface_value)
        output_dir = Path(directory) if directory else self.directory
        output_dir.mkdir(parents=True, exist_ok=True)

        if extrema_df is not None:
            cv_extrema_df = extrema_df[extrema_df["step"] >= int(n_thermalize)].reset_index(drop=True)
        else:
            cv_extrema_df = self.compute_cv_extrema(
                storage=storage,
                cvs_dict=cvs_dict,
                n_thermalize=n_thermalize,
                direction=direction,
            )
        # External filenames should be stable and human-readable.
        # Do not fall back to the internal CV UUID-style name.
        safe_origin = self._sanitize_label(origin_label) or self.cv_label

        name_parts = ["cv_extrema"]
        if safe_origin:
            name_parts.append(f"origin-{safe_origin}")
        if iteration is not None:
            name_parts.append(f"it{int(iteration)}")
        name_parts.append(f"int{interface_ind}")
        name_parts.append(direction)
        csv_path = output_dir / ("_".join(name_parts) + ".csv")
        cv_extrema_df.to_csv(csv_path, index=False)

        # Also emit legacy-style CSV name for simple single-iteration workflows.
        legacy_csv_path = output_dir / f"cv_extrema_int{interface_ind}_{direction}.csv"
        if str(legacy_csv_path) != str(csv_path):
            cv_extrema_df.to_csv(legacy_csv_path, index=False)

        # Backward compatibility with previous npy-based q extrema file.
        if "min_cv_q" in cv_extrema_df.columns and "max_cv_q" in cv_extrema_df.columns:
            legacy_path = output_dir / f"max_min_q_int{interface_ind}_{direction}.npy"
            np.save(legacy_path, cv_extrema_df[["min_cv_q", "max_cv_q"]].to_numpy())

        return csv_path


    def run_TIS_sequentially(
                       self,
                       n_mc_steps,
                       initial_path,
                       interface_values,
                       template,
                       scheme_move="TwoWay",
                       scheme_selector="Uniform",
                       scheme_modifier="RandomVelocities",
                       gaussian_parameter_width=0.5,
                       gaussian_parameter_shift=0.2,
                       direction="forward",
                       directory=None,
                       overwrite=True,
                       cvs_dict=None,
                       n_thermalize=0,
                       origin_label=None,
                       iteration=None,
                       descriptor_storage_path=None,
                       descriptor_set_name=None,
                       descriptor_name="descriptors",
                       descriptor_flush_every=1,
                       extrema_source="on_the_fly"):
        
        if directory is not None:
            self.directory = Path(directory)
            self.directory.mkdir(parents=True, exist_ok=True)
        if interface_values is None or len(interface_values)==0:
            raise ValueError("Provide a list of interface values to run TIS sequentially")
        run_summaries = []
        for interface_value in interface_values:
            print(f"Running AIMMD-TIS for interface {interface_value}...")
            storage, storage_path = self._make_interface_storage(
                interface_value=interface_value,
                template=template,
                direction=direction,
                overwrite=overwrite,
                iteration=iteration,
            )
            try:
                run_summary = self.run_TIS(
                    n_mc_steps=n_mc_steps,
                    storage=storage,
                    initial_path=initial_path,
                    interface_value=interface_value,
                    scheme_move=scheme_move,
                    scheme_selector=scheme_selector,
                    scheme_modifier=scheme_modifier,
                    gaussian_parameter_width=gaussian_parameter_width,
                    gaussian_parameter_shift=gaussian_parameter_shift,
                    direction=direction,
                    directory=self.directory,
                    cvs_dict=cvs_dict,
                    n_thermalize=n_thermalize,
                    origin_label=origin_label,
                    iteration=iteration,
                    descriptor_storage_path=descriptor_storage_path,
                    descriptor_set_name=descriptor_set_name,
                    descriptor_name=descriptor_name,
                    descriptor_flush_every=descriptor_flush_every,
                    extrema_source=extrema_source,
                )
                run_summary["storage_path"] = str(storage_path)
                run_summaries.append(run_summary)
            finally:
                storage.close()
            print(f"Finished interface {interface_value}. Storage: {storage_path}")

        if self._uses_model_pickle and self._model_pickle_path.exists() and self.cvq_type in ["pkl", "model"]:
            self._model_pickle_path.unlink()

        return run_summaries

    def run_single_TIS(self,
                       n_mc_steps,
                       storage,
                       initial_path,
                       interface_value,
                       scheme_move="TwoWay",
                       scheme_selector="Uniform",
                       scheme_modifier="RandomVelocities",
                       gaussian_parameter_width=0.5,
                       gaussian_parameter_shift=0.2,
                       direction="forward",
                       directory=Path.cwd(),
                       cvs_dict=None,
                       n_thermalize=0,
                       origin_label=None,
                       iteration=None,
                       descriptor_storage_path=None,
                       descriptor_set_name=None,
                       descriptor_name="descriptors",
                       descriptor_flush_every=1,
                       extrema_source="on_the_fly"):
        return self.run_TIS(
            n_mc_steps=n_mc_steps,
            storage=storage,
            initial_path=initial_path,
            interface_value=interface_value,
            scheme_move=scheme_move,
            scheme_selector=scheme_selector,
            scheme_modifier=scheme_modifier,
            gaussian_parameter_width=gaussian_parameter_width,
            gaussian_parameter_shift=gaussian_parameter_shift,
            direction=direction,
            directory=directory,
            cvs_dict=cvs_dict,
            n_thermalize=n_thermalize,
            origin_label=origin_label,
            iteration=iteration,
            descriptor_storage_path=descriptor_storage_path,
            descriptor_set_name=descriptor_set_name,
            descriptor_name=descriptor_name,
            descriptor_flush_every=descriptor_flush_every,
            extrema_source=extrema_source,
        )

    def run_TIS(self, 
                n_mc_steps, 
                storage, 
                initial_path, 
                interface_value, 
                scheme_move="TwoWay", 
                scheme_selector="Uniform", 
                scheme_modifier="RandomVelocities",
                gaussian_parameter_width=0.5, 
                gaussian_parameter_shift=0.2,
                direction="forward", 
                directory=Path.cwd(),
                cvs_dict=None,
                n_thermalize=0,
                origin_label=None,
                iteration=None,
                descriptor_storage_path=None,
                descriptor_set_name=None,
                descriptor_name="descriptors",
                descriptor_flush_every=1,
                log_diagnostics=False,
                extrema_source="on_the_fly",
                ):
        """Run TIS for one interface and write the per-path extrema CSV.

        ``extrema_source``:
          * ``"on_the_fly"`` (default): extrema of the interface order parameter
            are recorded from each new path while sampling, using one batched
            network pass per path, so the storage is not read again afterwards.
          * ``"storage"``: computed after sampling by reading every snapshot back
            from the OPS storage.

        Both exclude frames inside the ensemble's initial state, consistent with
        the interface definition. With ``descriptor_storage_path`` the path
        descriptors and shot results are stored per MC step, storage index = MC
        step (the initial path is step 0).
        """
        if extrema_source not in ("on_the_fly", "storage"):
            raise ValueError(f"extrema_source must be 'on_the_fly' or 'storage', got {extrema_source!r}")


        print("Creating network...")
        paths.InterfaceSet._reset()
        # Interface = initial state ∪ {cv_q on the state side of interface_value}
        interface_set = self.create_interface_set(interface_value, direction)
        if direction=="forward":
            # Create a MISTIS network for transitions: A -> Interface -> B
            network = paths.MISTISNetwork([(self.stateA, interface_set, self.stateB)]).named('mstis')
        else:
            # Create a MISTIS network for transitions: B -> Interface -> A
            network = paths.MISTISNetwork([(self.stateB, interface_set, self.stateA)]).named('mstis')


        if hasattr(storage, "engines"):
            engine = storage.engines[-1]
            if scheme_selector=="Uniform":
                selector = None #this will give a uniform selector
            elif scheme_selector=="Gaussian":
                if direction=="forward":
                    # gaussian selector with given parameter width and shift from the interface (in front of the interface by gaussian parameter shift) 
                    #therefore - for forward and + for backward in shift.
                    selector = paths.GaussianBiasSelector(self.cv_q, alpha=1/(2*gaussian_parameter_width**2), l_0=interface_value-gaussian_parameter_shift)
                elif direction=="backward":
                    selector = paths.GaussianBiasSelector(self.cv_q, alpha=1/(2*gaussian_parameter_width**2), l_0=interface_value+gaussian_parameter_shift)
            elif scheme_selector == "InterfaceConstrained":
                selector = paths.InterfaceConstraiendaSelector(interface_value)                                                                                    
            else: 
                raise ImportError("Incorrect selector type given choose from either: Uniform, Gaussian, Interface_constrained ")
            modifier_method= None
            if scheme_modifier==None or scheme_modifier=="NoModifier":
                modifier_method = None
            elif scheme_modifier=="RandomVelocities":
                # velocity randomizer setup
                print(engine)
                if hasattr(engine, 'integrator') and callable(getattr(engine.integrator, 'getTemperature', None)):
                    # If the engine is compatible with OpenMM's structure
                    beta = 1 / (engine.integrator.getTemperature() * unit.BOLTZMANN_CONSTANT_kB)
                elif isinstance(engine, toys.Engine):
                    # If the engine is an ops toy engine
                    beta = engine.options["integ"].beta
                else:
                    raise AttributeError("Engine type not supported for beta calculation.")
                # velocity randomizer setup
                modifier_method = paths.RandomVelocities(beta=beta, engine=engine)       

            else:
                raise ImportError("Incorrect modifier type given. Choose from either: 'NoModifier' or 'RandomVelocities'")                           

            if scheme_move=="OneWay":
                move_scheme = paths.OneWayShootingMoveScheme(network, 
                                            selector= selector,
                                            engine = engine).named("scheme")
            elif scheme_move=="TwoWay":
                # create the sampling move scheme, i.e. the recipe on how to generate new TP trials from previous ones in the MC chain
                move_scheme = paths.MoveScheme(network=network)
                tw_strategy = paths.strategies.TwoWayShootingStrategy(modifier=modifier_method, selector=selector, engine=engine, group='TwoWayShooting')
                move_scheme.append(tw_strategy)
                move_scheme.append(paths.strategies.OrganizeByMoveGroupStrategy())
                move_scheme.build_move_decision_tree()
            else:
                raise ValueError("Incorrect move scheme given. Choose from either: 'OneWay' or 'TwoWay'")

            initial_conditions = move_scheme.initial_conditions_from_trajectories(initial_path)
            initial_conditions.sanity_check()
            storage.save(initial_conditions)
            storage.save(network)
            storage.save(move_scheme)
            sampler = paths.PathSampling(storage=storage,
                                move_scheme=move_scheme,
                                sample_set=initial_conditions)
            storage.save(sampler)
            print("Runnng MISTIS at interface q={}".format(interface_value))
            print("Now performing MC steps: ",n_mc_steps)
            desc_storage = None
            next_storage_index = 0
            if descriptor_storage_path is not None:
                try:
                    from ops_setup.descriptor_storage import DescriptorStorage
                    from ops_setup.descriptor_storage import get_last_storage_index
                except Exception as exc:
                    raise RuntimeError(
                        "Descriptor storage requested but ops_setup.descriptor_storage is unavailable"
                    ) from exc

                descriptor_storage_path = Path(descriptor_storage_path)
                descriptor_storage_path.parent.mkdir(parents=True, exist_ok=True)
                descriptor_set_name = descriptor_set_name or direction
                next_storage_index = get_last_storage_index(
                    descriptor_storage_path,
                    descriptor_set_name,
                    float(interface_value),
                )
                mode = "a" if descriptor_storage_path.with_suffix(".meta.json").exists() else "w"
                desc_storage = DescriptorStorage(
                    descriptor_storage_path,
                    mode=mode,
                    method_type="TIS",
                )
                print(
                    "Descriptor storage enabled: {} (set='{}', start_index={})".format(
                        descriptor_storage_path,
                        descriptor_set_name,
                        next_storage_index,
                    )
                )

            def _fmt_duration(seconds):
                """Format seconds as d:HH:MM."""
                seconds = int(seconds)
                d, rem = divmod(seconds, 86400)
                h, rem = divmod(rem, 3600)
                m = rem // 60
                return f"{d}d {h:02d}:{m:02d}"

            origin_state, sentinel = self._origin_state(direction)
            on_the_fly = extrema_source == "on_the_fly"
            extrema_rows = []

            def _record_path(step_index, traj):
                """Extrema row and/or descriptor-storage entry for one path."""
                nonlocal next_storage_index
                if not on_the_fly and desc_storage is None:
                    return
                desc_arr = np.asarray(self.model.descriptor_transform(traj))
                if desc_arr.ndim == 1:
                    desc_arr = desc_arr.reshape(-1, 1)
                if on_the_fly:
                    in_state = np.array([bool(origin_state(snapshot)) for snapshot in traj])
                    extrema_rows.append(self._path_extrema_row(
                        step_index, traj, desc_arr, in_state, sentinel, cvs_dict))
                if desc_storage is not None:
                    # Shot result per frame: state at both path ends, as in AIMMD/RPE trainsets.
                    ends = (traj[0], traj[-1])
                    shot = np.array([sum(bool(self.stateA(s)) for s in ends),
                                     sum(bool(self.stateB(s)) for s in ends)], dtype=float)
                    desc_storage.add_trajectory(
                        set_name=descriptor_set_name,
                        interface_value=float(interface_value),
                        descriptors={descriptor_name: desc_arr},
                        n_points=len(traj),
                        storage_index=next_storage_index,
                        shot_results=np.tile(shot, (len(traj), 1)),
                    )
                    next_storage_index += 1

            import time as _time
            _t_start = _time.monotonic()
            _step_times = []
            try:
                # Initial path = MC step 0, so CSV rows and descriptor-storage
                # indices both equal the OPS step number.
                if on_the_fly or next_storage_index == 0:
                    initial_traj = storage.steps[0].active[0].trajectory
                    if desc_storage is not None and next_storage_index != 0:
                        # Appending to an existing descriptor storage: keep its numbering.
                        saved, desc_storage = desc_storage, None
                        _record_path(0, initial_traj)
                        desc_storage = saved
                    else:
                        _record_path(0, initial_traj)
                for step_ind in range(int(n_mc_steps)):
                    _t_step_start = _time.monotonic()
                    sampler.run(1)
                    _step_elapsed = _time.monotonic() - _t_step_start
                    _step_times.append(_step_elapsed)
                    _avg_step = sum(_step_times) / len(_step_times)
                    _steps_done = step_ind + 1
                    _steps_left = int(n_mc_steps) - _steps_done
                    _elapsed_total = _time.monotonic() - _t_start
                    _eta = _avg_step * _steps_left
                    print(
                        f"  step {_steps_done}/{int(n_mc_steps)} | "
                        f"step={_step_elapsed:.1f}s avg={_avg_step:.1f}s | "
                        f"elapsed={_fmt_duration(_elapsed_total)} "
                        f"ETA={_fmt_duration(_eta)}",
                        flush=True,
                    )
                    _record_path(step_ind + 1, storage.steps[-1].active[0].trajectory)

                    if desc_storage is not None and descriptor_flush_every is not None and descriptor_flush_every > 0:
                        if ((step_ind + 1) % int(descriptor_flush_every)) == 0:
                            desc_storage.flush()
            finally:
                if desc_storage is not None:
                    desc_storage.close()

            print('snapshots:', len(storage.snapshots))
            print('trajectories:', len(storage.trajectories))
            print('samples:', len(storage.samples))
            print(f"storing path extrema ({extrema_source})")
            extrema_csv_path = self.store_max_min_q(
                storage=storage,
                direction=direction,
                interface_value=interface_value,
                directory=directory,
                cvs_dict=cvs_dict,
                n_thermalize=n_thermalize,
                origin_label=origin_label,
                iteration=iteration,
                extrema_df=pd.DataFrame(extrema_rows) if on_the_fly else None,
            )
            if log_diagnostics:
                print(move_scheme.move_summary(storage.steps))
                for i, step in enumerate(storage.steps[:]):
                    print(f"Step {i+1}: {step.change.canonical.details}")


            stable_origin = self._sanitize_label(origin_label) or self.cv_label
            return {
                "interface_value": float(interface_value),
                "direction": direction,
                "iteration": int(iteration) if iteration is not None else None,
                "origin_label": stable_origin,
                "n_mc_steps": int(n_mc_steps),
                "n_snapshots": int(len(storage.snapshots)),
                "n_trajectories": int(len(storage.trajectories)),
                "n_samples": int(len(storage.samples)),
                "cv_extrema_file": str(extrema_csv_path),
                # Ready-to-use descriptor for rpe_mbar
                "interface_spec": {
                    "filename": stable_origin,
                    "cv_column_base": self.cv_label,
                    "iteration": int(iteration) if iteration is not None else None,
                },
            }

        else :
            raise InterruptedError("Storage does not contain an engine")