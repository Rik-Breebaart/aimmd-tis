''' 
This file will contain the setup script for performing TIS using the AIMMD learned committor model. 

The goal of this setup file is to be able to work with any toy model for now. 
Not yet looking into how to work on real systems, although the method should mostly be conversible to that case.

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
    device = next(nnet.parameters()).device
    dtype = next(nnet.parameters()).dtype
    n_split = (descriptors.shape[0] // batch_size) + 1
    predictions = []
    nnet.eval()
    with torch.no_grad():
        for descript_part in np.array_split(descriptors, n_split):
            descript_part = torch.as_tensor(descript_part, device=device, dtype=dtype)
            pred = nnet(descript_part).cpu().numpy()
            predictions.append(pred)
    nnet.train()
    return np.concatenate(predictions, axis=0)


def _q_committor_pkl(snapshot, nnet_location, descriptor_transform, batch_size):
    descriptors = descriptor_transform(snapshot)
    with open(nnet_location, "rb") as file_handle:
        nnet = pickle.load(file_handle)
    return _predict_with_nnet(descriptors, nnet, batch_size)


def _q_committor_nnet(snapshot, nnet, descriptor_transform, batch_size):
    descriptors = descriptor_transform(snapshot)
    return _predict_with_nnet(descriptors, nnet, batch_size)


def _q_committor_temp(snapshot, descriptor_transform, weights, bias):
    descriptors = descriptor_transform(snapshot)
    return np.dot(descriptors, weights[:len(descriptors)]) + bias


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
            ValueError("Incorrect cvq_type given as argument, choose from either 'pkl','nnet','model' ")

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

    @staticmethod
    def _as_scalar(value):
        value_array = np.asarray(value)
        return float(value_array.reshape(-1)[0])

    def _merge_cvs_dict(self, cvs_dict=None):
        merged = {self.cv_label: self.cv_q}
        if cvs_dict is not None:
            merged.update(cvs_dict)
        return merged

    def compute_cv_extrema(self, storage, cvs_dict=None, n_thermalize=0):
        cvs = self._merge_cvs_dict(cvs_dict)
        data = {"step": []}

        for cv_name in cvs:
            data[f"min_{cv_name}"] = []
            data[f"max_{cv_name}"] = []

        for i, step in enumerate(storage.steps[n_thermalize:]):
            step_index = i + n_thermalize
            traj = step.active[0].trajectory
            data["step"].append(step_index)

            for cv_name, cv_func in cvs.items():
                values = [self._as_scalar(cv_func(snapshot)) for snapshot in traj]
                data[f"min_{cv_name}"].append(min(values))
                data[f"max_{cv_name}"].append(max(values))

        return pd.DataFrame(data)
    
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
    ):
        interface_ind = int(np.round(interface_value*100))
        output_dir = Path(directory) if directory else self.directory
        output_dir.mkdir(parents=True, exist_ok=True)

        cv_extrema_df = self.compute_cv_extrema(
            storage=storage,
            cvs_dict=cvs_dict,
            n_thermalize=n_thermalize,
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
                       iteration=None):
        
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
                       iteration=None):
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
                iteration=None):


        print("Creating network...")
        paths.InterfaceSet._reset()
        if direction=="forward":
            # Define the interface volume where cv_q <= interface_value
            InterfaceVolume = paths.VolumeInterfaceSet(self.cv_q, minvals=-float("inf"), maxvals=interface_value)

            UnionVolumeStateA = paths.UnionVolume(InterfaceVolume[0], self.stateA)

            # Refine the interface to exclude state A
            InterfaceOutStateVolume = paths.VolumeInterfaceSet(
                self.cv_q, 
                minvals=-float("inf"), 
                maxvals=interface_value, 
                intersect_with=UnionVolumeStateA
            )
            # Create a MISTIS network for transitions: A -> Interface -> B
            network = paths.MISTISNetwork([(self.stateA, InterfaceOutStateVolume, self.stateB)]).named('mstis')

        elif direction=="backward":
            # Define the interface volume where cv_q >= interface_value
            InterfaceVolume = paths.VolumeInterfaceSet(self.cv_q, minvals=interface_value, maxvals=float("inf"))

            # Combine the interface volume with state B
            UnionVolumeStateB = paths.UnionVolume(InterfaceVolume[0], self.stateB)

            # Refine the interface to exclude state B
            InterfaceOutStateVolume = paths.VolumeInterfaceSet(self.cv_q, minvals=interface_value, maxvals=float("inf"), intersect_with=UnionVolumeStateB) 

            # Create a MISTIS network for transitions: B -> Interface -> A
            network = paths.MISTISNetwork([(self.stateB, InterfaceOutStateVolume, self.stateA)]).named('mstis')

        else:
            NameError("incorrect direction method has been given, choose from either 'forward' or 'backward'")
        

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
                ImportError("Incorrect selector type given choose from either: Uniform, Gaussian, Interface_constrained ")
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
                ImportError("Incorrect modifier type given. Choose from either: 'NoModifier' or 'RandomVelocities'")                           

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
                ImportError("Incorrect move scheme given. Choose from either: 'OneWay' or 'TwoWay'")

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
            sampler.run(n_mc_steps)
            print('snapshots:', len(storage.snapshots))
            print('trajectories:', len(storage.trajectories))
            print('samples:', len(storage.samples))
            print("storing maximum q")
            extrema_csv_path = self.store_max_min_q(
                storage=storage,
                direction=direction,
                interface_value=interface_value,
                directory=directory,
                cvs_dict=cvs_dict,
                n_thermalize=n_thermalize,
                origin_label=origin_label,
                iteration=iteration,
            )

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
            InterruptedError("Storage does not contain an engine")