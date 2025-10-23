''' 
This file will contain the setup script for performing TIS using the AIMMD learned committor model. 

The goal of this setup file is to be able to work with any toy model for now. 
Not yet looking into how to work on real systems, although the method should mostly be conversible to that case.

'''
import numpy as np 
import matplotlib.pyplot as plt
import openpathsampling as paths
import aimmd 
import torch
import sys
import simtk.unit as unit
import openpathsampling.engines.toy as toys
from functools import reduce
from .Tools import read_config
from pathlib import Path
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

class AIMMD_TIS:
    def __init__(self, engine, AIMMD_model, stateA, stateB, descriptor_transform=None, directory="", use_transform = False, model_pkl=True):
        self.engine = engine
        self.model = AIMMD_model
        self.stateA = stateA
        self.stateB = stateB
        self.use_transform = use_transform
        self.model_pkl = model_pkl
        self.directory = directory
        if descriptor_transform==None:
            self.model.descriptor_transform = paths.FunctionCV('descriptor_transform', lambda s: s.coordinates[0], cv_wrap_numpy_array=True).with_diskcache()
        else :
            self.model.descriptor_transform = descriptor_transform
        
        if model_pkl:
            with open("model.pkl", "wb") as f:
                pickle.dump(self.model.nnet, f)
            self.cv_q=paths.CoordinateFunctionCV(name="cv_q", f=self.q_committor_pkl, nnet_location="model.pkl", descriptor_transform=descriptor_transform, batch_size=4096).with_diskcache()
        else:
            self.cv_q= paths.CoordinateFunctionCV(name="cv_q", f=self.q_committor, model=self.model, descriptor_transform=descriptor_transform).with_diskcache()  

    def q_committor_pkl(self, snapshot, nnet_location, descriptor_transform, batch_size):
        import torch 
        import numpy as np
        import pickle
        descriptors = descriptor_transform(snapshot)
        
        with open(nnet_location, "rb") as f:
            nnet = pickle.load(f)

        device = next(nnet.parameters()).device
        dtype = next(nnet.parameters()).dtype

        n_split = (descriptors.shape[0] // batch_size) + 1
        predictions = []
        nnet.eval()  # put model in evaluation mode
        # no gradient accumulation for predictions!
        with torch.no_grad():
            for descript_part in np.array_split(descriptors, n_split):
                descript_part = torch.as_tensor(descript_part, device=device,
                                                dtype=dtype)
                pred = nnet(descript_part).cpu().numpy()
                predictions.append(pred)
        return np.concatenate(predictions, axis=0)

    def q_committor(self, snapshot, model):
        import numpy as np
        pred = model.log_prob(snapshot,use_transform=True, batch_size=None)[:,0]
        return pred    
    
    def store_max_min_q(self, storage, direction, interface_value,  directory: Path =""):
        interface_indicator = int(np.round(interface_value*100))
        fileStore = Path(directory/ "max_min_q_int{}_{}".format(interface_indicator,direction))
        max_min_q = np.zeros((len(storage.steps),2))
        for j, step in enumerate(storage.steps):
            traj = step.active[0].trajectory
            q_values =np.array([self.cv_q(snapshot) for snapshot in traj ]).reshape(len(traj.snapshots))
            max_min_q[j,0] = np.min(q_values)
            max_min_q[j,1] = np.max(q_values)
        np.save(fileStore, max_min_q)

    def run_single_TIS(self, 
                       n_mc_steps, 
                       storage, 
                       initial_path, 
                       interface_value, 
                       scheme_move="TwoWay", 
                       scheme_selector="Uniform", 
                       scheme_modifier="RandomVelocities",
                       gaussian_parameter_width=0.5, 
                       gausssian_parameter_shift=0.2,
                       direction="forward", 
                       directory=Path.cwd()):


        print("Creating network...")
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
                    selector = paths.GaussianBiasSelector(self.cv_q, alpha=1/(gaussian_parameter_width)**2, l_0=interface_value-gausssian_parameter_shift)
                elif direction=="backward":
                    selector = paths.GaussianBiasSelector(self.cv_q, alpha=1/(gaussian_parameter_width)**2, l_0=interface_value+gausssian_parameter_shift)
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
            self.store_max_min_q(storage, direction, interface_value, directory=directory)
            if self.model_pkl:
                os.remove("model.pkl")

        else :
            InterruptedError("Storage does not contain an engine")