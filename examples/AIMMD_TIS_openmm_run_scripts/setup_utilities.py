import numpy as np 
import matplotlib.pyplot as plt
import openpathsampling as paths
import torch
import sys
import os
from functools import reduce
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import argparse
from pathlib import Path

current_directory = os.path.dirname(os.path.abspath(os.getcwd()))
# Add the current directory to sys.path
sys.path.append(current_directory)

# Get the parent directory
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))
parent_parent_directory = os.path.abspath(os.path.join(parent_directory, os.pardir))
# Add the parent directory to sys.path
sys.path.append(parent_directory)
sys.path.append(parent_parent_directory)

# Custom imports
from aimmd import aimmd
import TIS_AIMMD_toy_framework as TAI
from TIS_AIMMD_toy_framework import TIS_AIMMD_setup, read_config
from AIMMD_TIS_openmm_run_scripts.transform_functions import cv_hg_distance



from openmm import app
import openmm as mm
from openmmtools.integrators import VVVRIntegrator
from simtk import unit
import openpathsampling as paths
import openpathsampling.engines.openmm as eng
from collections import namedtuple

class TPS_setup:
    def __init__(self, config_path, resource_directory="", print_config=True):
        self.resource_directory = resource_directory
        self.config = read_config(config_path)
        if print_config:
            TAI.print_config(self.config)

        #fixed units
        self.size_unit = getattr(unit, self.config['UNITS']['size_unit'])
        self.press_unit = getattr(unit, self.config['UNITS']['press_unit'])
        self.temp_unit = getattr(unit, self.config['UNITS']['temp_unit'])
        self.time_unit = getattr(unit, self.config['UNITS']['time_unit'])

        self.bound_distance = self.config['States']['bound']
        self.unbound_distance = self.config['States']['unbound']


        # Simulation settings from config
        integrator_config = self.config.get("Integrator_settings")
        self.pressure = integrator_config['pressure'] * self.press_unit
        self.temperature = integrator_config['temperature'] * self.temp_unit
        self.dt = integrator_config["dt"] * self.time_unit
        self.friction = integrator_config["friction"] / self.time_unit
        self.barostat_frequency = integrator_config["barostat_frequency"]
        self.constant_variable = integrator_config["constant_variable"]
        #System name
        self.system_name = self.config['system_name']
        self.platform_name = self.config["platform_name"]

        # TPS Specific settings
        self.TPS_settings = self.config.get("TPS_settings")
        self.n_frames_max = self.TPS_settings["n_frames_max"]
        self.n_steps_per_frame = self.TPS_settings["n_steps_per_frame"]

        # Initialize components
        self.integrator_temp = self._setup_integrator()
        self._topology, self.system, self.positions = self._setup_system()
        self.template = self._create_template()
        self.topology = self.template.topology
        self.integrator = self._setup_integrator()
        self.md_engine = self._setup_engine(self.integrator)
        self.states = self._define_states()

    def _setup_integrator(self):
        print("Setting up integrator...")
        integrator = VVVRIntegrator(
            self.temperature,
            self.friction,
            self.dt
        )
        integrator.setConstraintTolerance(0.00001)
        return integrator

    def _setup_engine(self, integrator):
        print(f"Setting up engine on {self.platform_name} platform...")
        platform = mm.Platform.getPlatformByName(self.platform_name)
        engine_options = {
            'n_frames_max': self.n_frames_max,
            'n_steps_per_frame': self.n_steps_per_frame
        }
        engine = eng.Engine(
            topology=self.topology,
            system=self.system,
            integrator=integrator,
            openmm_properties=self._get_platform_properties(),
            options=engine_options
        )
        engine.name = 'default'
        engine.initialize(platform)
        
        return engine

    def _setup_system(self):
        # Obtain an initial state of the system 
        # with the correct corresponding boundaries.
        print("Extracting the topology and creating the system...")
        print(self.resource_directory)
        print(os.path.dirname(os.path.abspath(os.getcwd())))
        prmtop = app.AmberPrmtopFile(Path(self.resource_directory / 'complex-explicit.prmtop'))
        inpcrd = app.AmberInpcrdFile(Path(self.resource_directory / 'complex-explicit.inpcrd'))
        system = prmtop.createSystem(
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
            rigidWater=True,
            ewaldErrorTolerance=0.0005
        )
        print("hier ben ik nu 1")

        if self.constant_variable == "P":
            system.addForce(mm.MonteCarloBarostat(self.pressure, self.temperature, self.barostat_frequency))
        
        init_simulation = app.Simulation(prmtop.topology, system, self.integrator_temp)
        init_simulation.context.setPositions(inpcrd.positions)
        init_simulation.loadState(str(self.resource_directory)+ "/cb7_b2_mdV_cont2.state")
        newboxVectors=  init_simulation.context.getState(getParameters=True).getPeriodicBoxVectors()
        init_simulation.topology.setPeriodicBoxVectors(newboxVectors)
        print("hier ben ik nu 2")
        print(newboxVectors)
        oldboxVectors = init_simulation.topology.getPeriodicBoxVectors()
        print(oldboxVectors)
        positions = init_simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
        return prmtop.topology, system, positions

    def _get_platform_properties(self):
        if self.platform_name == 'OpenCL':
            return {'OpenCLPrecision': 'mixed', 'OpenCLDeviceIndex': "2"}
        elif self.platform_name == 'CUDA':
            return {'CudaPrecision': 'mixed'}
        elif self.platform_name == 'CPU':
            return {}
        else:
            return {}

    def _create_template(self):
        print("Creating snapshot template...")
        testsystem = namedtuple('LocalTestSystem', ['name', 'system', 'topology', 'positions'])
        test_system = testsystem(name='CB7:B2', system=self.system, topology=self._topology, positions=self.positions)
        return eng.snapshot_from_testsystem(test_system)

    def _define_states(self):
        print("Defining states and interfaces...")

        bound = paths.CVDefinedVolume(cv_hg_distance, lambda_min=0.0, lambda_max=self.bound_distance).named('bound')
        unbound = paths.CVDefinedVolume(cv_hg_distance, lambda_min=self.unbound_distance, lambda_max=float("inf")).named('unbound')
        return (bound, unbound)
    
class AIMMD_setup:
    def __init__(self, config_path, descriptor_dim,  states, descriptor_transform=None, print_config=False):
        self.config = read_config(config_path)
        #TODO Clean clutter of mapping
        self.potential_name =self.config["system_name"]
        self.settings =self.config.get("settings", {})
        # Use theself.configuration parameters in your MD simulation
        self.AIMMD_settings =self.config.get("AIMMD_settings",{})
        self.distribution = self.AIMMD_settings["distribution"]
        self.lorentzian_scale = self.AIMMD_settings["scale"]
        self.train_decision_params = self.AIMMD_settings.get("ee_params",{})
        self.use_GPU = self.AIMMD_settings["use_GPU"]
        self.descriptor_dim = descriptor_dim
        self.descriptor_transform = descriptor_transform
        self.states = states
        if print_config:
            TAI.print_config(self.config)

    def __select_activation(self):
        activation_setting = self.AIMMD_settings["activation"]
        if activation_setting  == "ReLU":
            activation =  torch.nn.ReLU()
        elif activation_setting == "Identity":
            activation = torch.nn.Identity()
        else:
            ValueError("not yet implemented now only 'Identity' and 'ReLU' are activation options")
        return activation

    def __setup_torch_model(self):

        """ 
        Create the NN model for committor learning and shooting point predictions.
        (In this case the ResNet layer is removed increasing the nonlinearity in the output)

        """

        # we will create a simple non-linear network which can capture the details of the commmittor of the simple 2d potentials. (and thus has enough variability to do so).
        # descriptor_dim -> 16 -> 8 -> 1 (the final 1 is inside the aimmd modelstack network function.) (for total of 513 weights. (if bias is included))
        # each layer will use a Relu activation function.

        layers_settings = self.AIMMD_settings.get("layers",{})
        dropout_settings = self.AIMMD_settings.get("dropout", {})
        Hidden_layers = []
        for layer in list(layers_settings.keys()):
            Hidden_layers.append(layers_settings[layer])


        dropout = []
        for layer in list(dropout_settings.keys()):
            dropout.append(dropout_settings[layer])

        n_unit_layers = [self.descriptor_dim] + Hidden_layers

        modules = []
        for i in range(len(Hidden_layers)):
            modules += [aimmd.pytorch.networks.FFNet(n_in=n_unit_layers[i],
                                                n_hidden = [n_unit_layers[i+1]],
                                                activation = self.__select_activation(),
                                                dropout = {"0":dropout[i]})]


        # this step below does an internal linear layer from the final hidden layer size to the final n_out=1 output.
        torch_model = aimmd.pytorch.networks.ModuleStack(n_out=1,
                                                        modules = modules)

        # move model to GPU if CUDA is available
        if torch.cuda.is_available() and self.use_GPU:
            torch_model = torch_model.to('cuda')
            print("using cuda")
        elif torch.backends.mps.is_available() and self.use_GPU:
            torch_model = torch_model.to('mps')
        else: 
            torch_model = torch_model.to("cpu")
        
        return torch_model

    
    
    def __setup_descriptor_transform(self):
        return paths.FunctionCV('descriptor_transform', lambda s: s.coordinates[0], cv_wrap_numpy_array=True).with_diskcache()


    def setup_RCModel(self, aimmd_storage, load_model_path=None, loss=None):
        trainset = None
        if load_model_path is not None:
            # load aimmd storage from model
            aimmd_store_old = aimmd.Storage(load_model_path ,"r")
            RCModel_old = aimmd_store_old.rcmodels["most_recent"]
            try:
                trainset = deepcopy(aimmd_store_old.load_trainset())
            except:
                trainset = None
            finally:
                print("Let me check I should be printing trainset here:", trainset)
                torch_model = deepcopy(RCModel_old.nnet)

                if torch.cuda.is_available() and self.use_GPU:
                    torch_model = torch_model.to('cuda')
                    print("using cuda")
                elif torch.backends.mps.is_available() and self.use_GPU:
                    torch_model = torch_model.to('mps')
                else: 
                    torch_model = torch_model.to("cpu")
                
                aimmd_store_old.close()
            
        else :
            torch_model = self.__setup_torch_model()
                    
        # choose and initialize an optimizer to train the model
        optimizer = torch.optim.AdamW(torch_model.parameters(), lr=self.train_decision_params["lr_0"])

        if self.descriptor_transform is None:
            self.descriptor_transform = self.__setup_descriptor_transform()

        model = aimmd.pytorch.TIS_EEScalePytorchRCModel(nnet=torch_model,
                                                    optimizer=optimizer,
                                                    states=self.states,
                                                    ee_params=self.train_decision_params,
                                                    descriptor_transform=self.descriptor_transform,  # the function transforming snapshots to descriptors
                                                    loss=loss,  # if loss=None it will choose either binomial or multinomial loss, depending on the number of model outputs,
                                                                # but we could have also passed a custom loss function if we wanted to
                                                    
                                                    cache_file=aimmd_storage,  # cache file for density collector
                                                    )
        aimmd_storage.rcmodels["most_recent"] = model
        if trainset is not None:
            aimmd_storage.save_trainset(trainset)
        return model
    
    def load_RCModel(self,aimmd_storage, key="most_recent", mode="r"):
        # Ensure the file has a .h5 extension using pathlib's with_suffix
        aimmd_store = aimmd.Storage(aimmd_storage, mode)
        model = aimmd_store.rcmodels[key]
        if self.descriptor_transform is None:
            self.descriptor_transform = self.__setup_descriptor_transform()
        model.descriptor_transform = self.descriptor_transform
        # move model to GPU if CUDA is available

        use_cuda = self.use_GPU
        #TODO can be single lines with model.nnet = model_cuda object
        if torch.cuda.is_available() and use_cuda:
            model = TAI.Tools.model_to(model, "cuda")
            print("using cuda")
        elif torch.backends.mps.is_available() and use_cuda:
            model = TAI.Tools.model_to(model, "mps")
        else: 
            model = TAI.Tools.model_to(model, "cpu")
        return model

    def setup_selector(self, RCModel):
        selector = aimmd.ops.UniformRCModelSelector(model=RCModel,  # always takes a RCModel
                                     # we can greatly speed up rejecting/accepting trial TPS by passing the list of states
                                     # this enables testing if a trial TP is even a TP and calculating the potentially costly
                                     # transformation from Cartesian to descriptor space only if neccessary
                                     # if we are lazy and know that the transformation is fast we can also explicitly pass None
                                     states=self.states,
                                     # new shooting points are selected with p_sel(SP) ~ p_lorentz(model.z_sel(SP))
                                     # could also choose 'gaussian'
                                     distribution=self.distribution,
                                     density_adaptation=False,
                                     # softness of the selection distribtion,
                                     # lower values result in a sharper concentration around the predicted transition state,
                                     # higher values result in a more uniform selection
                                     scale=self.lorentzian_scale,
                                    )
        return selector


class TIS_setup(TPS_setup):
    def __init__(self, TPS_config_path, TIS_config_path, resource_directory="", print_config=True):
        super(TIS_setup,self).__init__(TPS_config_path, resource_directory, print_config)

        self.TIS_config = read_config(TIS_config_path)
        if print_config:
            TAI.print_config(self.TIS_config)

        self.shooting_move = self.TIS_config["TIS"]["Scheme"]["Shooting_Move"]
        self.TIS_selector = self.TIS_config["TIS"]["Scheme"]["selector"]
        self.gaussian_width = self.TIS_config["TIS"]["Scheme"]["parameters"]["gaussian width"]
        self.gaussian_origin_shift = self.TIS_config["TIS"]["Scheme"]["parameters"]["origin_shift"]
        self.modification_method = self.TIS_config["TIS"]["modification_method"]
        self.interface_values_forward =  self.TIS_config['TIS']['interface_parameters']['interfaces_forward']
        self.interface_values_backward =  self.TIS_config['TIS']['interface_parameters']['interfaces_backward']



def create_parser(arguments_list):
    """
    Creates an ArgumentParser with arguments specified in arguments_list.

    Args:
        arguments_list (list): List of tuples or dictionaries defining arguments:
            - If tuple: (flags, type, required, help_text, [default], [nargs])
            - If dictionary: {
                "flags": list of flags (e.g., ["-d", "--directory"]),
                "type": argument type (e.g., str, int, Path),
                "required": if the argument is required,
                "help": help message for the argument,
                "default": (optional) default value if argument is not provided,
                "nargs": (optional) number of arguments expected (e.g., "+", "*")
              }

    Returns:
        argparse.ArgumentParser: Configured ArgumentParser
    """
    parser = argparse.ArgumentParser()
    for arg in arguments_list:
        if isinstance(arg, tuple):  # Handle tuple format
            # Unpack flags, type, required, help_text, with optional default and nargs
            flags, arg_type, required, help_text, *rest = arg
            default = rest[0] if len(rest) > 0 and not isinstance(rest[0], str) else None
            nargs = rest[1] if len(rest) > 1 and isinstance(rest[1], str) else None
            
            # Add the argument to the parser
            parser.add_argument(
                *flags,
                type=arg_type,
                required=required,
                help=help_text,
                default=default,
                nargs=nargs
            )
        elif isinstance(arg, dict):  # Handle dictionary format
            # Add the argument to the parser using dict keys
            parser.add_argument(
                *arg["flags"],
                type=arg["type"],
                required=arg.get("required", False),
                help=arg.get("help", ""),
                default=arg.get("default", None),
                nargs=arg.get("nargs", None)
            )
    return parser

# Global argument dictionary
global_arguments = {
    "directory": (["-dir", "--directory"], Path, True, "Directory containing initial trajectory and configuration files."),
    "config_file": (["-cfg", "--config_file"], str, True, "File in python configparser format with simulation settings."),
    "n_steps": (["-nr", "--n_steps"], int, True, "The number of desired TPS/TIS MC cycles."),
    "output_path": (["-out", "--output_path"], Path, False, "Directory for storing TPS output files."),
    "previous_model": (["-old_aimmd", "--old_aimmd"], str, False, "Name of the old AIMMD model to be used."),
    "aimmd_store": (["-aimmd_store", "--aimmd_store"], str, False, "Name for output AIMMD storage, default describes which selector and PES is used."),
    "ops_store": (["-ops_store", "--ops_store"], str, False, "Name for output OPS storage, default describes which selector and PES is used."),
    "final_storage": (["-final_storage", "--final_storage"], str, False, "Save final trajectory as separate OPS storage."),
    "trajectory_path": (["-traj", "--trajectory_path"], Path, False, "Path to the input trajectory storage file."),
    "save_final_traj": (["-save_traj", "--save_final_traj"], bool, False, "Flag to save the final trajectory as a separate OPS storage.", True),
    "final_traj_storage_path": (["-final_traj_store", "--final_traj_storage_path"], str, False, "Path to save the final trajectory storage file."),
    "system_resource_directory": (["-sys_source_dir", "--system_resource_directory"], Path, False, "Directory for system resource files."),
    "n_epoch": (["-n_epoch", "--n_epoch"], int, True, "The number of epochs for training."),
    "stable_ops_storages":(["-stable_ops_store", "--stable_ops_store"], str, False, "Paths to OPS storage files for stable states A and B.", None, "+"),
    "config_file_TPS": (["-cfg_TPS", "--config_file_TPS"], str, True, "File in python configparser format with simulation settings."),
    "config_file_TIS": (["-cfg_TIS", "--config_file_TIS"], str, True, "File in python configparser format with TIS settings."),
    "interface_model": (["-interface_model", "--interface_model"], str, True, "Name of the old AIMMD model to be used."),
    "interface_values": (["-interface_values", "--interface_values"], float, True, "The value of the interface used in the simulation.", None, "+"),
    "direction": (["-direction", "--direction"], str, True, "The direction of the simulation, either 'forward' or 'backward'.", None, None, ["forward", "backward"]),
    "RPE_name": (["-RPE_name","--RPE_name"],str,False,"RPE storage name, default name is RPE_storage"),
    "number_trajectories": (["-n_traj", "--n_traj"], int, False, "The number of trajectories obtained for each TIS run."),
    "n_thermalization": (["-n_therm", "--n_therm"], int, False, "The number of trajectories removed as thermalization.", 200),
    "cutoff": (["-cutoff", "--cutoff"], float, False, "The WHAM cutoff used.", 0.01),
    "max_min_filename": (["-max_min", "--max_min"], str, False, "filename for the max_min stored path (should be first part of the name e.g. 'max_min_q_int')")
}
