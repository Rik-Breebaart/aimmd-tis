"""
AIMMD model setup utilities.

This module centralizes AIMMD RC model setup logic used by OpenMM
sampling examples, without touching training or analysis components.
"""

from copy import deepcopy
from pathlib import Path
from typing import Optional

import openpathsampling as paths
import torch
import torch.nn as nn
import aimmd


class AIMMDSetup:
    """
    Utility class for AIMMD RCModel setup and selector construction.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing AIMMD_settings.
    descriptor_dim : int
        Descriptor dimensionality.
    states : tuple
        Tuple of stable state volumes (stateA, stateB).
    descriptor_transform : callable, optional
        Descriptor transform function for snapshots.
    print_config : bool, optional
        Whether to print config.
    """

    def __init__(self, config, descriptor_dim: int, states,
                 descriptor_transform=None, print_config: bool = False):
        self.config = config
        self.potential_name = self.config.get("system_name", "unknown_system")
        self.settings = self.config.get("settings", {})
        self.AIMMD_settings = self.config.get("AIMMD_settings", {})
        self.distribution = self.AIMMD_settings.get("distribution", "lorentzian")
        self.lorentzian_scale = self.AIMMD_settings.get("scale", 1.0)
        self.train_decision_params = self.AIMMD_settings.get("ee_params", {})
        self.use_GPU = self.AIMMD_settings.get("use_GPU", False)
        self.descriptor_dim = descriptor_dim
        self.descriptor_transform = descriptor_transform
        self.states = states

        if print_config:
            try:
                from aimmdTIS.Tools import print_config
                print_config(self.config)
            except Exception:
                pass

    def __select_activation(self, in_features: Optional[int] = None):
        """
        Returns nn.Sequential with activation and optional normalization layers.

        Parameters
        ----------
        in_features : int, optional
            Number of input features for BatchNorm/LayerNorm.

        Returns
        -------
        nn.Sequential
            Activation function with optional normalization.
        """
        activation_setting = self.AIMMD_settings.get("activation", "ReLU")
        use_batchnorm = self.AIMMD_settings.get("use_batchnorm", False)
        use_layernorm = self.AIMMD_settings.get("use_layernorm", False)
        activation_params = self.AIMMD_settings.get("activation_params", {})

        activation_map = {
            "ReLU": nn.ReLU(**activation_params),
            "LeakyReLU": nn.LeakyReLU(**activation_params),
            "PReLU": nn.PReLU(**activation_params),
            "ELU": nn.ELU(**activation_params),
            "SELU": nn.SELU(),
            "GELU": nn.GELU(),
            "Tanh": nn.Tanh(),
            "Sigmoid": nn.Sigmoid(),
            "Softplus": nn.Softplus(**activation_params),
            "Softsign": nn.Softsign(),
            "SiLU": nn.SiLU(),
            "Mish": nn.Mish(),
            "Identity": nn.Identity(),
        }

        if activation_setting not in activation_map:
            raise ValueError(f"Unsupported activation: {activation_setting}")

        layers = [activation_map[activation_setting]]

        if use_layernorm:
            if in_features is None:
                raise ValueError("in_features required for LayerNorm")
            layers.append(nn.LayerNorm(in_features))
        elif use_batchnorm:
            if in_features is None:
                raise ValueError("in_features required for BatchNorm")
            layers.append(nn.BatchNorm1d(in_features))

        return nn.Sequential(*layers)

    def __setup_torch_model(self):
        """
        Create NN model for committor learning and shooting point predictions.

        Returns
        -------
        torch.nn.Module
            Torch model for RC learning.
        """
        layers_settings = self.AIMMD_settings.get("layers", {})
        dropout_settings = self.AIMMD_settings.get("dropout", {})

        hidden_layers = [layers_settings[k] for k in sorted(layers_settings.keys())]
        dropout = [dropout_settings[k] for k in sorted(dropout_settings.keys())]

        n_unit_layers = [self.descriptor_dim] + hidden_layers

        modules = []
        for i in range(len(hidden_layers)):
            activation = self.__select_activation(in_features=n_unit_layers[i + 1])
            modules += [
                aimmd.pytorch.networks.FFNet(
                    n_in=n_unit_layers[i],
                    n_hidden=[n_unit_layers[i + 1]],
                    activation=activation,
                    dropout={"0": dropout[i] if i < len(dropout) else 0.0},
                )
            ]

        torch_model = aimmd.pytorch.networks.ModuleStack(
            n_out=1,
            modules=modules
        )

        if torch.cuda.is_available() and self.use_GPU:
            torch_model = torch_model.to("cuda")
            print("Using CUDA")
        elif torch.backends.mps.is_available() and self.use_GPU:
            torch_model = torch_model.to("mps")
            print("Using MPS")
        else:
            torch_model = torch_model.to("cpu")

        return torch_model

    def __setup_descriptor_transform(self):
        """
        Setup descriptor transform from snapshots.

        Returns
        -------
        paths.FunctionCV
            Function CV for descriptor transformation.
        """
        return paths.FunctionCV(
            "descriptor_transform",
            lambda s: s.coordinates[0],
            cv_wrap_numpy_array=True
        ).with_diskcache()

    def setup_RCModel(self, aimmd_storage, load_model_path: Optional[Path] = None, loss=None):
        """
        Setup RC model for committor prediction.

        Parameters
        ----------
        aimmd_storage : aimmd.Storage
            Storage object for caching.
        load_model_path : Path, optional
            Path to pre-trained model.
        loss : callable, optional
            Custom loss function.

        Returns
        -------
        aimmd.pytorch.TIS_EEScalePytorchRCModel
            Configured RC model.
        """
        trainset = None
        if load_model_path is not None:
            aimmd_store_old = aimmd.Storage(load_model_path, "r")
            rcmodel_old = aimmd_store_old.rcmodels["most_recent"]
            try:
                trainset = deepcopy(aimmd_store_old.load_trainset())
            except Exception:
                trainset = None
            finally:
                torch_model = deepcopy(rcmodel_old.nnet)

                if torch.cuda.is_available() and self.use_GPU:
                    torch_model = torch_model.to("cuda")
                    print("Using CUDA")
                elif torch.backends.mps.is_available() and self.use_GPU:
                    torch_model = torch_model.to("mps")
                else:
                    torch_model = torch_model.to("cpu")

                aimmd_store_old.close()
        else:
            torch_model = self.__setup_torch_model()

        optimizer = torch.optim.AdamW(
            torch_model.parameters(),
            lr=self.train_decision_params.get("lr_0", 1e-3)
        )

        if self.descriptor_transform is None:
            self.descriptor_transform = self.__setup_descriptor_transform()

        model = aimmd.pytorch.TIS_EEScalePytorchRCModel(
            nnet=torch_model,
            optimizer=optimizer,
            states=self.states,
            ee_params=self.train_decision_params,
            descriptor_transform=self.descriptor_transform,
            loss=loss,
            cache_file=aimmd_storage,
        )

        aimmd_storage.rcmodels["most_recent"] = model
        if trainset is not None:
            aimmd_storage.save_trainset(trainset)
        return model

    def load_RCModel(self, aimmd_storage, key="most_recent", mode="r"):
        """
        Load pre-trained RC model from storage.

        Parameters
        ----------
        aimmd_storage : str or Path
            Path to storage file.
        key : str, optional
            Key in storage (default: "most_recent").
        mode : str, optional
            File open mode (default: "r").

        Returns
        -------
        aimmd.pytorch.TIS_EEScalePytorchRCModel
            Loaded RC model.
        """
        aimmd_store = aimmd.Storage(aimmd_storage, mode)
        model = aimmd_store.rcmodels[key]

        if self.descriptor_transform is None:
            self.descriptor_transform = self.__setup_descriptor_transform()
        model.descriptor_transform = self.descriptor_transform

        if torch.cuda.is_available() and self.use_GPU:
            model.nnet = model.nnet.to("cuda")
            print("Using CUDA")
        elif torch.backends.mps.is_available() and self.use_GPU:
            model.nnet = model.nnet.to("mps")
        else:
            model.nnet = model.nnet.to("cpu")

        return model

    def setup_selector(self, RCModel):
        """
        Setup shooting point selector based on RC model.

        Parameters
        ----------
        RCModel : aimmd.pytorch.TIS_EEScalePytorchRCModel
            Trained RC model.

        Returns
        -------
        aimmd.ops.UniformRCModelSelector
            Configured selector.
        """
        return aimmd.ops.UniformRCModelSelector(
            model=RCModel,
            states=self.states,
            distribution=self.distribution,
            density_adaptation=False,
            scale=self.lorentzian_scale,
        )
