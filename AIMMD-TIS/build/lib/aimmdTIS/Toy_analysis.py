import numpy as np 
import torch 
import json
import matplotlib.pyplot as plt
import matplotlib
import openpathsampling as paths
from .Tools import create_discrete_cmap
from .Toy_potentials import CallableVolume
from .Training import snapshot_loss_original, snapshot_lnP, snapshot_loss_normalized_q, snapshot_loss_low_q_scaled, snapshot_loss_sqrt_rho_weight, snapshot_loss_smoothness

class ToyAimmdVisualizer:
    def __init__(self, temperature=1, resolution=501, descriptor_dims=[0,1], total_num_descriptors = None, dims_extent = None, pes=None, toy=True, standard_value=None):
        """
        Initialize the visualizer with potential energy surface (PES), temperature, and resolution.
        """
        self.initialize_nones()
        self.pes = pes 
        self.temperature = temperature
        self.beta = 1 / self.temperature
        self.resolution = resolution
        if dims_extent is None and self.pes is not None:
            dims_extent = self.pes.extent
        elif dims_extent is None:
            dims_extent = [-10,10,-10,10]
        self.dims_extent = dims_extent
        self.descriptor_dims = descriptor_dims
        if self.pes is not None:
            self.x = np.linspace(dims_extent[0], dims_extent[1], resolution)
            self.y = np.linspace(dims_extent[2], dims_extent[3], resolution)
            self.x_2d, self.y_2d, self.U = pes.plot_2d_pes(self.x, self.y)
            self.total_num_descriptors = self.pes.n_harmonics + self.pes.n_dims_pot
        else:
            self.total_num_descriptors = total_num_descriptors
        self.descriptor_dims = descriptor_dims
        self.descriptor_dims_hist_used = None
        self.standard_value = [0]*self.total_num_descriptors if standard_value is None else standard_value
        self.set_plot_settings()
    
    def committor_2d_projection(self, plot_model, n_epoch=None, fig=None, ax=None):
        if ax is None:
            fig,ax = plt.subplots(1,1)
        self.plot_committor_model_projection(plot_model,ax=ax)
        if self.pes is not None:
            self.plot_potential_contour(ax=ax)
        self.plot_committor_model_projection_contour(plot_model,ax=ax,colors="k")
        if n_epoch == None: 
            ax.set_title(r'model prediction $p_B$')
        else:   
            ax.set_title(r'model prediction $p_B$ at epoch {:d}'.format(n_epoch))

    def q_space_2d_projection(self, plot_model, n_epoch=None, v_min_max = 15, fig=None, ax=None):
        if ax is None:
            fig,ax = plt.subplots(1,1)
        self.plot_q_model_projection(plot_model,ax=ax)
        if self.pes is not None:
            self.plot_potential_contour(ax=ax)
        self.plot_q_model_projection_contour(plot_model,ax=ax,colors="k")

        if n_epoch == None:
            ax.set_title(r'model prediction in $q(x)$-space ($q(x)=-\log(1/p_B -1)$)')
        else:   
            ax.set_title(r'model prediction in $q(x)$-space ($q(x)=-\log(1/p_B -1)$) at epoch {:d}'.format(n_epoch))

    def RPE_2d(self, fig=None, ax=None):
        if ax is None:
            fig_, ax = plt.subplots(1,1)
        self.plot_RPE_histogram(ax=ax)
        self.plot_RPE_contours(ax=ax,fig=fig)
        if self.pes is not None:
            self.plot_potential_contour(ax=ax)
        ax.set_title("RPE distribution")
        return 

    def set_plot_settings(self):
        """
        Set the plotting settings, including colors, line widths, and color maps.
        """
        colors = [
            [33, 104, 108],
            [1, 82, 54],
            [107, 196, 166],
            [254, 152, 42],
            [187, 79, 12],
            [110, 41, 12],
            [158, 158, 158]
        ]
        self.colors = [[c[0] / 254, c[1] / 254, c[2] / 254] for c in colors]

        import matplotlib.pylab as pylab
        params = {
            'legend.fontsize': 'x-large',
            'figure.figsize': (5, 5),
            'axes.labelsize': 'x-large',
            'axes.titlesize': 'x-large',
            'xtick.labelsize': 'x-large',
            'ytick.labelsize': 'x-large'
        }
        pylab.rcParams.update(params)

        self.linewidth = 2
        self.color_forward = self.colors[2]
        self.color_backward = self.colors[4]
        self.color_prob = self.colors[-2]
        self.alpha = 0.7
        self.fontsize = 20 
        self.cmap = "Spectral"


        # Commitor plot settings
        self.alpha_committor = 0.7
        # self.cmap_committor ="Spectral"
        self.cmap_committor =create_discrete_cmap(24)

        self.levels_committor = np.linspace(0,1,11)
        q_min = -16
        q_max = 16
        delta_q =2
        self.linewidth_committor_contour = 3
        self.levels_q_model = np.arange(q_min, q_max+1,delta_q)
        self.levels_q_data = np.arange(q_min, q_max+1,delta_q)
        
        # Theory plot settings
        self.alpha_theory = 0.7
        self.cmap_theory = None
        self.levels_theory = np.arange(q_min, q_max+1,delta_q)
        self.linewidth_theory = 1
        self.linestyle_theory = "--"
        self.color_theory = "black"

        # RPE distribution and potential plot settings
        self.alpha_potential = 0.7
        self.alpha_distribution = 0.7
        self.cmap_distribution = "Blues"
        self.cmap_potential_contours = "gray"
        self.linewidth_potential_contours = 2
        if self.pes is not None:
            self.levels_U = self.beta * self.pes.levels
        else:
            self.levels_U = np.linspace(0,1,11)
        self.cmap_RPE_committor_contours = "Spectral"
        self.cmap_RPE_committor_model_contours = "Spectral"

    def plot_states(self, fig=None, ax=None):
        # Collective variables to define the states
        opA = paths.CoordinateFunctionCV(name="opA", f=self.pes.stable_interface_function, center=self.pes.state_A)
        opB = paths.CoordinateFunctionCV(name="opB", f=self.pes.stable_interface_function, center=self.pes.state_B)

        # State volumes in CV space
        stateA = paths.CVDefinedVolume(opA, 0.0, self.pes.state_boundary).named('StateA')
        stateB = paths.CVDefinedVolume(opB, 0.0, self.pes.state_boundary).named('StateB')
        xedges, yedges = self.create_x_y_edges(dims_extent=self.pes.extent, n_bins_2d=[501,501])

        x_2d, y_2d = np.meshgrid(xedges, yedges)
        states_plot_A = np.vectorize(CallableVolume(stateA))(x_2d,y_2d)
        states_plot_B = np.vectorize(CallableVolume(stateB))(x_2d,y_2d)
        if ax is None:
            fig,ax = plt.subplots(1,1)
        ax.contour(x_2d,y_2d,states_plot_A, colors='red',linewidth=self.linewidth_theory)
        ax.contour(x_2d,y_2d,states_plot_B, colors='blue', linewidth=self.linewidth_theory)

    def load_theoretical_committor(self, theoretical_committor_path, n_x, n_y):
        """
        Load the theoretical committor function from a file.
        Computed using the method found in:
        "Molecular free energy profiles from force spectroscopy experiments by inversion of observed committors" 
        (Covino, Woodside, Hummer, Szabo and Cossio; J. Chem. Phys. 151, 154115 (2019); https://doi.org/10.1063/1.5118362).
        """
        if theoretical_committor_path is None:
            theoretical_committor_path = f"committor_{self.pes.__repr__()}_xy_{n_x}_beta.npy"
        P_theory = np.load(theoretical_committor_path)
    
        xedges_theory = np.linspace(self.dims_extent[0], self.dims_extent[1], n_x)
        yedges_theory = np.linspace(self.dims_extent[2], self.dims_extent[3], n_y)
        # x_2d_theory, y_2d_theory = np.meshgrid(xedges_theory, yedges_theory)

        q_theory = -np.log(1 / P_theory - 1)
        return P_theory, q_theory, xedges_theory, yedges_theory

    def theoretical_committor_contour(self, theoretical_committor_path, n_x, n_y=None, levels=None, fig=None, ax=None):
        """
        Plot the contour of the theoretical committor function.
        """
        if n_y is None:
            n_y = n_x
        p_theory, q_theory, xedges, yedges = self.load_theoretical_committor(theoretical_committor_path, n_x, n_y)
        x_2d, y_2d = np.meshgrid(xedges, yedges)

        if ax is None:
            fig, ax = plt.subplots(1)

        contour = ax.contour(x_2d, y_2d, p_theory, levels=self.levels_committor, cmap=self.cmap_theory, linewidths=self.linewidth_theory, colors=self.color_theory)
        ax.clabel(contour, inline=1, fontsize=10)
        if fig is not None:
            fig.colorbar(contour)

    def plot_theoretical_q(self,theoretical_committor_path, n_x, n_y=None, fig=None, ax=None):
        """
        Plot the contour of the theoretical q function.
        """
        if n_y is None:
            n_y = n_x
        p_theory, q_theory, xedges, yedges = self.load_theoretical_committor(theoretical_committor_path, n_x, n_y)

        if ax is None:
            fig, ax = plt.subplots(1)

        im = ax.imshow(q_theory, interpolation='nearest', origin='lower',cmap=self.cmap_theory,
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],aspect="auto",alpha=0.8)
        if fig is not None:
            cb = fig.colorbar(im, ax=ax)
            cb.set_label("Theoretical q")
    
    def plot_theoretical_committor(self,theoretical_committor_path, n_x, n_y=None, fig=None, ax=None):
        """
        Plot the contour of the theoretical q function.
        """
        if n_y is None:
            n_y = n_x
        p_theory, q_theory, xedges, yedges = self.load_theoretical_committor(theoretical_committor_path, n_x, n_y)

        if ax is None:
            fig, ax = plt.subplots(1)

        im = ax.imshow(p_theory, interpolation='nearest', origin='lower',cmap=self.cmap_theory,
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],aspect="auto",alpha=0.8)
        if fig is not None:
            cb = fig.colorbar(im, ax=ax)
            cb.set_label(r"Theoretical $p_b$")

    def theoretical_q_contour(self, theoretical_committor_path, n_x, n_y=None, fig=None, ax=None, colorbar=True):
        """
        Plot the contour of the theoretical q function.
        """
        if n_y is None:
            n_y = n_x
        p_theory, q_theory, xedges, yedges = self.load_theoretical_committor(theoretical_committor_path, n_x, n_y)
        x_2d, y_2d = np.meshgrid(xedges, yedges)

        if ax is None:
            fig, ax = plt.subplots(1)

        contour = ax.contour(x_2d, y_2d, q_theory, levels=self.levels_theory, cmap=self.cmap_theory, linewidths=self.linewidth_theory, alpha=self.alpha_theory, linestyles=self.linestyle_theory, colors=self.color_theory)
        ax.clabel(contour, inline=1, fontsize=10)
        if fig is not None and colorbar:
            cb = fig.colorbar(contour, ax=ax)
            cb.set_label("Theoretical q contour")

    def load_RPE_data(self, RPE):
        """
        Load RPE (Reweighted Path Ensemble) data.
        """
        self.RPE = RPE

    def create_x_y_edges(self,dims_extent=None, n_bins_2d=100):
        """
        Create x and y edges for the 2D grid.
        """
            # Generate edges for the 2D grid
        if dims_extent is None:
            dims_extent = self.dims_extent
        if np.shape(n_bins_2d) == ():
            n_bins_2d = [n_bins_2d, n_bins_2d]
        xedges = np.linspace(dims_extent[0], dims_extent[1], n_bins_2d[0])
        yedges = np.linspace(dims_extent[2], dims_extent[3], n_bins_2d[1])
        return xedges, yedges

    def model_2d_output(self, plot_model):
        """
        Generate 2D output from the model on a grid.
        """
        if self.pes is None:
            raise ValueError("PES is not defined.")
        oscis = [0. for _ in range(self.total_num_descriptors - 2)]
        coord = np.array([[xv, yv] + oscis for yv in self.y for xv in self.x], dtype=np.float32)
        q = plot_model.log_prob(torch.as_tensor(coord, device=plot_model._device), use_transform=False)
        q = q.reshape((len(self.x), len(self.y)))
        return q
    
    def compute_histogram_weighted(self, descriptor_dims=None):
        """
        Compute weighted histograms for the forward, backward, and stable data.
        """
        if descriptor_dims is None:
            descriptor_dims = self.descriptor_dims

        if self.H_forward is None or descriptor_dims != self.descriptor_dims_hist_used :
            self.H_forward, self.H_each_interface_forward, self.xedges, self.yedges = self.weighted_histogram2d(
                self.RPE.data_Forward[0], self.RPE.data_Forward[1], descriptor_dims=descriptor_dims)
 
        if self.H_backward is None or descriptor_dims != self.descriptor_dims_hist_used :
            self.H_backward, self.H_each_interface_backward, _, _ = self.weighted_histogram2d(
                self.RPE.data_Backward[0], self.RPE.data_Backward[1], descriptor_dims=descriptor_dims)
        
        if self.RPE.data_Stable is not None:
            if self.H_stable is None or descriptor_dims != self.descriptor_dims_hist_used :
                self.H_stable, self.H_each_state, _, _ = self.weighted_histogram2d(
                    self.RPE.data_Stable[0], self.RPE.data_Stable[1], descriptor_dims=descriptor_dims)
        self.descriptor_dims_hist_used = descriptor_dims
              
    def create_2d_projection_coord(self, n_bins_2d=100, descriptor_dims=None, dims_extent = None, standard_value=None):
        if descriptor_dims is None:
            descriptor_dims = self.descriptor_dims
        
        xedges,yedges = self.create_x_y_edges(dims_extent=dims_extent, n_bins_2d=n_bins_2d)

        # Initialize standard_value if not provided
        if self.standard_value is None and standard_value is None:
            standard_value = [0] * self.total_num_descriptor
        elif standard_value is None:
            standard_value = self.standard_value

        # Ensure standard_value has the correct length
        assert len(standard_value) == self.total_num_descriptors, \
            f"standard_value must have length {self.total_num_descriptors}, got {len(standard_value)}"

        # Create coordinates for the 2D grid with the specified standard values
        coord = []
        for yv in yedges:
            for xv in xedges:
                point = standard_value.copy()
                point[descriptor_dims[0]] = xv
                point[descriptor_dims[1]] = yv
                coord.append(point)
        coord = np.array(coord, dtype=np.float32)
        return coord, xedges, yedges

    def create_1d_slice_coord(self, n_bins=100, descriptor_dim=None, dims_extent = None, standard_value=None):
        if descriptor_dim is None:
            descriptor_dim = self.descriptor_dims[0]
        
        xedges = np.linspace(dims_extent[0],dims_extent[1],n_bins)


        # Ensure standard_value has the correct length
        assert len(standard_value) == self.total_num_descriptors, \
            f"standard_value must have length {self.total_num_descriptors}, got {len(standard_value)}"

        # Create coordinates for the 2D grid with the specified standard values
        coord = []
        for xv in xedges:
            point = standard_value.copy()
            point[descriptor_dim] = xv
            coord.append(point)
        coord = np.array(coord, dtype=np.float64)
        return coord, xedges
    
    def compute_q_model_2d(self, model, descriptor_dims=None, dims_extent = None, standard_values = None, n_bins_2d = 300):
        """
        Compute the q model in 2D.
        """
        coord, xedges, yedges = self.create_2d_projection_coord(n_bins_2d=n_bins_2d, descriptor_dims=descriptor_dims, dims_extent=dims_extent, standard_value=standard_values)
        
        self._model_contours = model
        q = model.log_prob(torch.as_tensor(coord, device=model._device), use_transform=False)
        q = q.reshape((len(yedges), len(xedges)))
        X_pb_2d, Y_pb_2d = np.meshgrid(xedges, yedges)
        return q, X_pb_2d, Y_pb_2d

    def compute_q_model_1d(self, model, descriptor_dim=None, dims_extent = None, standard_values = None, n_bins = 300):
        """
        Compute the q model in 2D.
        """
        coord, xedges= self.create_1d_slice_coord(n_bins=n_bins, descriptor_dim=descriptor_dim, dims_extent=dims_extent, standard_value=standard_values)
        q = model.log_prob(torch.as_tensor(coord, device=model._device), use_transform=False)
        return q, xedges
    
    def full_RPE_histogram(self, n_bins_2d=100, descriptor_dims=None):
        """
        Compute the full histogram for RPE data.
        """
        if descriptor_dims is None:
            descriptor_dims = self.descriptor_dims

        descriptors_total, weights_total, shot_results_total = self.RPE.create_total_trainset()

        xedges, yedges = self.create_x_y_edges(n_bins_2d=n_bins_2d)
        H, _, _ = np.histogram2d(
            descriptors_total[:, descriptor_dims[0]],
            descriptors_total[:, descriptor_dims[1]],
            weights=weights_total, bins=(xedges, yedges),
            range=[[self.dims_extent[0], self.dims_extent[1]], [self.dims_extent[2], self.dims_extent[3]]],
            density=True
        )
        self.H_full = H

    def RPE_histogram_allong(self, n_bins, extent,descriptor_dim):
        """
        Compute the full histogram for RPE data.
        """
        if descriptor_dim is None:
            descriptor_dim = self.descriptor_dims[0]

        descriptors_total, weights_total, shot_results_total = self.RPE.create_total_trainset()
        xedges = np.linspace(self.dims_extent[0], self.dims_extent[1], n_bins)
        H, edges = np.histogram(descriptors_total[:,descriptor_dim], weights=weights_total, bins=xedges,
        range=[self.dims_extent[0], self.dims_extent[1]], density=True)
        return H, edges

    def plot_potential_contour(self,fig=None,ax=None):
        """
        Plot the potential energy surface.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        # im = ax.contour(self.x_2d, self.y_2d, -self.beta*self.U, levels=-self.levels_U[::-1], cmap=self.cmap_potential_contours, alpha=self.alpha_potential, linewidths=self.linewidth_potential_contours)
        im = ax.contour(self.x_2d, self.y_2d, self.beta*self.U, levels=self.beta*self.levels_U, cmap=self.cmap_potential_contours, alpha=self.alpha_potential, linewidths=self.linewidth_potential_contours)
        ax.set_xlabel(r"$s_x$")
        ax.set_ylabel(r"$s_y$")
        if fig is not None:
            cb = fig.colorbar(im, ax=ax)
            cb.set_label(r"potential ($K_b T$)")
    
    def plot_potential_density_contour(self,fig=None,ax=None):
        """
        Plot the potential energy surface.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        contour = ax.contour(self.x_2d, self.y_2d, -self.beta*self.U, levels=-self.levels_U[::-1], cmap=self.cmap_potential_contours, alpha=self.alpha_potential, linewidths=self.linewidth_potential_contours, colors=self.color_theory)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.clabel(contour, inline=1, fontsize=10)

        if fig is not None:
            cb = fig.colorbar(contour, ax=ax)
            cb.set_label(r"weight $e^{-\beta U}$ ($K_b T$)")
    
    def plot_potential(self, vmax=None,vmin=None, fig=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        im = ax.imshow(self.beta*self.U, origin='lower', vmax=vmax,vmin=vmin,cmap=self.cmap_distribution, extent=self.dims_extent, aspect='auto')
        # im = ax.imshow(-self.beta*self.U, origin='lower', vmin=-vmax, cmap=self.cmap_distribution, extent=self.dims_extent, aspect='auto')
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        if fig is not None:
            fig.colorbar(im, ax=ax)
    
    def plot_log_density(self, vmax=None,vmin=None, fig=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        im = ax.imshow(-self.beta*self.U, origin='lower', vmin=vmin, vmax=vmax, cmap=self.cmap_distribution, extent=self.dims_extent, aspect='auto')
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        if fig is not None:
            fig.colorbar(im, ax=ax)
        
    def plot_density_potential(self, fig=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        im = ax.imshow(np.exp(-self.beta*self.U), origin='lower', cmap=self.cmap_distribution, extent=self.dims_extent, aspect='auto')
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        if fig is not None:
            fig.colorbar(im, ax=ax)

    def plot_RPE_histogram(self, ax=None, fig=None, n_bins_2d=100, descriptor_dims=None, log_rho=True, v_min_max=None, offset=False):
        """
        Plot the data histogram.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        # if self.H_full is None or descriptor_dims != None:
        #     self.full_RPE_histogram(descriptor_dims=descriptor_dims)
        self.H_full, xedges, yedges = self.weighted_RPE(n_bins_2d=n_bins_2d,descriptor_dims=descriptor_dims) 
        with np.errstate(divide='ignore'):
            H_log_rho = np.log(self.H_full.T)
        if v_min_max is not None:
            v_min, v_max = v_min_max[0], v_min_max[1]
        else:
            v_min, v_max = None, None
        if offset:
            ind = np.unravel_index(np.argmin(-H_log_rho, axis=None), H_log_rho.shape)
            RPE_FE_offset = H_log_rho[ind[0],ind[1]]
        else: 
            RPE_FE_offset = 0   
            
        if log_rho:
            im = ax.imshow(H_log_rho, origin='lower', cmap=self.cmap_distribution, extent=self.dims_extent, aspect='auto',vmin=v_min,vmax=v_max)
        else:  
            im = ax.imshow(-H_log_rho+RPE_FE_offset, origin='lower', cmap=self.cmap_distribution, extent=self.dims_extent, aspect='auto',vmin=v_min,vmax=v_max)
        if fig is not None:
            fig.colorbar(im, ax=ax)
        return im

    def plot_RPE_contours(self, ax=None, fig=None, n_bins_2d=100, descriptor_dims=None, offset=True, levels=None,cmap="Greys_r", clabel=False):
        """
        Plot the data histogram contours.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        self.H_full, xedges, yedges = self.weighted_RPE(n_bins_2d=n_bins_2d,descriptor_dims=descriptor_dims) 
        with np.errstate(divide='ignore'):
            H_log_rho = np.log(self.H_full.T)
        if self.pes is not None and offset:
            xedges, yedges = self.create_x_y_edges(n_bins_2d=n_bins_2d)

            x_2d,y_2d, U = self.pes.plot_2d_pes(xedges,yedges)
            ind = np.unravel_index(np.argmin(U, axis=None), U.shape)
            RPE_FE = -np.log(self.H_full.T)/self.beta

        if offset:
            ind = np.unravel_index(np.argmin(-H_log_rho, axis=None), H_log_rho.shape)
            RPE_FE_offset = H_log_rho[ind[0],ind[1]]
        else: 
            RPE_FE_offset = 0   
        if levels is None:
            levels = len(self.levels_U)
        else:
            levels = levels
        

        im = ax.contour(-H_log_rho+RPE_FE_offset, levels = levels, cmap=cmap, extent=self.dims_extent, linewidths=self.linewidth)
        if clabel==True:
            ax.clabel(im, inline=True, fontsize=10)
        if fig is not None:
            fig.colorbar(im, ax=ax)
    
    def plot_RPE_error(self, ax=None, fig=None, n_bins_2d=100,vmax=3):
        eps=0.02
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        if self.H_full is None:
            self.full_RPE_histogram()
        xedges, yedges = self.create_x_y_edges(n_bins_2d=n_bins_2d)

        x_2d,y_2d, U = self.pes.plot_2d_pes(xedges,yedges)
        ind = np.unravel_index(np.argmin(U, axis=None), U.shape)
        # ind = [50,50]
        offset =  U[ind[0],ind[1]]
        U_shifted = U - offset
        RPE_FE = -np.log(self.H_full.T)/self.beta
        RPE_FE_offset = RPE_FE[ind[0],ind[1]]
        RPE_shifted = RPE_FE - RPE_FE_offset
        im = ax.imshow(np.abs(U_shifted[1:,1:]-RPE_shifted), origin='lower', cmap=self.cmap, vmax=vmax, extent=self.dims_extent, aspect='auto')
        ax.scatter(x_2d[0,ind[1]],y_2d[ind[0],0], s=8, c="black")

        if fig is not None:
            cb = fig.colorbar(im, ax=ax)
            cb.set_label(r"Error RPE [$K_b T$]")

    def error_in_q_model_vs_theory(self, plot_model, theoretical_committor_path,n_x,n_y=None,fig=None, ax=None):
        """
        Plot the relative error between the model and 0,0 projection
        """
        if n_y is None:
            n_y = n_x
        p_theory, q_theory, xedges, yedges = self.load_theoretical_committor(theoretical_committor_path, n_x, n_y)
        q, x_2d, y_2d = self.compute_q_model_2d(plot_model, n_bins_2d=n_x)
        error = np.abs(q_theory-q)

        return error, x_2d,y_2d
    

    def error_in_q_model_on_RPE_vs_theory(self, plot_model, theoretical_committor_path,n_x,n_y=None,fig=None, ax=None):
        """
        Plot the relative error between the model and 0,0 projection
        """
        if n_y is None:
            n_y = n_x
        p_theory, q_theory, xedges, yedges = self.load_theoretical_committor(theoretical_committor_path, n_x, n_y)
        average_pb_model, average_q_model, xedges, yedges = self.committor_model_2d_RPE(plot_model,n_bins_2d=n_x+1, descriptor_dims=[0,1])
        error = np.abs(q_theory-average_q_model)
        x_2d, y_2d = np.meshgrid(xedges, yedges)
        return error, x_2d,y_2d

    def error_in_q_of_RPE_vs_theory(self, plot_model, theoretical_committor_path,n_x,n_y=None,fig=None, ax=None):
        """
        Plot the relative error between the model and 0,0 projection
        """
        if n_y is None:
            n_y = n_x
        p_theory, q_theory, xedges, yedges = self.load_theoretical_committor(theoretical_committor_path, n_x, n_y)
        p_B_histogram, xedges, yedges = self.weighted_committor_RPE(n_bins_2d=n_x+1, descriptor_dims=[0,1])
        error = np.abs(q_theory-(np.log(p_B_histogram)-np.log(1-p_B_histogram)))
        x_2d, y_2d = np.meshgrid(xedges, yedges)
        return error, x_2d,y_2d

    def plot_error_in_q_model_vs_theory_projection(self, plot_model,theoretical_committor_path,n_x,n_y=None,fig=None, ax=None, vmax=3):
        error, x_2d, y_2d = self.error_in_q_model_vs_theory(plot_model,theoretical_committor_path,n_x,n_y)
        if ax is None:
            fig,ax = plt.subplots(1,1)
        im = ax.imshow(error, origin='lower',interpolation='nearest', 
            extent=self.dims_extent,aspect="auto", cmap=self.cmap_committor, vmax=vmax)
        if fig is not None:
            fig.colorbar(im);
        return im

    def initialize_nones(self):
        """
        Initialize variables to None.
        """
        self.q, self.q_model, self.X_pb_2d, self.Y_pb_2d = None, None, None, None
        self.RPE = None
        self.H_forward, self.H_each_interface_forward = None, None
        self.H_backward, self.H_each_interface_backward = None, None
        self.H_stable, self.H_each_state = None, None
        self.H_full = None
        self.contour_committor = None
        self._model_contours = None

    def weighted_histogram2d(self,descriptors_interface, weights_interface, n_bins_2d=100, descriptor_dims = [0,1]):
        
        if descriptor_dims is None:
            descriptor_dims = self.descriptor_dims
        if np.shape(n_bins_2d) == ():
            n_bins_2d = [n_bins_2d, n_bins_2d]
        H = np.zeros(((n_bins_2d[0]-1),(n_bins_2d[1]-1)))
        xedges, yedges = self.create_x_y_edges(n_bins_2d=n_bins_2d)

        H_each_interface = []
        for index_interface in range(len(descriptors_interface)):
            H_interface, xedges, yedges = np.histogram2d(descriptors_interface[index_interface][:,descriptor_dims[0]],
                                                        descriptors_interface[index_interface][:,descriptor_dims[1]],
                                                        weights=weights_interface[index_interface],
                                                        bins=(xedges, yedges))
            H += H_interface    
            H_each_interface.append(H_interface)
        return H, H_each_interface, xedges, yedges

    def plot_q_model_projection(self, plot_model, fig=None, ax=None):
        q, x_2d, y_2d = self.compute_q_model_2d(plot_model)
        if ax is None:
            fig,ax = plt.subplots(1,1)
        im = ax.imshow(q, origin='lower',interpolation='nearest', 
            extent=self.dims_extent,aspect="auto", cmap=self.cmap_committor)
        if fig is not None:
            fig.colorbar(im);
        
    def plot_q_model_projection_contour(self, plot_model, levels = None, fig=None, ax=None, colors=None):
        q, x_2d, y_2d = self.compute_q_model_2d(plot_model)
        if levels is None:
            levels = self.levels_q_model
        if ax is None:
            fig,ax = plt.subplots(1,1)
        if colors is None:
            CS = ax.contour(x_2d, y_2d, q , levels=levels, cmap=self.cmap_committor, linewidths=self.linewidth_committor_contour)
        else: 
            CS = ax.contour(x_2d, y_2d, q , levels=levels, colors=colors, linewidths=self.linewidth_committor_contour)
        ax.clabel(CS, inline=1, fontsize=10)
        if fig is not None:
            fig.colorbar(CS);
    
    def plot_committor_model_projection(self, plot_model, fig=None, ax=None):
        q, x_2d, y_2d = self.compute_q_model_2d(plot_model)
        p = 1/(1+np.exp(-q))

        if ax is None:
            fig,ax = plt.subplots(1,1)
        im = ax.imshow(p, origin='lower',interpolation='nearest', 
            extent=self.dims_extent,aspect="auto", cmap=self.cmap_committor)
        if fig is not None:
            fig.colorbar(im);
        
    def plot_committor_model_projection_contour(self, plot_model, levels = None, fig=None, ax=None, colors=None):
        q, x_2d, y_2d = self.compute_q_model_2d(plot_model)
        p = 1/(1+np.exp(-q))
        if levels is None:
            levels = np.linspace(0,1,11)
        if ax is None:
            fig,ax = plt.subplots(1,1)
        if colors is None:
            CS = ax.contour(x_2d, y_2d, p , levels=levels, cmap=self.cmap_committor, linewidths=self.linewidth_committor_contour)
        else: 
            CS = ax.contour(x_2d, y_2d, p , levels=levels, colors=colors, linewidths=self.linewidth_committor_contour)
        ax.clabel(CS, inline=1, fontsize=10)
        if fig is not None:
            fig.colorbar(CS);

    #TODO seperate TIS ensembles into smaller tasks for plotting
    def all_interfaces_ensembles(self,dim_y=1,model=None, fig=None, ax=None, dq_above=None, show_qpoints=False):

        for i, interface in enumerate(self.RPE.interfaces_forward[:]):
            self.TIS_ensemble_2d(direction="forward", interface_q = interface, dim_y=dim_y, model=model, dq_above=dq_above, show_qpoints=show_qpoints)

        for i, interface in enumerate(self.RPE.interfaces_backward[:]):
            self.TIS_ensemble_2d(direction="backward", interface_q = interface, dim_y=dim_y,  model=model,dq_above=dq_above, show_qpoints=show_qpoints)

        if self.RPE.data_Stable is not None:
            for i in range(2):
                self.TIS_ensemble_2d(direction="stable", state=i, dim_y=dim_y,  model=model)

    def TIS_ensemble_2d(self, direction=None, interface_q = None, dim_y=1, model=None, state=None, dq_above=None, show_qpoints=False):
        descriptor_dims = [0,dim_y]
        self.compute_histogram_weighted(descriptor_dims=descriptor_dims)
        fig2, ax = plt.subplots(1,figsize=(10,8),dpi=120)
        eps = 0.001
        levels = [interface_q]
        if direction=="forward":
            index_interface= int(np.where(np.abs(np.array(self.RPE.interfaces_forward)-interface_q)<eps)[0])


            H_plot = self.H_each_interface_forward[index_interface]
        if direction=="backward":
            index_interface= int(np.where(np.abs(np.array(self.RPE.interfaces_backward)-interface_q)<eps)[0])
            H_plot = self.H_each_interface_backward[index_interface]
            q_range = [interface_q-100, interface_q]

        if direction=="stable" and self.RPE.data_Stable is not None:
            index_interface = state
            levels = self.levels_q_model
            H_plot = self.H_each_state[index_interface]
            # H_plot = self.H_each_state[0] + self.H_each_state[1]


        H_plot = H_plot.T
        H_normalized = H_plot/(np.sum(H_plot)*(self.xedges[1]-self.xedges[0])*(self.yedges[1]-self.yedges[0]))
    #     im = ax[index_interface].imshow(np.log(H_plot), interpolation='nearest', origin='lower',cmap="Blues",
    #             extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],aspect="auto")
        im = ax.imshow(np.log(H_normalized), interpolation='nearest', origin='lower',cmap="Blues",
            extent=[self.xedges[0], self.xedges[-1], self.yedges[0], self.yedges[-1]],aspect="auto",alpha=0.8)
        if self.pes is not None:
            x_2d,y_2d,U = self.pes.plot_2d_pes(self.xedges, self.yedges, dim_y=dim_y)
            ax.contour(x_2d,y_2d,self.beta*U,levels= self.levels_U, cmap="gray",alpha = 0.8)
        X, Y = np.meshgrid(self.xedges[1:], self.yedges[1:])
        ax.contour(X, Y, np.log(H_normalized),cmap="Blues")
        if model !=None:
            if self.q is None or self._model_contours != model:
                q, X_pb_2d, Y_pb_2d = self.compute_q_model_2d(model, descriptor_dims=descriptor_dims)
            CS = ax.contour(X_pb_2d, Y_pb_2d, q, levels=levels,linewidths=4, colors='k',alpha=0.5)
            ax.clabel(CS, inline=1, fontsize=20)
        if interface_q is not None and show_qpoints:
            if dq_above is None:
                dq_above = 100
            if direction=="forward":
                q_range = [interface_q, interface_q+dq_above]
                data_interface = self.RPE.data_Forward[0][index_interface]
            else: 
                q_range = [interface_q-dq_above, interface_q]
                data_interface = self.RPE.data_Backward[0][index_interface]
            self.density_given_q_model_value_range(plot_model=model,q_value_range=q_range,ax=ax, descriptor_dims=[0,dim_y], descriptors=data_interface)
        fig2.colorbar(im);
        plt.show()

    def weighted_histogram_1d(self, descriptors_interface, weights_interface, n_bins=100, descriptor_dim = 0, range=None):
        H = np.zeros((n_bins))
        H_each_interface = []
        for index_interface, interface in enumerate(len(descriptors_interface)):
            H_interface, edges= np.histogram(descriptors_interface[index_interface][:,descriptor_dim],
                                                       bins=n_bins, range=range, weights = weights_interface[index_interface])
            
            H += H_interface    
            H_each_interface.append(H_interface)
        return H, H_each_interface, edges

    def weighted_committor_RPE(self, n_bins_2d=100, dims_extent=None, descriptor_dims=None):
        if dims_extent is None:
            dims_extent = self.dims_extent
        if descriptor_dims is None:
            descriptor_dims = self.descriptor_dims

        descriptors_total, weights_total,shot_results_total = self.RPE.create_total_trainset()
        xedges, yedges = self.create_x_y_edges(dims_extent=dims_extent, n_bins_2d=n_bins_2d)

        # now look at the P_b based on the training data
        H_weighted_pB, xedges, yedges = np.histogram2d(descriptors_total[:,descriptor_dims[0]],
                                                descriptors_total[:,descriptor_dims[1]],
                                                bins=(xedges, yedges), 
                                                weights =weights_total *shot_results_total[:,1])
        H_weighted_pA, xedges, yedges = np.histogram2d(descriptors_total[:,descriptor_dims[0]],
                                                descriptors_total[:,descriptor_dims[1]],
                                                bins=(xedges, yedges), 
                                                weights =  weights_total*shot_results_total[:,0])
        normalizing_weight = H_weighted_pA.T+H_weighted_pB.T

        p_B_histogram = H_weighted_pB.T/normalizing_weight

        return p_B_histogram, xedges, yedges
    
    def unweighted_PE(self,n_bins_2d=100, descriptor_dims=None):
        if descriptor_dims is None:
            descriptor_dims = self.descriptor_dims

        descriptors_total, weights_total,shot_results_total = self.RPE.create_total_trainset()
        xedges, yedges = self.create_x_y_edges(n_bins_2d=n_bins_2d)

        # now look at the P_b based on the training data
        H_unweighted, xedges, yedges = np.histogram2d(descriptors_total[:,descriptor_dims[0]],
                                                descriptors_total[:,descriptor_dims[1]],
                                                bins=(xedges, yedges))

        return H_unweighted, xedges, yedges
    
    def weighted_RPE(self,n_bins_2d=100, descriptor_dims=None):
        if descriptor_dims is None:
            descriptor_dims = self.descriptor_dims

        descriptors_total, weights_total,shot_results_total = self.RPE.create_total_trainset()
        xedges, yedges = self.create_x_y_edges(n_bins_2d=n_bins_2d)

        # now look at the P_b based on the training data
        H_weighted, xedges, yedges = np.histogram2d(descriptors_total[:,descriptor_dims[0]],
                                                descriptors_total[:,descriptor_dims[1]],
                                                weights=weights_total,
                                                bins=(xedges, yedges))

        return H_weighted, xedges, yedges

    def model_output_RPE(self, plot_model, descriptors_total = None):
        if descriptors_total is None:
            descriptors_total, weights_total,shot_results_total = self.RPE.create_total_trainset()
        q_model_RPE = plot_model.log_prob(descriptors_total, use_transform=False, batch_size=None)
        p_B_model_RPE = 1/(1+np.exp(-q_model_RPE))
        return p_B_model_RPE, q_model_RPE
    
    def committor_model_2d_RPE(self,plot_model, n_bins_2d=100, descriptor_dims=None):
        if descriptor_dims is None:
            descriptor_dims = self.descriptor_dims

        descriptors_total, weights_total,shot_results_total = self.RPE.create_total_trainset()

        xedges, yedges = self.create_x_y_edges(n_bins_2d=n_bins_2d)

        p_B_model_RPE, q_model_RPE = self.model_output_RPE(plot_model=plot_model)
        # H_unweighted,_, _ = self.unweighted_PE(n_bins_2d=n_bins_2d,descriptor_dims=descriptor_dims)
        H_weighted, _,_ = self.weighted_RPE(n_bins_2d=n_bins_2d,descriptor_dims=descriptor_dims)
        H_p_b_model, xedges, yedges = np.histogram2d(descriptors_total[:,descriptor_dims[0]],
                                                descriptors_total[:,descriptor_dims[1]],
                                                bins=(xedges, yedges),
                                                weights = p_B_model_RPE[:,0]*weights_total)
        average_pb_model = H_p_b_model.T/H_weighted.T

        H_q_model, xedges, yedges = np.histogram2d(descriptors_total[:,descriptor_dims[0]],
                                        descriptors_total[:,descriptor_dims[1]],
                                        bins=(xedges, yedges),
                                        weights = q_model_RPE[:,0]*weights_total)
        
        # H_unweighted, _, _ = self.unweighted_PE(n_bins_2d=n_bins_2d,descriptor_dims=descriptor_dims)
        average_q_model = H_q_model.T/H_weighted.T
        # average_q_model = H_q_model.T/H_unweighted.T
        return average_pb_model, average_q_model, xedges, yedges
    
    def loss_histogram(self, plot_model,n_bins_2d=100,descriptor_dims=None):
        if descriptor_dims is None:
            descriptor_dims = self.descriptor_dims

        p_B_model_RPE, q_model_RPE = self.model_output_RPE(plot_model)
        descriptors_total , weights_total, shot_results_total = self.RPE.create_total_trainset()
        loss = snapshot_loss_original(torch.tensor(q_model_RPE),torch.tensor(weights_total),torch.tensor(shot_results_total))
        xedges, yedges = self.create_x_y_edges(n_bins_2d=n_bins_2d)

        H_unweighted,_, _ = self.unweighted_PE(n_bins_2d=n_bins_2d,descriptor_dims=descriptor_dims)

        H_loss, xedges, yedges = np.histogram2d(descriptors_total[:,descriptor_dims[0]],
                                                descriptors_total[:,descriptor_dims[1]],
                                                bins=(xedges, yedges),
                                                weights = loss)
        normalization= np.sum(weights_total)*(xedges[1]-xedges[0])*(yedges[1]-yedges[0])
        H_loss = H_loss/normalization
        return H_loss.T, xedges, yedges

    def plot_committor_model_RPE_data(self, plot_model, n_bins_2d=100, descriptor_dims=None, fig=None, ax=None):
        average_pb_model, average_q_model, xedges, yedges = self.committor_model_2d_RPE(plot_model,n_bins_2d=n_bins_2d, descriptor_dims=descriptor_dims)
        if ax is None:
            fig, ax = plt.subplots(1)
        im1 = ax.imshow(average_pb_model, interpolation='nearest', origin='lower',cmap=self.cmap,
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],alpha=0.6,aspect="auto")
        if fig is not None:
            fig.colorbar(im1)

    def plot_papb_model_RPE_data(self,plot_model, n_bins_2d=100, descriptor_dims=None, ax=None, fig=None, v_min_max=None):
        if descriptor_dims is None:
            descriptor_dims = self.descriptor_dims
        if v_min_max is not None:
            vmin = v_min_max[0]
            vmax = v_min_max[1]
        else:
            vmin=None
            vmax=None

        descriptors_total, weights_total,shot_results_total = self.RPE.create_total_trainset()

        xedges, yedges = self.create_x_y_edges(n_bins_2d=n_bins_2d)

        p_B_model_RPE, q_model_RPE = self.model_output_RPE(plot_model=plot_model)
        # H_unweighted,_, _ = self.unweighted_PE(n_bins_2d=n_bins_2d,descriptor_dims=descriptor_dims)
        # H_weighted, _,_ = self.weighted_RPE(n_bins_2d=n_bins_2d,descriptor_dims=descriptor_dims)
        H_papb_model, xedges, yedges = np.histogram2d(descriptors_total[:,descriptor_dims[0]],
                                                descriptors_total[:,descriptor_dims[1]],
                                                bins=(xedges, yedges),
                                                weights = p_B_model_RPE[:,0]*(1-p_B_model_RPE[:,0])*weights_total)
                                                # weights = weights_total)
        average_pb_model = H_papb_model.T

        # average_q_model = H_q_model.T/H_unweighted.T
        if ax is None:
            fig, ax = plt.subplots(1)
        im1 = ax.imshow(np.log(average_pb_model), interpolation='nearest', origin='lower',cmap=self.cmap,
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],alpha=0.6,aspect="auto", vmin=vmin,vmax=vmax)
        if fig is not None:
            fig.colorbar(im1)
        return im1


    def plot_committor_model_RPE_data_contours(self, plot_model, n_bins_2d=100, descriptor_dims=None, fig=None, ax=None):
        average_pb_model, average_q_model, xedges, yedges = self.committor_model_2d_RPE(plot_model,n_bins_2d=n_bins_2d, descriptor_dims=descriptor_dims)
        X, Y = np.meshgrid(xedges[1:], yedges[1:])
        if ax is None:
            fig, ax = plt.subplots(1)
        contour = ax.contour(X, Y,average_pb_model ,levels = self.levels_committor, cmap=self.cmap_RPE_committor_contours, linewidths=self.linewidth_committor_contour)
        ax.clabel(contour, inline=1, fontsize=10)
        if fig is not None:
            fig.colorbar(contour)
      
    def committor_model_2d_RPE_data(self, plot_model,n_bins_2d=100, descriptor_dims=None,  fig=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1)
        
        self.plot_potential_contour(ax=ax)
        self.plot_committor_model_RPE_data(plot_model, n_bins_2d=n_bins_2d, descriptor_dims=self.descriptor_dims, ax=ax)
        self.plot_committor_model_RPE_data_contours(plot_model, n_bins_2d=n_bins_2d, descriptor_dims=self.descriptor_dims, ax=ax)
        ax.set_title(r'$P_b$ model on RPE data')
        
    def plot_q_model_RPE_data(self, plot_model, n_bins_2d=100, v_min_max = None, descriptor_dims=None, fig=None, ax=None):
        average_pb_model, average_q_model, xedges, yedges = self.committor_model_2d_RPE(plot_model,n_bins_2d=n_bins_2d, descriptor_dims=descriptor_dims)
        if ax is None:
            fig, ax = plt.subplots(1)
        if v_min_max is not None:
            vmin = v_min_max[0]
            vmax = v_min_max[1]
            norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        else:
            norm=None
        # extent = [np.mean(xedges[0:2]), np.mean(xedges[-2:]), np.mean(yedges[0:2]), np.mean(yedges[-2:])]
        extent = [xedges[0], xedges[-1], yedges[0],yedges[-1]]

        im1 = ax.imshow(average_q_model, interpolation='nearest', origin='lower',cmap=self.cmap,
                extent=extent,alpha=0.8,aspect="auto",
                norm=norm)
        if fig is not None:
            fig.colorbar(im1)
        return im1

    def plot_q_model_RPE_data_contours(self, plot_model, n_bins_2d=100, descriptor_dims=None, fig=None, ax=None,v_min_max=15, colorbar=True ):
        if v_min_max is int or float:
            v_min_max = [-v_min_max, v_min_max]
        if len(v_min_max)==2:
            vmin = v_min_max[0]
            vmax = v_min_max[1]
        else:
            vmin= None
            vmax= None

        average_pb_model, average_q_model, xedges, yedges = self.committor_model_2d_RPE(plot_model,n_bins_2d=n_bins_2d, descriptor_dims=descriptor_dims)
        xedges_center = (xedges[1:]+xedges[:-1])/2
        yedges_center = (yedges[1:]+yedges[:-1])/2
        X, Y = np.meshgrid(xedges_center, yedges_center)

        if ax is None:
            fig, ax = plt.subplots(1)
        contour = ax.contour(X, Y,average_q_model ,levels = self.levels_q_model, cmap=self.cmap_RPE_committor_model_contours,vmax=vmax, vmin=vmin, alpha=self.alpha_committor, linewidths=self.linewidth_committor_contour)
        ax.clabel(contour, inline=1, fontsize=16)
        if fig is not None and colorbar:
            cb = fig.colorbar(contour)
            cb.set_label("RPE data q contour")
   
    def q_model_2d_RPE_data(self, plot_model,n_bins_2d=100, descriptor_dims=None,  fig=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1)
        
        self.plot_potential_contour(ax=ax)
        self.plot_q_model_RPE_data(plot_model, n_bins_2d=n_bins_2d, descriptor_dims=self.descriptor_dims, ax=ax)
        self.plot_q_model_RPE_data_contours(plot_model, n_bins_2d=n_bins_2d, descriptor_dims=self.descriptor_dims, ax=ax)

    def plot_committor_RPE_data(self, n_bins_2d=100, descriptor_dims=[0,1], fig=None, ax=None):
        p_B_histogram, xedges, yedges = self.weighted_committor_RPE(n_bins_2d=n_bins_2d, descriptor_dims=descriptor_dims)
        if ax is None:
            fig, ax = plt.subplots(1)
        im1 = ax.imshow(p_B_histogram, interpolation='nearest', origin='lower',cmap=self.cmap_committor,
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], alpha=self.alpha_committor, aspect="auto")
        if fig is not None:
            fig.colorbar(im1)
        

    def plot_committor_RPE_data_contours(self, n_bins_2d=100, descriptor_dims=[0,1], fig=None, ax=None):
        p_B_histogram, xedges, yedges = self.weighted_committor_RPE(n_bins_2d=n_bins_2d, descriptor_dims=descriptor_dims)
        X, Y = np.meshgrid(xedges[1:], yedges[1:])
        if ax is None:
            fig, ax = plt.subplots(1)
        contour = ax.contour(X, Y,p_B_histogram ,levels = self.levels_committor, cmap=self.cmap_RPE_committor_contours, alpha=self.alpha_committor, linewidths=self.linewidth_committor_contour)
        ax.clabel(contour, inline=1, fontsize=10)
        if fig is not None:
            fig.colorbar(contour)

    def committor_2d_RPE_data(self,n_bins_2d=100, levels = np.arange(0,1,0.2), fig=None, ax=None):
        if ax is None:
            fig_, ax = plt.subplots(1)
        
        self.plot_potential_contour(ax=ax)
        self.plot_committor_RPE_data(n_bins_2d=n_bins_2d, descriptor_dims=self.descriptor_dims, ax=ax,fig=fig)
        self.plot_committor_RPE_data_contours(n_bins_2d=n_bins_2d, descriptor_dims=self.descriptor_dims, ax=ax)
        ax.set_title(r'$P_b$ contour of the data')

    def plot_q_RPE_data(self, n_bins_2d=100, descriptor_dims=[0,1], v_min_max = 15,fig=None, ax=None):
        if v_min_max is int or float:
            v_min_max = [-v_min_max, v_min_max]
        if len(v_min_max)==2:
            vmin = v_min_max[0]
            vmax = v_min_max[1]
        else:
            vmin= None
            vmax= None
        norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

        p_B_histogram, xedges, yedges = self.weighted_committor_RPE(n_bins_2d=n_bins_2d, descriptor_dims=descriptor_dims)
        with np.errstate(divide='ignore'):
            q_histogram = -np.log(1/p_B_histogram-1)
        if ax is None:
            fig, ax = plt.subplots(1)
        im1 = ax.imshow(q_histogram, interpolation='nearest', origin='lower',cmap=self.cmap_committor,
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], alpha=self.alpha_committor, aspect="auto", norm=norm)
        if fig is not None:
            cb = fig.colorbar(im1)
            cb.set_label(r"$q(x)$ RPE")
        return im1
        
    def plot_q_RPE_data_contours(self, n_bins_2d=100, descriptor_dims=[0,1], v_min_max = 15,fig=None, ax=None):
        if v_min_max is int or float:
            v_min_max = [-v_min_max, v_min_max]
        if len(v_min_max)==2:
            vmin = v_min_max[0]
            vmax = v_min_max[1]
        else:
            vmin= None
            vmax= None

        p_B_histogram, xedges, yedges = self.weighted_committor_RPE(n_bins_2d=n_bins_2d, descriptor_dims=descriptor_dims)
        with np.errstate(divide='ignore'):
            q_histogram = -np.log(1/p_B_histogram-1)
        xedges_center = (xedges[1:]+xedges[:-1])/2
        yedges_center = (yedges[1:]+yedges[:-1])/2
        X, Y = np.meshgrid(xedges_center, yedges_center)

        if ax is None:
            fig, ax = plt.subplots(1)
        contour = ax.contour(X, Y,q_histogram ,levels = self.levels_q_data, cmap=self.cmap_RPE_committor_contours,vmax=vmax, vmin=vmin, alpha=self.alpha_committor, linewidths=self.linewidth_committor_contour)
        ax.clabel(contour, inline=1, fontsize=16)
        if fig is not None:
            cb = fig.colorbar(contour)
            cb.set_label(r"$q(x)$ RPE")
    
    def plot_loss(self,plot_model,ax=None, fig=None):
        H_loss, xedges, yedges = self.loss_histogram(plot_model)
        if ax is None:
            fig, ax = plt.subplots(1)
        im1 = ax.imshow(np.log(H_loss), interpolation='nearest', origin='lower',cmap=self.cmap_committor,
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], vmin=-16,vmax=-6, alpha=self.alpha_committor,aspect="auto")
        if fig is not None:
            cb = fig.colorbar(im1)
            cb.set_label(r"log loss")
    
    def plot_pdatalnpmodel(self,plot_model,ax=None, fig=None):
        H_loss, xedges, yedges = self.loss_histogram(plot_model)
        if self.H_full is None:
            self.full_RPE_histogram()
        H_pdatalnpmodel = H_loss/self.H_full.T
        if ax is None:
            fig, ax = plt.subplots(1)
        im1 = ax.imshow(np.log(H_pdatalnpmodel), interpolation='nearest', origin='lower',cmap=self.cmap_theory,
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], vmin=-16,vmax=-6, alpha=self.alpha_committor,aspect="auto")
        if fig is not None:
            cb = fig.colorbar(im1)
            cb.set_label(r"log loss")
    
    def plot_loss_contour(self,plot_model,ax=None, fig=None):
        H_loss, xedges, yedges = self.loss_histogram(plot_model)
        X, Y = np.meshgrid(xedges[1:], yedges[1:])
        levels_loss = np.arange(-20,0,2)
        if ax is None:
            fig, ax = plt.subplots(1)
        contour = ax.contour(X, Y,np.log(H_loss) ,cmap=self.cmap_theory,alpha=self.alpha_committor, levels=levels_loss, linewidths=self.linewidth_committor_contour, colors=self.color_theory)
        ax.clabel(contour, inline=1, fontsize=10)
        if fig is not None:
            cb = fig.colorbar(contour)
            cb.set_label(r"log loss")

    def plot_loss_theory(self,theoretical_committor_path, n_x,n_y=None,ax=None, fig=None):
        """
        Plot the contour of the loss allong the theoretical q function.
        """
        if n_y is None:
            n_y = n_x
        p_theory, q_theory, xedges, yedges = self.load_theoretical_committor(theoretical_committor_path, n_x, n_y)
        x_2d, y_2d, U = self.pes.plot_2d_pes(xedges, yedges)
        plnp = -(p_theory*np.log(p_theory)+(1-p_theory)*np.log(1-p_theory))

        weight = np.exp(-self.beta*U)
        norm = np.sum(weight)*(xedges[1]-xedges[0])*(yedges[1]-yedges[0])
        rho = weight/norm
        loss = rho*plnp

        if ax is None:
            fig, ax = plt.subplots(1)
        im1 = ax.imshow(np.log(loss), interpolation='nearest', origin='lower',cmap=self.cmap_committor,
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],vmin=-16,vmax=-6, alpha=self.alpha_committor,aspect="auto")
        if fig is not None:
            cb = fig.colorbar(im1)
            cb.set_label(r"log loss theory ideal")

    def plot_plnp_theory(self,theoretical_committor_path, n_x,n_y=None,ax=None, fig=None):
        """
        Plot the contour of the loss allong the theoretical q function.
        """
        if n_y is None:
            n_y = n_x
        p_theory, q_theory, xedges, yedges = self.load_theoretical_committor(theoretical_committor_path, n_x, n_y)
        x_2d, y_2d, U = self.pes.plot_2d_pes(xedges, yedges)
        plnp = -(p_theory*np.log(p_theory)+(1-p_theory)*np.log(1-p_theory))
        weight = np.exp(-self.beta*U)
        norm = np.sum(weight)*(xedges[1]-xedges[0])*(yedges[1]-yedges[0])
        rho = weight/norm
        loss = rho*plnp

        if ax is None:
            fig, ax = plt.subplots(1)
        im1 = ax.imshow(np.log(plnp), interpolation='nearest', origin='lower',cmap=self.cmap_theory,
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],vmin=-20, alpha=self.alpha_committor,aspect="auto")
        if fig is not None:
            cb = fig.colorbar(im1)
            cb.set_label(r"$log of $pLn(p)$ theory ideal")
        
    def plot_plnp_theory_contour(self,theoretical_committor_path, n_x,n_y=None,ax=None, fig=None):
        """
        Plot the contour of the loss allong the theoretical q function.
        """
        if n_y is None:
            n_y = n_x
        p_theory, q_theory, xedges, yedges = self.load_theoretical_committor(theoretical_committor_path, n_x, n_y)
        X, Y = np.meshgrid(xedges, yedges)

        x_2d, y_2d, U = self.pes.plot_2d_pes(xedges, yedges)
        plnp = -(p_theory*np.log(p_theory)+(1-p_theory)*np.log(1-p_theory))
        weight = np.exp(-self.beta*U)
        norm = np.sum(weight)*(xedges[1]-xedges[0])*(yedges[1]-yedges[0])
        rho = weight/norm
        loss = rho*plnp

        if ax is None:
            fig, ax = plt.subplots(1)
        contour = ax.contour(X, Y,np.log(plnp) ,cmap=self.cmap_theory,alpha=self.alpha_theory, linewidths=self.linewidth_theory, colors=self.color_theory)
        ax.clabel(contour, inline=1, fontsize=10)
    
        if fig is not None:
            cb = fig.colorbar(contour)
            cb.set_label(r"log $pln(p)$ theory ideal")
    
    def plot_loss_theory_contour(self,theoretical_committor_path, n_x,n_y=None,ax=None, fig=None):
        """
        Plot the contour of the loss allong the theoretical q function.
        """
        if n_y is None:
            n_y = n_x
        p_theory, q_theory, xedges, yedges = self.load_theoretical_committor(theoretical_committor_path, n_x, n_y)
        X, Y = np.meshgrid(xedges, yedges)

        x_2d, y_2d, U = self.pes.plot_2d_pes(xedges, yedges)
        plnp = -(p_theory*np.log(p_theory)+(1-p_theory)*np.log(1-p_theory))

        weight = np.exp(-self.beta*U)
        norm = np.sum(weight)*(xedges[1]-xedges[0])*(yedges[1]-yedges[0])
        rho = weight/norm
        loss = rho*plnp
        levels_loss = np.arange(-20,0,2)
        if ax is None:
            fig, ax = plt.subplots(1)
        contour = ax.contour(X, Y,np.log(loss) ,levels=levels_loss, cmap=self.cmap_theory,alpha=self.alpha_theory, linewidths=self.linewidth_theory, colors=self.color_theory)
        ax.clabel(contour, inline=1, fontsize=10)
        if fig is not None:
            cb = fig.colorbar(contour)
            cb.set_label(r"log loss theory ideal")

    def q_2d_RPE_data(self,n_bins_2d=100, fig=None, ax=None):
        if ax is None:
            fig_, ax = plt.subplots(1)
        
        self.plot_potential_contour(ax=ax)
        self.plot_q_RPE_data(n_bins_2d=n_bins_2d, descriptor_dims=self.descriptor_dims, ax=ax,fig=fig)
        self.plot_q_RPE_data_contours(n_bins_2d=n_bins_2d, descriptor_dims=self.descriptor_dims, ax=ax)
        ax.set_title(r'$q$ contour of the data')
        
    def plot_free_energy_RPE_allong_q_model(self,plot_model, q_bins=None, fig=None, ax=None):
        p_B_model_RPE, q_model_RPE = self.model_output_RPE(plot_model)
        _ , weights_total, _ = self.RPE.create_total_trainset()
        if q_bins is None:
            n_bins = 100
            q_bins = np.linspace(-20,20,n_bins)
        prob_allong_q ,edges= np.histogram(q_model_RPE[:,0],
                                                       bins=q_bins, weights = weights_total, density=True)
        if ax is None:
            fig, ax = plt.subplots(1,1)
        ax.plot(edges[1:],-np.log(prob_allong_q)+np.log(prob_allong_q[int(n_bins/2)]), label="model")
    
    def plot_free_energy_RPE_allong_committor_model(self,plot_model, fig=None, ax=None):
        p_B_model_RPE, q_model_RPE = self.model_output_RPE(plot_model)
        _ , weights_total, _ = self.RPE.create_total_trainset()
        
        n_bins = 80
        committor_bins = np.linspace(0,1,n_bins)
        prob_allong_committor ,edges= np.histogram(p_B_model_RPE[:,0],
                                                       bins=committor_bins, weights = weights_total, density=True)
        if ax is None:
            fig, ax = plt.subplots(1,1)
        ax.plot(edges[1:],-np.log(prob_allong_committor)+np.log(prob_allong_committor[int(n_bins/2)]), label="model")

    def plot_loss_allong_q_model(self, plot_model,ax=None, fig=None, density=False,n_bins=100,color="black", normalized=False):
        p_B_model_RPE, q_model_RPE = self.model_output_RPE(plot_model)
        _ , weights_total, shot_results_total = self.RPE.create_total_trainset()
        weights_total = weights_total/np.min(weights_total)
        if normalized:
            weights_total = weights_total/np.sum(np.sum(shot_results_total, axis=-1) * weights_total)
        
        loss = snapshot_loss_original(torch.tensor(q_model_RPE),torch.tensor(weights_total),torch.tensor(shot_results_total))
        if density==True:
            density_indicator = "*"
        else:
            density_indicator = ""
        print("loss original log likelihood loss: ", np.sum(loss.detach().numpy()))

        prob_allong_q ,edges= np.histogram(q_model_RPE[:,0],
                                                       bins=n_bins, weights = loss, density=density)
        if ax is None:
            fig, ax = plt.subplots(1,1)
        ax.plot(edges[1:],prob_allong_q, label=r"$\mathcal{L}(q')$ "+ "{}".format(density_indicator),color=color)


    def plot_loss_vs_weights(self, plot_model,ax=None, fig=None,normalized=False):
        p_B_model_RPE, q_model_RPE = self.model_output_RPE(plot_model)
        descriptors_total , weights_total, shot_results_total = self.RPE.create_total_trainset()
        mask = (descriptors_total[:,0] > 0.05) | (descriptors_total[:,0] < 1.1) 
        print("mask: ", mask)
        print("weights_total: ", weights_total[mask])
        weights_total = weights_total[mask]/np.min(weights_total[mask])
        if normalized:
            weights_total = weights_total/np.sum(np.sum(shot_results_total, axis=-1) * weights_total)
        loss = snapshot_loss_original(torch.tensor(q_model_RPE[mask]),torch.tensor(weights_total),torch.tensor(shot_results_total[mask]))

        ax.scatter(np.log(weights_total),loss)

    
    def plot_loss_distribution(self, plot_model,ax=None, fig=None,normalized=False, reactive=True, distance_range=[0.05,1.1],bins=100):
        p_B_model_RPE, q_model_RPE = self.model_output_RPE(plot_model)
        descriptors_total , weights_total, shot_results_total = self.RPE.create_total_trainset()
        print("distance range: ",distance_range)
        mask_states = (descriptors_total[:,0] > distance_range[0]) & (descriptors_total[:,0] < distance_range[1]) 
        if reactive:
            mask_reactive = (shot_results_total[:,0]==1) & (shot_results_total[:,1]==1)
            mask = mask_states & mask_reactive
        else:
            mask = mask_states
        print("mask: ", mask)
        print("weights_total: ", weights_total[mask])
        weights_total = weights_total[mask]/np.min(weights_total[mask])
        if normalized:
            weights_total = weights_total/np.sum(np.sum(shot_results_total[mask], axis=-1) * weights_total)
        
   

        loss = snapshot_loss_original(torch.tensor(q_model_RPE[mask]),torch.tensor(weights_total),torch.tensor(shot_results_total[mask]))
        print("loss max:", np.max(loss.detach().numpy()))
        descriptors_total = descriptors_total[mask]
        shot_results_total = shot_results_total[mask]
        q_model_RPE = q_model_RPE[mask]
        print("descriptors_total[np.argmax(loss),:]", descriptors_total[np.argmax(loss),:])
        print("shot results total: ", shot_results_total[np.argmax(loss),:])
        print("weights_total[np.argmax(loss)]", weights_total[np.argmax(loss)])
        print("q_model_RPE[np.argmax(loss),:]", q_model_RPE[np.argmax(loss)])


        print("loss minimum: ", np.min(loss.detach().numpy()))
        ax.hist(loss,bins=bins)



    def plot_loss_total_along_q(self, plot_model,ax=None, fig=None,n_bins = 100, normalized=False, ee_par_weight=1, ):
        p_B_model_RPE, q_model_RPE = self.model_output_RPE(plot_model)
        descriptors_total , weights_total, shot_results_total = self.RPE.create_total_trainset()
        weights_total = weights_total/np.min(weights_total)
        loss = snapshot_loss_original(torch.tensor(q_model_RPE),torch.tensor(weights_total),torch.tensor(shot_results_total))
        loss_smoothness = snapshot_loss_smoothness(plot_model.nnet,descriptors=descriptors_total)/(len(q_model_RPE))
        loss_total = loss + ee_par_weight*loss_smoothness
        print("loss total: ", np.mean(loss_total.detach().numpy()))
        normalization = np.sum(np.sum(shot_results_total, axis=-1) * weights_total) if normalized else 1
        print("normalization: ", normalization)
        loss = loss_total/(normalization)
        print("loss total normalized: ", np.mean(loss.detach().numpy()))
        prob_allong_q ,edges= np.histogram(q_model_RPE[:,0],
                                                       bins=n_bins, weights = loss, density=False)
        if ax is None:
            fig, ax = plt.subplots(1,1)
        ax.plot(edges[1:],prob_allong_q, label=r"$\mathcal{L}(q')$ total")
    

    def plot_loss_smoothness_along_q_model(self, plot_model,ax=None, fig=None,n_bins = 100, normalized=False, ee_par_weight=1, ):
        p_B_model_RPE, q_model_RPE = self.model_output_RPE(plot_model)
        descriptors_total , weights_total, shot_results_total = self.RPE.create_total_trainset()
        weights_total = weights_total/np.min(weights_total)
        loss = snapshot_loss_smoothness(plot_model.nnet,descriptors=descriptors_total)
        print("loss smoothness: ", np.mean(loss.detach().numpy()))
        normalization = np.sum(np.sum(shot_results_total, axis=-1) * weights_total) if normalized else 1
        print("normalization: ", normalization)
        loss = ee_par_weight*loss/(len(q_model_RPE)*normalization)
        print("loss smoothness normalized: ", np.mean(loss.detach().numpy()))
        prob_allong_q ,edges= np.histogram(q_model_RPE[:,0],
                                                       bins=n_bins, weights = loss, density=False)
        if ax is None:
            fig, ax = plt.subplots(1,1)
        ax.plot(edges[1:],prob_allong_q, label=r"$\mathcal{L}(q')$ smoothness")
    
    def plot_loss_scaled_q_allong_q_model(self, plot_model,ax=None, fig=None,n_bins = 100):
        p_B_model_RPE, q_model_RPE = self.model_output_RPE(plot_model)
        _ , weights_total, shot_results_total = self.RPE.create_total_trainset()
        loss = snapshot_loss_low_q_scaled(torch.tensor(q_model_RPE),torch.tensor(weights_total),torch.tensor(shot_results_total))
        print("loss scaled q: ", np.sum(loss.detach().numpy())/np.sum(weights_total))
        
        prob_allong_q ,edges= np.histogram(q_model_RPE[:,0],
                                                       bins=n_bins, weights = loss/np.sum(weights_total), density=False)
        if ax is None:
            fig, ax = plt.subplots(1,1)
        ax.plot(edges[1:],prob_allong_q, label=r"$\mathcal{L}(q')(1+9e^{-q'^2/5})$")
    
    def plot_loss_scaled_weight_sqrtrho(self, plot_model,ax=None, fig=None,n_bins = 100):
        p_B_model_RPE, q_model_RPE = self.model_output_RPE(plot_model)
        _ , weights_total, shot_results_total = self.RPE.create_total_trainset()
        loss = snapshot_loss_sqrt_rho_weight(torch.tensor(q_model_RPE),torch.tensor(weights_total),torch.tensor(shot_results_total))
        print(r"loss devided by $1/\sqrt{\rho_{RPE}(q)}$)): ", np.sum(loss.detach().numpy())/np.sum(weights_total))
        prob_allong_q ,edges= np.histogram(q_model_RPE[:,0],
                                                       bins=n_bins, weights = loss/np.sum(weights_total), density=False)
        if ax is None:
            fig, ax = plt.subplots(1,1)
        ax.plot(edges[1:],prob_allong_q, label=r"$\ln \mathcal{L}(q')/\sqrt{\rho_{RPE}(q')}$")

    def plot_loss_normalized_q_allong_q_model(self, plot_model,ax=None, fig=None, density=False,n_bins = 100,color="black",normalized=False):
        p_B_model_RPE, q_model_RPE = self.model_output_RPE(plot_model)
        _ , weights_total, shot_results_total = self.RPE.create_total_trainset()
        weights_total = weights_total/np.min(weights_total)
        H_q, q_bins = np.histogram(q_model_RPE, bins=100, density=True)
        q_bin_indices = np.digitize(q_model_RPE, q_bins[:-1])-1
        normalization = 1/np.sum(np.sum(shot_results_total, axis=-1) * weights_total) if normalized else 1
        scaling = torch.tensor(np.nan_to_num(1/H_q[q_bin_indices])[:,0]).float().numpy()
        weights_total = weights_total*scaling*normalization
        loss = snapshot_loss_original(torch.tensor(q_model_RPE),torch.tensor(weights_total),torch.tensor(shot_results_total))
        density_indicator_loss = ""
        if density==True:
            density_indicator = "*"
        else:
            density_indicator = ""
        loss_allong_q ,edges= np.histogram(q_model_RPE[:,0],
                                                       bins=n_bins, weights = loss)
        prob_allong_q ,edges= np.histogram(q_model_RPE[:,0],
                                                bins=n_bins, density=density)
        
 
        print(r"loss devided by $1/rho_PE(q))$: ", np.sum((loss).detach().numpy()))

        if ax is None:
            fig, ax = plt.subplots(1,1)
        ax.plot(edges[1:],loss_allong_q, label=r"$ \mathcal{L}(q')$"+ "{}".format(density_indicator_loss)+r"$/rho^{PE}(q')$"+ "{}".format(density_indicator),color=color)


    def plot_distribution_of_points_allong_q_model(self, plot_model,ax=None, fig=None, density=False,n_bins = 100):
        p_B_model_RPE, q_model_RPE = self.model_output_RPE(plot_model)
        if density==True:
            density_indicator = "*"
        else:
            density_indicator = ""
        prob_allong_q ,edges= np.histogram(q_model_RPE[:,0],
                                                       bins=n_bins, density=density)
        if ax is None:
            fig, ax = plt.subplots(1,1)
        ax.plot(edges[1:],prob_allong_q, label=r"$\rho^{PE}(q')$ "+"{}".format(density_indicator))

    def plot_pAandpB_of_RPE_data_allong_q_model(self, plot_model,ax=None, fig=None,n_bins=100):
        p_B_model_RPE, q_model_RPE = self.model_output_RPE(plot_model)
        _ , weights_total, shot_results_total = self.RPE.create_total_trainset()
        rhoA_allong_q ,edges= np.histogram(q_model_RPE[:,0],
                                                       bins=n_bins, weights=weights_total*shot_results_total[:,0], density=False)
        rhoB_allong_q, edges= np.histogram(q_model_RPE[:,0],
                                                       bins=edges, weights=weights_total*shot_results_total[:,1], density=False)
        p_A_allong_q = rhoA_allong_q/(rhoA_allong_q+rhoB_allong_q)
        p_B_allong_q = rhoB_allong_q/(rhoA_allong_q+rhoB_allong_q)


        if ax is None:
            fig, ax = plt.subplots(1,1)
        ax.plot(edges[1:],p_A_allong_q, label=r"$p_A^{RPE}(q')$")
        ax.plot(edges[1:],p_B_allong_q, label=r"$p_B^{RPE}(q')$")
        return edges, p_A_allong_q, p_B_allong_q

    def plot_plnp_allong_q_model(self, plot_model,ax=None, fig=None,n_bins=100):
        p_B_model_RPE, q_model_RPE = self.model_output_RPE(plot_model)
        _ , weights_total, shot_results_total = self.RPE.create_total_trainset()
        weights_total *= 1/np.sum(weights_total)
        loss = snapshot_loss_original(torch.tensor(q_model_RPE),torch.tensor(weights_total), torch.tensor(shot_results_total))
        prob_allong_q ,edges= np.histogram(q_model_RPE[:,0],
                                                       bins=n_bins, weights = loss, density=False)
        n_q ,edges= np.histogram(q_model_RPE[:,0],
                                                bins=n_bins,weights = weights_total, density=False)
        if ax is None:
            fig, ax = plt.subplots(1,1)
        ax.plot(edges[1:],prob_allong_q/n_q, label=r"$p^{RPE}(q')\ln p^{model}(q')$")
    
    def plot_weight_allong_q_model(self, plot_model,ax=None, fig=None, density=False,n_bins=100):
        p_B_model_RPE, q_model_RPE = self.model_output_RPE(plot_model)
        _ , weights_total, shot_results_total = self.RPE.create_total_trainset()
        weights_total *= 1/np.sum(weights_total)

        if density==True:
            density_indicator = "*"
        else:
            density_indicator = ""
        prob_allong_q ,edges= np.histogram(q_model_RPE[:,0],bins=n_bins, weights = weights_total,density=density)
        if ax is None:
            fig, ax = plt.subplots(1,1)
        ax.plot(edges[1:],prob_allong_q, label=r"$\rho^{RPE}(q')$ "+"{}".format(density_indicator))

    def compute_components_theoretical_loss(self, theoretical_committor_path, n_x, n_y=None):
        if n_y is None:
            n_y = n_x
        p_theory, q_theory, xedges, yedges = self.load_theoretical_committor(theoretical_committor_path, n_x, n_y)
        x_2d, y_2d, U = self.pes.plot_2d_pes(xedges, yedges)
        plnp = -(p_theory[:,:]*np.log(p_theory[:,:])+(1-p_theory)*np.log(1-p_theory))
        weight = np.exp(-self.beta*U)
        norm = np.sum(weight)*(xedges[1]-xedges[0])*(yedges[1]-yedges[0])
        rho = weight/norm
        loss = rho*plnp
        return x_2d, y_2d, p_theory, q_theory, loss, rho, plnp, norm

    def plot_loss_allong_theory_q(self, theoretical_committor_path, n_x, n_y=None, fig=None, ax=None, density=False,q_extent=[-20,20], q_flatten=True):
        """
        Plot the contour of the free energy allong the theoretical q function.
        """
        _,_, p_theory, q_theory, loss, rho, plnp, norm = self.compute_components_theoretical_loss(theoretical_committor_path,n_x,n_y)
        q_bins_high_res = np.arange(q_extent[0],q_extent[1],0.1)
        H_q, _ = np.histogram(q_theory.ravel(),
                                bins=q_bins_high_res, density=False)
        q_bins = np.arange(q_extent[0],q_extent[1],0.1)
        if q_flatten == True:
            q_bin_indices = np.digitize(q_theory, q_bins_high_res[:-1])-1
            H_loss, _ = np.histogram(q_theory.ravel(),
                                    bins=q_bins, 
                                    weights = np.nan_to_num(loss/H_q[q_bin_indices]).ravel(),density=density)
        else: 
            H_loss, _ = np.histogram(q_theory.ravel(),
                                    bins=q_bins, 
                                    weights = np.nan_to_num(loss).ravel(),density=density)
        ax.plot(q_bins[:-1], H_loss, label=r"$\ln \mathcal{L}_{theory}(q_{theory})$ ")

    def plot_plnp_allong_theory_q(self, theoretical_committor_path, n_x, n_y=None, fig=None, ax=None, density=False, q_extent=[-20,20], q_flatten=True):
        """
        Plot plnp component of the loss allong the theoretical q function.
        """
        _,_, p_theory, q_theory, loss, rho, plnp, norm = self.compute_components_theoretical_loss(theoretical_committor_path,n_x,n_y)
        q_bins_high_res = np.arange(q_extent[0],q_extent[1],0.1)
        H_q, _ = np.histogram(q_theory.ravel(),
                                bins=q_bins_high_res, density=False)
        q_bins = np.arange(q_extent[0],q_extent[1],0.1)
        if q_flatten == True:
            q_bin_indices = np.digitize(q_theory, q_bins_high_res[:-1])-1
            H_plnp, _ = np.histogram(q_theory.ravel(),
                                    bins=q_bins, weights=np.nan_to_num(plnp/H_q[q_bin_indices]).ravel(), density=False)
        else: 
            H_plnp, _ = np.histogram(q_theory.ravel(),
                                    bins=q_bins, weights=np.nan_to_num(plnp).ravel(), density=False)
        ax.plot(q_bins[:-1], H_plnp, label=r"$p_{model} \ln p_{model}(q')$ theory")
    
    def plot_weight_allong_theory_q(self, theoretical_committor_path, n_x, n_y=None, fig=None, ax=None, density=False,q_extent=[-20,20], q_flatten=True):
        """
        Plot the theoretical distribution allong the theoretical q function.
        """
        _,_, p_theory, q_theory, loss, rho, plnp, norm = self.compute_components_theoretical_loss(theoretical_committor_path,n_x,n_y)
        q_bins_high_res = np.arange(q_extent[0],q_extent[1],0.1)
        H_q, _ = np.histogram(q_theory.ravel(),
                                bins=q_bins_high_res, density=False)
        q_bins = np.arange(q_extent[0],q_extent[1],0.1)
        if q_flatten == True:
            q_bin_indices = np.digitize(q_theory, q_bins_high_res[:-1])-1
            H_rho, _ = np.histogram(q_theory.ravel(),
                                    bins=q_bins, weights=np.nan_to_num(rho/H_q[q_bin_indices]).ravel(), density=density)
        else: 
            H_rho, _ = np.histogram(q_theory.ravel(),
                                    bins=q_bins, weights=np.nan_to_num(rho).ravel(), density=density)
        ax.plot(q_bins[:-1], H_rho, label=r"$\rho_{theory}(q_{theory})$")

    def plot_distribution_of_q_points_allong_theory_q(self, theoretical_committor_path, n_x, n_y=None, fig=None, ax=None, U_max=12, q_extent=[-20,20], q_flatten=True):
        """
        Plot the theoretical distribution allong the theoretical q function.
        """
        _,_, p_theory, q_theory, loss, rho, plnp, norm = self.compute_components_theoretical_loss(theoretical_committor_path,n_x,n_y)
        q_bins_high_res = np.arange(q_extent[0],q_extent[1],0.1)
        H_q, _ = np.histogram(q_theory.ravel(),
                                bins=q_bins_high_res, density=False)
        q_bins = np.arange(q_extent[0],q_extent[1],0.1)
        if q_flatten == True:
            q_bin_indices = np.digitize(q_theory, q_bins_high_res[:-1])-1
            H_flatq, _ = np.histogram(q_theory.ravel(),
                                    bins=q_bins, weights=np.nan_to_num(1/H_q[q_bin_indices]).ravel(), density=False)
        else: 
            H_flatq, _ = np.histogram(q_theory.ravel(),
                                    bins=q_bins, density=False)
        ax.plot(q_bins[:-1], H_flatq, label=r"n(q_{theory})")

    def plot_free_energy_allong_theory_q(self, theoretical_committor_path, n_x, n_y=None, fig=None, ax=None, U_max=12):
        """
        Plot the contour of the free energy allong the theoretical q function.
        """
        n_bins = 50
        q_bins = np.linspace(-20,20,n_bins)
        if n_y is None:
            n_y = n_x
        p_theory, q_theory, xedges, yedges = self.load_theoretical_committor(theoretical_committor_path, n_x, n_y)
        x_2d, y_2d, U = self.pes.plot_2d_pes(xedges, yedges)
        index_low_U = np.where(np.reshape(self.beta*(U-np.min(U)),-1)<U_max)
        # index_low_U = np.where(-np.reshape(x_2d,-1)>np.reshape(y_2d,-1))
        print(index_low_U)
        prob_allong_q ,edges= np.histogram(np.reshape(q_theory,-1)[index_low_U],
                                                bins=q_bins, weights = np.reshape(np.exp(-self.beta*U),-1)[index_low_U], density=False)
        if ax is None:
            fig, ax = plt.subplots(1,1)
        ax.plot(edges[1:],-np.log(prob_allong_q)+np.log(prob_allong_q[int(n_bins/2)]), label="theory")

    def plot_free_energy_allong_theory_committor(self, theoretical_committor_path, n_x, n_y=None, fig=None, ax=None, U_max=12):
        """
        Plot the contour of the free energy allong the theoretical q function.
        """
        n_bins = 80
        committor_bins = np.linspace(0,1,n_bins)
        if n_y is None:
            n_y = n_x
        p_theory, q_theory, xedges, yedges = self.load_theoretical_committor(theoretical_committor_path, n_x, n_y)
        x_2d, y_2d, U = self.pes.plot_2d_pes(xedges, yedges)
        index_low_U = np.where(np.reshape(self.beta*(U-np.min(U)),-1)<U_max)
        # index_low_U = np.where(-np.reshape(x_2d,-1)>np.reshape(y_2d,-1))
        prob_allong_committor ,edges= np.histogram(np.reshape(p_theory,-1)[index_low_U],
                                                bins=committor_bins, weights = np.reshape(np.exp(-self.beta*U),-1)[index_low_U], density=True)
        if ax is None:
            fig, ax = plt.subplots(1,1)
        ax.plot(edges[1:],-np.log(prob_allong_committor)+np.log(prob_allong_committor[int(n_bins/2)]), label="theory",alpha=0.6)

    def plot_RPE_along_x(self,fig=None, ax=None):
        n_bins=100
        H, x = self.RPE_histogram_allong(n_bins,self.dims_extent[0:2],0)
        if ax is None:
            fig,ax = plt.subplots(1,1)
        y =-np.log(H)
        ax.plot(x[1:],y-y[int(n_bins/2)],label="RPE")
    
    def plot_potential_1d(self,extent=None,fig=None, ax=None):
        n_bins=100
        if extent is None:
            extent = self.dims_extent[0:2]
        x = np.linspace(extent[0],extent[1],n_bins)
        x, U = self.pes.plot_1d_pes(x)
        if ax is None:
            fig,ax = plt.subplots(1,1)
        y =self.beta*U
        ax.plot(x,y-y[int(n_bins/2)],label="Theory")
    
    def scatter_rc_path(self, plot_model, q_bins=np.arange(-12,12,0.1), descriptor_dims=[0,1], ax=None,fig=None):
        descriptors_total, weights_total, shot_results_total = self.RPE.create_total_trainset()
        p_B_model_RPE, q_model_RPE = self.model_output_RPE(plot_model)
        normalization,q_bins = np.histogram(q_model_RPE[:,0], bins=q_bins, weights=weights_total)
        print(normalization)
        x_value_allong_q = np.zeros((len(descriptor_dims),len(q_bins)-1))
        for i, dim in enumerate(descriptor_dims):
            pathx,q_bins = np.histogram(q_model_RPE[:,0], bins=q_bins, weights=weights_total*descriptors_total[:,dim])
            x_value_allong_q[i,:]= pathx/normalization
            print(x_value_allong_q[i,:])

        if ax is None:
            fig,ax=plt.subplots(1,1)
        ax.scatter(x_value_allong_q[0,:],x_value_allong_q[1,:], color="black",s=2 )
    
    def scatter_rc_minima_path(self, plot_model, q_bins=np.arange(-12,12,0.1), descriptor_dims=[0,1], ax=None,fig=None, rolling_mean=1):
        descriptors_total, weights_total, shot_results_total = self.RPE.create_total_trainset()
        p_B_model_RPE, q_model_RPE = self.model_output_RPE(plot_model)
        ind_bin_x = np.digitize(q_model_RPE, q_bins, right=False)
        # normalization,q_bins = np.histogram(q_model_RPE[:,0], bins=q_bins, weights=weights_total)
        x_value_allong_q = np.zeros((2,len(q_bins)-1))
        # print(np.shape(ind_bin_x))
        # print(ind_bin_x)
        if ax is None:
            fig,ax=plt.subplots(1,1)
        for i_dim in range(2):
            # print(dim)
            for bin in range(len(q_bins[1:])):
                ind_x_bin = np.where(ind_bin_x==bin)[0]
                # print("descriptors:", np.shape(descriptors_total))
                # print("ind_x_bin:", np.shape(ind_x_bin))
                # print(ind_x_bin)
                # print(dim)
                pathx,bins= np.histogram(descriptors_total[ind_x_bin,descriptor_dims[i_dim]], bins=100, weights=weights_total[ind_x_bin], density=True)
                x_value_allong_q[i_dim,bin]= bins[np.argmax(pathx)]
        ax.scatter(self.moving_average(x_value_allong_q[0,:],rolling_mean),self.moving_average(x_value_allong_q[1,:],rolling_mean), color="black",s=3, alpha=0.3+0.7*(bin/len(q_bins)) )
        
    def scatter_rc_deriv_rho(self, plot_model, q_bins=np.linspace(-12,12,80), descriptor_dims=[0,1], ax=None,fig=None):
        descriptors_total, weights_total, shot_results_total = self.RPE.create_total_trainset()
        p_B_model_RPE, q_model_RPE = self.model_output_RPE(plot_model)
        ind_bin_x = np.digitize(q_model_RPE, q_bins, right=False)
        # normalization,q_bins = np.histogram(q_model_RPE[:,0], bins=q_bins, weights=weights_total)
        x_value_allong_q = np.zeros((2,len(q_bins)-1))
        # print(np.shape(ind_bin_x))
        # print(ind_bin_x)
        if ax is None:
            fig,ax=plt.subplots(1,1)
        for dim in range(2):
            # print(dim)
            for bin in range(len(q_bins[1:])):
                ind_x_bin = np.where(ind_bin_x==bin)[0]
                # print("descriptors:", np.shape(descriptors_total))
                # print("ind_x_bin:", np.shape(ind_x_bin))
                # print(ind_x_bin)
                # print(dim)
                pathx,bins= np.histogram(descriptors_total[ind_x_bin,dim], bins=100, weights=weights_total[ind_x_bin], density=True)
                gradient = np.gradient(pathx)
                x_value_allong_q[dim,bin]= bins[np.argmax(pathx)]
        ax.scatter(self.moving_average(x_value_allong_q[0,:],5),self.moving_average(x_value_allong_q[1,:],5), color="black",s=3 )
    
    def mean_descriptors_along_q_normalized_per_q_reactive_paths(self, plot_model, n_descriptors=22, q_bins=np.linspace(-12,12,40),ax=None,fig=None, rolling_mean=1, plot=True):
        descriptors_total, weights_total, shot_results_total = self.RPE.create_total_trainset()
        p_B_model_RPE, q_model_RPE = self.model_output_RPE(plot_model)
        x_value_allong_q = np.zeros((n_descriptors,len(q_bins)-1))
        x_rolling_average = np.zeros((n_descriptors,len(q_bins)-1))

        indices = np.where(np.all(shot_results_total == 1, axis=1))[0]
        weight_TP = np.zeros(len(descriptors_total))
        weight_TP[indices] = 1
        normalization,bins = np.histogram(q_model_RPE[:,0], bins=q_bins, weights=weights_total*weight_TP, density=False)

        if ax is None and plot:
            fig,ax=plt.subplots(1,1)
        for dim in range(n_descriptors):
            pathx,bins= np.histogram(q_model_RPE[:,0], bins=q_bins, weights=descriptors_total[:,dim]*weights_total*weight_TP, density=False)
            x_value_allong_q[dim,:]= pathx/normalization
            x_rolling_average[dim,:] = self.moving_average(x_value_allong_q[dim,:],rolling_mean)
            if plot:
                q_bins_center = (q_bins[1:]-q_bins[:-1])/2
                ax.plot(q_bins[1:],x_rolling_average[dim,:], label=dim)

        return x_value_allong_q, x_rolling_average, bins

    def moving_average(self,data, window_size):
        y_padded = np.pad(data, (window_size//2, window_size-1-window_size//2), mode='edge')
        y_smooth = np.convolve(y_padded, np.ones((window_size,))/window_size, mode='valid') 
        return y_smooth

    def scatter_rc_path(self, plot_model, q_bins=np.arange(-12,12,0.1), descriptor_dims=[0,1], ax=None,fig=None):
        descriptors_total, weights_total, shot_results_total = self.RPE.create_total_trainset()
        p_B_model_RPE, q_model_RPE = self.model_output_RPE(plot_model)
        normalization,q_bins = np.histogram(q_model_RPE[:,0], bins=q_bins, weights=weights_total)
        x_value_allong_q = np.zeros((len(descriptor_dims),len(q_bins)-1))
        for i, dim in enumerate(descriptor_dims):
            pathx,q_bins = np.histogram(q_model_RPE[:,0], bins=q_bins, weights=weights_total*descriptors_total[:,dim])
            x_value_allong_q[i,:]= pathx/normalization

        if ax is None:
            fig,ax=plt.subplots(1,1)
        ax.scatter(x_value_allong_q[0,:],x_value_allong_q[1,:], color="black",s=2 )

    def rc_mean_descriptors_allong_q(self, plot_model, n_descriptors=22, q_bins=np.linspace(-12,12,40),ax=None,fig=None, plot=True):
        descriptors_total, weights_total, shot_results_total = self.RPE.create_total_trainset()
        p_B_model_RPE, q_model_RPE = self.model_output_RPE(plot_model)
        normalization,q_bins = np.histogram(q_model_RPE[:,0], bins=q_bins, weights=weights_total,density=True)
        x_value_allong_q = np.zeros((n_descriptors,len(q_bins)-1))
        x_rolling_average = np.zeros((n_descriptors,len(q_bins)-1))


        if ax is None and plot:
            fig,ax=plt.subplots(1,1)
        for dim in range(n_descriptors):
            pathx,q_bins = np.histogram(q_model_RPE[:,0], bins=q_bins, weights=weights_total*descriptors_total[:,dim])
            x_value_allong_q[dim,:]= pathx/normalization
            x_rolling_average[dim,:] = self.moving_average(x_value_allong_q[dim,:],5)
            if plot:
                ax.plot(q_bins[1:],x_rolling_average[dim,:], label=dim)
        return x_value_allong_q, x_rolling_average, q_bins
    
    def descriptor_distribution_along_q(self, plot_model, index_descriptor=0, q_bins=np.linspace(-12,12,40),ax=None,fig=None):
        descriptors_total, weights_total, shot_results_total = self.RPE.create_total_trainset()
        p_B_model_RPE, q_model_RPE = self.model_output_RPE(plot_model)

        yedges = np.linspace(np.min(descriptors_total[:,index_descriptor]),np.max(descriptors_total[:,index_descriptor]),100)
        if ax is None:
            fig,ax=plt.subplots(1,1)
        H_descriptor_allong_q, xedges, yedges = np.histogram2d(q_model_RPE[:,0],descriptors_total[:,index_descriptor],
                                        bins=(q_bins, yedges), weights = weights_total)

        im1 = ax.imshow(H_descriptor_allong_q.T, interpolation='nearest', origin='lower',cmap="Blues",
                extent=[q_bins[0], q_bins[-1], yedges[0], yedges[-1]], aspect="auto")
        if fig is not None:
            cb = fig.colorbar(im1)
            cb.set_label(r"Density")
        
    def scatter_allong_q_extreme_weights(self,plot_model, descriptor_dims = [0,1], q_bins=np.linspace(-12,12,40),ax=None,fig=None, plot=True, rolling_mean=1):
        descriptors_total, weights_total, shot_results_total = self.RPE.create_total_trainset()
        p_B_model_RPE, q_model_RPE = self.model_output_RPE(plot_model)
        ind_bin_x = np.digitize(q_model_RPE, q_bins, right=False)


        if ax is None and plot:
            fig,ax=plt.subplots(1,1)
        pathx = np.zeros((len(q_bins)-1, np.shape(descriptors_total)[1]))
        for bin in range(len(q_bins[1:])):
            ind_x_bin = np.where(ind_bin_x==bin)[0]
            pathx[bin]= np.average(descriptors_total[ind_x_bin,:], weights=weights_total[ind_x_bin],axis=0)
        if plot:
            ax.scatter(pathx[:,descriptor_dims[0]],pathx[:,descriptor_dims[1]], color="black",s=2 )
        
        return pathx, q_bins
        
    def descriptors_distribution_along_q_normalized_per_q(self, plot_model, index_descriptor=0, q_bins=np.linspace(-12,12,40),ax=None,fig=None):
        descriptors_total, weights_total, shot_results_total = self.RPE.create_total_trainset()
        p_B_model_RPE, q_model_RPE = self.model_output_RPE(plot_model)
        ind_bin_x = np.digitize(q_model_RPE, q_bins, right=False)


        if ax is None:
            fig,ax=plt.subplots(1,1)
    
        for bin in range(len(q_bins[1:])):
            ind_x_bin = np.where(ind_bin_x==bin)[0]
            pathx,bins= np.histogram(descriptors_total[ind_x_bin,index_descriptor], bins=100, weights=weights_total[ind_x_bin], density=True)
            for i in range(len(pathx)):
                ax.fill_betweenx([bins[i], bins[i + 1]], q_bins[bin] , q_bins[bin+1], 
                color=plt.cm.Blues(pathx[i]), alpha=0.7)
                ax.set_xlabel(r"$q(x)$ model")
                ax.set_ylabel(r"$x$"+"{}".format(index_descriptor))
        
    def descriptors_distribution_along_q_normalized_per_q_reactive_paths(self, plot_model, index_descriptor=0, q_bins=np.linspace(-12,12,40),ax=None,fig=None):
        descriptors_total, weights_total, shot_results_total = self.RPE.create_total_trainset()
        p_B_model_RPE, q_model_RPE = self.model_output_RPE(plot_model)
        ind_bin_x = np.digitize(q_model_RPE, q_bins, right=False)

        indices = np.where(np.all(shot_results_total == 1, axis=1))[0]
        weight_TP = np.zeros(len(descriptors_total))
        weight_TP[indices] = 1
        if ax is None:
            fig,ax=plt.subplots(1,1)
    
        for bin in range(len(q_bins[1:])):
            ind_x_bin = np.where(ind_bin_x==bin)[0]
            pathx,bins= np.histogram(descriptors_total[ind_x_bin,index_descriptor], bins=100, weights=weights_total[ind_x_bin]*weight_TP[ind_x_bin], density=True)
            for i in range(len(pathx)):
                ax.fill_betweenx([bins[i], bins[i + 1]], q_bins[bin] , q_bins[bin+1], 
                color=plt.cm.Blues(pathx[i]), alpha=0.7)
                ax.set_xlabel(r"$q(x)$ model")
                ax.set_ylabel(r"$x$"+"{}".format(index_descriptor))

    def minimum_energy_rc_descriptors_allong_q(self, plot_model, n_descriptors=22, q_bins=np.arange(-12,12,0.1),ax=None,fig=None,plot=True,rolling_mean=1):
        descriptors_total, weights_total, shot_results_total = self.RPE.create_total_trainset()
        p_B_model_RPE, q_model_RPE = self.model_output_RPE(plot_model)
        ind_bin_x = np.digitize(q_model_RPE, q_bins, right=False)

        if ax is None and plot:
            fig,ax=plt.subplots(1,1)
        x_value_allong_q = np.zeros((n_descriptors,len(q_bins)-1))
        x_rolling_average = np.zeros((n_descriptors,len(q_bins)-1))
        for dim in range(n_descriptors):
            # print(dim)
            for bin in range(len(q_bins[1:])):
                ind_x_bin = np.where(ind_bin_x==bin)[0]
                # print("descriptors:", np.shape(descriptors_total))
                # print("ind_x_bin:", np.shape(ind_x_bin))
                # print(ind_x_bin)
                # print(dim)
                pathx,bins= np.histogram(descriptors_total[ind_x_bin,dim], bins=100, weights=weights_total[ind_x_bin], density=True)
                x_value_allong_q[dim,bin]= bins[np.argmax(pathx)]
            x_rolling_average[dim,:] = self.moving_average(x_value_allong_q[dim,:],rolling_mean)
            if plot:
                ax.plot(q_bins[1:],x_rolling_average[dim,:], label=dim)
        return x_value_allong_q, x_rolling_average, q_bins
    
    def model_loss_plnp_weight_allong_q(self, plot_model, q_min=-20,q_max=20):
  
        fig, ax = plt.subplots(1,2,figsize=(14,5))
        self.plot_loss_allong_q_model(plot_model, ax=ax[0])
        self.plot_plnp_allong_q_model(plot_model, ax=ax[0])
        self.plot_weight_allong_q_model(plot_model, ax=ax[0])
        # visualizer.plot_distribution_of_points_allong_q_model(plot_model=best_model,ax=ax, density=False)
        # visualizer.plot_loss_normalized_q_allong_q_model(plot_model=best_model, ax=ax,density=True)
        ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax[0].set_yscale("log")
        ax[0].set_xlabel("q'")
        ax[0].grid()

        xedges, _, _ = self.plot_pAandpB_of_RPE_data_allong_q_model(plot_model,ax=ax[1])
        q_space = xedges
        ax[1].plot(q_space, 1/(1+np.exp(q_space)), label=r"$p_A(q')$ Theory")
        ax[1].plot(q_space, 1/(1+np.exp(-q_space)), label=r"$p_B(q')$ Theory")

        ax[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax[1].set_yscale("log")
        ax[1].set_xlabel("q'")
        ax[1].set_xlim([xedges[0],xedges[-1]])
        ax[1].grid()
        fig.tight_layout()

    def model_projections(self,plot_model):
        fig, ax = plt.subplots(1,2, figsize=(10,5)) 
        self.committor_2d_projection(plot_model,ax=ax[0])
        self.q_space_2d_projection(plot_model,fig=fig, ax=ax[1])
        fig.tight_layout()
        plt.show()

    def density_given_q_model_value_range(self,plot_model, q_value_range, descriptors=None, descriptor_dims = [0,1], ax=None,fig=None, n_bins_2d=100, cmap=None):
        if descriptors is None:
            descriptors, weights, shot_results_total = self.RPE.create_total_trainset()
        else: 
            weights = np.ones(np.shape(descriptors)[0])
            
        p_B_model_RPE, q_model_RPE = self.model_output_RPE(plot_model, descriptors_total=descriptors)
        # q_bins = np.linspace(q_value_range[0],q_value_range[1],2)
        # ind_bin_x = np.digitize(q_model_RPE, q_bins, right=False)
        # print(q_bins)
        # normalization,q_bins = np.histogram(q_model_RPE[:,0], bins=q_bins, weights=weights_total)
        if cmap is None:
            cmap = matplotlib.cm.get_cmap('Spectral')
        # print(np.shape(ind_bin_x))
        # print(ind_bin_x)
        points_in_bin = (q_model_RPE >= q_value_range[0]) & (q_model_RPE < q_value_range[1])
        points_in_bin = points_in_bin[:,0]
        # print(points_in_bin)
        # print(np.shape(q_model_RPE))
        # print(np.shape(descriptors))
        # print(np.shape(points_in_bin))
        # print(np.shape(dim_descriptors))
        xedges, yedges = self.create_x_y_edges(n_bins_2d=n_bins_2d)

        if ax is None:
            fig,ax=plt.subplots(1,1)
        H_given_q, xedges, yedges = np.histogram2d(descriptors[points_in_bin,descriptor_dims[0]],
                                        descriptors[points_in_bin,descriptor_dims[1]],
                                        bins=(xedges, yedges), weights = weights[points_in_bin], density=True)
        H_given_q = np.where(H_given_q==0, np.nan, H_given_q)

        im1 = ax.imshow(H_given_q.T, interpolation='nearest', origin='lower',cmap=cmap,
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], alpha=0.7, aspect="auto")

    def plot_gradient_field_2d(self, model, grid_size=20, ax=None):
        """
        Plots the gradient vectors of a PyTorch model with respect to two chosen dimensions
        (x and y), while keeping all other dimensions fixed at zero.
        
        Parameters:
        model : torch.nn.Module
            The PyTorch model to analyze.
        x_range : tuple (float, float)
            The range of x values (min, max) for the grid.
        y_range : tuple (float, float)
            The range of y values (min, max) for the grid.
        grid_size : int, optional
            Number of points along each axis for the grid (default is 20).
        fixed_dim : int, optional
            The number of input dimensions (default is 22).
        """
        # Create a grid of x and y values
        xedges, yedges = self.create_x_y_edges(n_bins_2d=grid_size)

        X,Y = np.meshgrid(xedges,yedges)
        
        # Initialize arrays to store gradients for each point in the grid
        dfdx = np.zeros(grid_size**2)
        dfdy = np.zeros(grid_size**2)

        # Set requires_grad=True for input tensor


        oscis1 = [0. for _ in range(1 - 1)]
        oscis2 = [0. for _ in range(22- 1 - 1)]
        coord = np.array([[xv] + oscis1 + [yv] + oscis2 for yv in xedges for xv in yedges], dtype=np.float32)
        # Forward pass: compute the model's prediction
        for i in range(len(coord)):
            input_tensor = torch.tensor(coord[i],requires_grad=True,device=model._device)
            output = model.nnet(input_tensor)
            # print(output)
            # Backward pass: compute the gradient of the output with respect to input
            output.backward()
            
            # Store the gradients with respect to x and y (0th and 1st dimensions)
            dfdx[i] = input_tensor.grad[0].item()
            dfdy[i] = input_tensor.grad[1].item()

            
            # Zero the gradients for the next iteration
            model.nnet.zero_grad()
        dfdx = dfdx.reshape(np.shape(X))       
        dfdy = dfdy.reshape(np.shape(X))

        # Plot the vector field using a quiver plot
        if ax is None:
            fig,ax = plt.subplots(1,1)

        ax.quiver(X, Y, dfdx, dfdy, color='b')
        ax.set_title('Gradient Vectors of Model Output in 2D (x, y)')

    # def plot_grad_average(self, model, epsilon=1e-3, ax=None, bins=10,scale=1):
    #     # Create the trainset (descriptors, weights, and shot_results_total from RPE)
    #     descriptors, weights, shot_results_total = self.RPE.create_total_trainset()
    #     # Initialize arrays to store the gradients
    #     dfdx = np.zeros(np.shape(descriptors))
    #     norm = 1/np.max(descriptors,axis=0)


    #     # Iterate over each descriptor to compute gradients
    #     # Create input tensor from the descriptor, setting requires_grad=True
    #     input_tensor = torch.tensor(descriptors, requires_grad=True, device=model._device, dtype=torch.float32)
    #     model.nnet.eval()
    #     # Forward pass: compute the output
    #     output = model.nnet(input_tensor)
        
    #     # Backward pass: compute the gradient of the output with respect to input
    #     output.backward(torch.ones_like(output))

    #     # Store the gradients with respect to the 0th and 1st dimensions
    #     gradients = input_tensor.grad.cpu().numpy()/norm

    #     # Zero the gradients for the next iteration
    #     model.nnet.zero_grad()
    #     model.nnet.train()
    #     # Stack the gradients into a single array for easy manipulation
    #     # gradients = np.stack((dfdx, dfdy), axis=1)
        
    #     # Compute the magnitude of the gradient
    #     gradient_magnitude = np.linalg.norm(gradients, axis=1)

    #     # Check if gradients are below the threshold epsilon and count them
    #     below_threshold = np.sum(gradient_magnitude < epsilon)
    #     print(f"Number of points with gradient magnitude below {epsilon}: {below_threshold}")

    #     # Now we compute the 2D histogram, weighted by the given weights
    #     xedges, yedges = self.create_x_y_edges(n_bins_2d=bins)
    #     # H_unweighted,_, _ = self.unweighted_PE(n_bins_2d=n_bins_2d,descriptor_dims=descriptor_dims)
    #     H_weighted, _,_ = self.weighted_RPE(n_bins_2d=bins,descriptor_dims=self.descriptor_dims)
    #     H_dfdx, xedges, yedges = np.histogram2d(descriptors[:,self.descriptor_dims[0]],
    #                                             descriptors[:,self.descriptor_dims[1]],
    #                                             bins=(xedges, yedges),
    #                                             weights = gradients[:,self.descriptor_dims[0]]*weights)
    #     H_dfdy, xedges, yedges = np.histogram2d(descriptors[:,self.descriptor_dims[0]],
    #                                     descriptors[:,self.descriptor_dims[1]],
    #                                     bins=(xedges, yedges),
    #                                     weights = gradients[:,self.descriptor_dims[1]]*weights)
    #     average_dfdx = H_dfdx.T/H_weighted.T
    #     average_dfdy = H_dfdy.T/H_weighted.T

    #     # Plotting the average gradient on a 2D histogram if an axis is provided
    #     if ax is None:
    #         fig, ax = plt.subplots()

    #     # Create the 2D histogram plot
    #     x_center = (xedges[:-1] + xedges[1:]) / 2
    #     y_center = (yedges[:-1] + yedges[1:]) / 2
    #     X, Y = np.meshgrid(x_center, y_center)
        
    #     ax.quiver(X, Y, average_dfdx,average_dfdy, color='b', scale_units="x",scale=scale)

    #     ax.set_title(f"Weighted Average Gradient on projected on xy-plane")

    #     return below_threshold
    
    def plot_grad_average(self, model, epsilon=1e-3, ax=None, bins=10, descriptor_dims=[0,1]):
        # Create the trainset (descriptors, weights, and shot_results_total from RPE)
        descriptors, weights, shot_results_total = self.RPE.create_total_trainset()
        # Initialize arrays to store the gradients
        dfdx = np.zeros(np.shape(descriptors))


        # Iterate over each descriptor to compute gradients
        # Create input tensor from the descriptor, setting requires_grad=True
        input_tensor = torch.tensor(descriptors, requires_grad=True, device=model._device, dtype=torch.float32)
        model.nnet.eval()
        # Forward pass: compute the output
        output = model.nnet(input_tensor)
        
        # Backward pass: compute the gradient of the output with respect to input
        output.backward(torch.ones_like(output))

        # Store the gradients with respect to the 0th and 1st dimensions
        gradients = input_tensor.grad.cpu().numpy()

        # Zero the gradients for the next iteration
        model.nnet.zero_grad()
        model.nnet.train()
        # Stack the gradients into a single array for easy manipulation
        # gradients = np.stack((dfdx, dfdy), axis=1)
        
        # Compute the magnitude of the gradient
        gradient_magnitude = np.linalg.norm(gradients, axis=1)

        # Check if gradients are below the threshold epsilon and count them
        below_threshold = np.sum(gradient_magnitude < epsilon)
        print(f"Number of points with gradient magnitude below {epsilon}: {below_threshold}")

        # Now we compute the 2D histogram, weighted by the given weights
        xedges = np.linspace(self.dims_extent[0], self.dims_extent[1], bins)
        yedges = np.linspace(self.dims_extent[2], self.dims_extent[3], bins)
        # H_unweighted,_, _ = self.unweighted_PE(n_bins_2d=n_bins_2d,descriptor_dims=descriptor_dims)
        H_weighted, _,_ = self.weighted_RPE(n_bins_2d=bins,descriptor_dims=descriptor_dims)
        H_dfdx, xedges, yedges = np.histogram2d(descriptors[:,descriptor_dims[0]],
                                                descriptors[:,descriptor_dims[1]],
                                                bins=(xedges, yedges),
                                                weights = gradients[:,0]*weights)
        H_dfdy, xedges, yedges = np.histogram2d(descriptors[:,descriptor_dims[0]],
                                        descriptors[:,descriptor_dims[1]],
                                        bins=(xedges, yedges),
                                        weights = gradients[:,1]*weights)
        average_dfdx = H_dfdx.T/H_weighted.T
        average_dfdy = H_dfdy.T/H_weighted.T

        # Plotting the average gradient on a 2D histogram if an axis is provided
        if ax is None:
            fig, ax = plt.subplots()

        # Create the 2D histogram plot
        x_center = (xedges[:-1] + xedges[1:]) / 2
        y_center = (yedges[:-1] + yedges[1:]) / 2
        X, Y = np.meshgrid(x_center, y_center)
        
        ax.quiver(X, Y, average_dfdx,average_dfdy, color='b')

        ax.set_title(f"Weighted Average Gradient on projected on xy-plane")

        return below_threshold


    def plot_logcurrent_pb(self, model, epsilon=1e-3, ax=None, bins=10,log=False,scale=1):
        # Create the trainset (descriptors, weights, and shot_results_total from RPE)
        descriptors, weights, shot_results_total = self.RPE.create_total_trainset()
        # Initialize arrays to store the gradients
        dfdx = np.zeros(np.shape(descriptors))


        # Iterate over each descriptor to compute gradients
        # Create input tensor from the descriptor, setting requires_grad=True
        input_tensor = torch.tensor(descriptors, requires_grad=True, device=model._device, dtype=torch.float32)
        model.nnet.eval()
        # Forward pass: compute the output
        output = model.nnet(input_tensor)
        norm = 1/np.max(descriptors,axis=0)
        
        # Backward pass: compute the gradient of the output with respect to input
        output.backward(torch.ones_like(output))

        # Store the gradients with respect to the 0th and 1st dimensions
        gradients = input_tensor.grad.cpu().numpy()/norm
        pb = 1/(1+np.exp(-output.detach().cpu().numpy()))
        pa = 1/(1+np.exp(output.detach().cpu().numpy()))

        current= pa*pb*gradients
        # Zero the gradients for the next iteration
        model.nnet.zero_grad()
        model.nnet.train()
        # Stack the gradients into a single array for easy manipulation
        # gradients = np.stack((dfdx, dfdy), axis=1)
        # Compute the magnitude of the gradient
        gradient_magnitude = np.linalg.norm(gradients, axis=1)

        # Check if gradients are below the threshold epsilon and count them
        below_threshold = np.sum(gradient_magnitude < epsilon)
        print(f"Number of points with gradient magnitude below {epsilon}: {below_threshold}")

        # Now we compute the 2D histogram, weighted by the given weights
        weights = weights #normalize the weights
        xedges, yedges = self.create_x_y_edges(n_bins_2d=bins)
        # H_unweighted,_, _ = self.unweighted_PE(n_bins_2d=n_bins_2d,descriptor_dims=descriptor_dims)
        H_weighted, _,_ = self.weighted_RPE(n_bins_2d=bins,descriptor_dims=self.descriptor_dims)
        H_dfdx, xedges, yedges = np.histogram2d(descriptors[:,self.descriptor_dims[0]],
                                                descriptors[:,self.descriptor_dims[1]],
                                                bins=(xedges, yedges),
                                                weights = current[:,self.descriptor_dims[0]]*weights)
        H_dfdy, xedges, yedges = np.histogram2d(descriptors[:,self.descriptor_dims[0]],
                                        descriptors[:,self.descriptor_dims[1]],
                                        bins=(xedges, yedges),
                                        weights = current[:,self.descriptor_dims[1]]*weights)
        
        # average_dfdx = H_dfdx.T/H_weighted.T
        # average_dfdy = H_dfdy.T/H_weighted.T
        if log:
            marginalized_magnitude = np.sqrt(H_dfdx.T**2 + H_dfdy.T**2)
            average_dfdx = H_dfdx.T/marginalized_magnitude*np.log(marginalized_magnitude)
            average_dfdy = H_dfdy.T/marginalized_magnitude*np.log(marginalized_magnitude)
        else:
            average_dfdx = H_dfdx.T
            average_dfdy = H_dfdy.T


        # Plotting the average gradient on a 2D histogram if an axis is provided
        if ax is None:
            fig, ax = plt.subplots()

        # Create the 2D histogram plot
        x_center = (xedges[:-1] + xedges[1:]) / 2
        y_center = (yedges[:-1] + yedges[1:]) / 2
        X, Y = np.meshgrid(x_center, y_center)
        
        ax.quiver(X, Y, average_dfdx,average_dfdy, color='b',scale_units='x', scale=scale)

        ax.set_title(f"Weighted Average Gradient on projected on xy-plane")

        return below_threshold

    def create_path_from_gradient(self, model, x_A, q_max = 14, alpha=0.01, max_steps=1000, grad_threshold=1e-5):
        """
        Generates a path from point A by following the gradient of q(x), using gradient calculations 
        similar to plot_grad_average.
        
        Parameters:
            model (torch.nn.Module): The neural network model q(x) taking x as input.
            x_A (np.array or torch.Tensor): Starting point in region A (requires gradient).
            alpha (float): Step size along the gradient direction.
            max_steps (int): Maximum number of steps to take along the gradient.
            grad_threshold (float): Threshold for the gradient magnitude to stop the path.
            
        Returns:
            path (list): List of tensors representing the path from A to B along the gradient.
        """
        # Convert x_A to a tensor with gradient tracking, compatible with model's device
        x = torch.tensor(x_A, requires_grad=True, device=model._device, dtype=torch.float32)
        path = [x.clone().detach().cpu().numpy()]  # Store path points
        q_path = []
        for i in range(max_steps):
            # Forward pass: compute output
            model.nnet.eval()
            output = model.nnet(x)
            q_path.append(output.clone().detach().cpu().numpy())
            if output>q_max or i==max_steps-1:
                break
            
            # Backward pass: compute gradient of output with respect to input x
            output.backward(torch.ones_like(output))
            
            # Get gradient and compute its magnitude
            grad = x.grad
            grad_magnitude = grad.norm().item()
            
            # Stop if gradient magnitude is below threshold
            if grad_magnitude < grad_threshold:
                break
            
            # Update x by moving along the gradient (gradient ascent)
            with torch.no_grad():
                x += alpha * grad  # Gradient ascent; use -alpha * grad for descent
                path.append(x.clone().detach().cpu().numpy())  # Store the current position
            
            # Clear gradients for the next iteration
            x.grad.zero_()
            model.nnet.train
        
        # Compute differences in q_path to identify decreases
        q_diff = np.diff(q_path)
        decreasing_points = np.sum(q_diff < 0)  
        if decreasing_points>0:
            print("Ow no!! Number of points where q(x) decreases along the gradient path is:", decreasing_points)
            print("The model is not monotonically increasing!")
        return np.array(q_path), np.array(path), decreasing_points

    # def hipr_allong_q(self, plot_model, index_descriptors=[0, 1], q_bins=np.linspace(-12, 12, 25), ax=None, fig=None, hipr_plus=False):
    #     descriptors_total, weights_total, shot_results_total = self.RPE.create_total_trainset()
    #     p_B_model_RPE, q_model_RPE = self.model_output_RPE(plot_model)
    #     ind_bin_x = np.digitize(q_model_RPE, q_bins, right=False)

    #     if ax is None:
    #         fig, ax = plt.subplots(1, 1)

    #     hipr_per_q_bin = np.zeros((len(q_bins) - 1, np.shape(descriptors_total)[1]))
    #     hipr_per_q_bin_std = np.zeros((len(q_bins) - 1, np.shape(descriptors_total)[1]))

    #     n_states = 2

    #     for bin in range(len(q_bins[1:])):
    #         ind_x_bin = np.where(ind_bin_x == bin)[0]
    #         trainset_q = aimmd.TrainSet(n_states=n_states,
    #                                     descriptors=descriptors_total[ind_x_bin],
    #                                     shot_results=shot_results_total[ind_x_bin],
    #                                     weights=weights_total[ind_x_bin])
    #         hipr_q = aimmd.analysis.HIPRanalysis(plot_model, trainset_q)
    #         if hipr_plus:
    #             hipr_plus_losses, hipr_plus_std = hipr_q.do_hipr_plus()
    #         else:
    #             hipr_plus_losses, hipr_plus_std = hipr_q.do_hipr()
    #         hipr_per_q_bin[bin, :] = hipr_plus_losses[:-1] - hipr_plus_losses[-1]
    #         hipr_per_q_bin_std[bin, :] = hipr_plus_std[:-1]

    #     bin_centers = (q_bins[:-1] + q_bins[1:]) / 2

    #     for dim in index_descriptors:
    #         ax.errorbar(bin_centers, hipr_per_q_bin[:, dim], yerr=hipr_per_q_bin_std[:, dim], fmt="o", capsize=5, label="dim {}".format(dim))

    #     # Adding titles and labels
    #     ax.set_title('HIPR Analysis Across q Bins')
    #     ax.set_xlabel('q Bins')
    #     ax.set_ylabel('HIPR Values')
    #     ax.grid(alpha=0.75)
    #     ax.legend()

    #     # Show the plot if no axis is provided
    #     if ax is None:
    #         plt.show()
        
    #     return bin_centers, hipr_per_q_bin, hipr_per_q_bin_std