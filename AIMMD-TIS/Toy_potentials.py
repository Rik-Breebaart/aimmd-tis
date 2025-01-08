'''
This file will contain the toy potentials that will be tested and have been investigated in the "ideas potentials" path.

The toy potentials will be stored as toy.PES subclasses based on the openpathsampling.engines.toy 

Additionally there should be a subclass which generates a higher dimensional toy potential from the desired potential by adding harmonic oscilattors.
From which one can obtain:
A topology
info on the stable states 
A artificial initial path from the stables states to each other. (both forward and backward)
'''

# Import necessary libraries and modules
import numpy as np 
import matplotlib.pyplot as plt
import openpathsampling as paths
from functools import reduce
import sys
import openpathsampling.engines.toy as toys

# Define a class for a higher-dimensional harmonic potential
class higher_d_harmonic(toys.PES):
    def __init__(self, n_dims_pot=2, n_harmonics=20, seed=123):
        super().__init__(n_harmonics)
        self.n_harmonics = n_harmonics
        self.n_dims_pot = n_dims_pot
        self.seed = seed

    # Create a potential energy surface (PES) by adding harmonic oscillators
    def create_pes(self, pes_list):
        # fix seed to get the same pes everytime
        np.random.seed(self.seed)
        print(np.random.get_state()[1][0])
        harmonic_omegas = [1.0, 1.5] + [10*np.random.uniform(0.2,1.0) for _ in range(self.n_harmonics-2)]
        if self.n_harmonics==0:
            omega=[0. for _ in range(self.n_dims_pot)]
        else:
            omega = [0. for _ in range(self.n_dims_pot)] + harmonic_omegas[:self.n_harmonics]

        print("pot dimensions without oscillators: {}".format(self.n_dims_pot))
        pes_list += [toys.HarmonicOscillator(A=[0. for _ in range(self.n_dims_pot)] + [1./2. for _ in range(self.n_harmonics)],
                                            omega = omega,
                                            x0=[0. for _ in range(self.n_harmonics + self.n_dims_pot)])
            ]
        print('harmonic oscillators omegas:')
        print(repr(pes_list[-1].omega))
        self.extent = [-1, 1, -1, 1]
        self.pes = reduce(lambda x,y: x+y, pes_list)
        #reseed to remove influence of seed on other aspects of the code
        np.random.seed(None)

    # Create a topology for the system
    def create_topology(self):
        self.topology = toys.Topology(n_spatial=self.n_dims_pot + self.n_harmonics,
                            masses=np.array([1.0 for _ in range(self.n_dims_pot + self.n_harmonics)]),
                            pes=self.pes,
                            n_atoms=1
                            )

    # Return the created topology
    def return_topology(self):
        return self.topology

    # Generate a linear path between two coordinates
    def linear_path(self, start_coord, end_coord, steps, toy_eng):
        step_size = np.array(end_coord) - np.array(start_coord)
        return [toys.Snapshot(coordinates=np.array([[start_coord[j] + step_size[j]*i/steps for j in range(self.n_dims_pot)] + [0. for _ in range(self.n_harmonics)]]), 
                              velocities=np.array([[0.5 for _ in range(self.n_dims_pot)]  + [0. for _ in range(self.n_harmonics)]]),
                              engine=toy_eng
                             )
                for i in range(steps)]

    # Generate a simple initial path using linear paths
    def simple_initial_path(self, steps, toy_eng):
        return self.linear_path(self.state_A, self.state_B, steps, toy_eng)

    # Define a template snapshot
    def template(self, toy_eng): 
        return toys.Snapshot(coordinates=np.array([[0.5 for _ in range(self.n_dims_pot)] + [0. for _ in range(self.n_harmonics)]]), 
                             velocities=np.array([[0. for _ in range(self.n_harmonics+self.n_dims_pot)]]),
                             engine=toy_eng
                            )

    # Define a 2D potential energy surface
    def pes_2d_pot(self, x, y, dim_y=1):
        self.positions = [x]
        self.positions += [0] * (dim_y-1)
        self.positions += [y]
        self.positions += [0] * (self.n_dims_pot+ self.n_harmonics-dim_y-1)
        self.mass = 1
        return self.pes.V(self) 
    
    def pes_2d_F_x(self, x, y, dim_y=1):
        self.positions = [x]
        self.positions += [0] * (dim_y-1)
        self.positions += [y]
        self.positions += [0] * (self.n_dims_pot+ self.n_harmonics-dim_y-1)
        self.mass = 1
        return -self.pes.dVdx(self)[0]
    
    def pes_2d_F_y(self, x, y, dim_y=1):
        self.positions = [x]
        self.positions += [0] * (dim_y-1)
        self.positions += [y]
        self.positions += [0] * (self.n_dims_pot+ self.n_harmonics-dim_y-1)
        self.mass = 1
        return -self.pes.dVdx(self)[1]

    # Define a 1D potential energy surface
    def pes_1d_pot(self, x, y_slice=0):
        self.positions = [x]
        if self.n_dims_pot==2:
            self.positions += [y_slice]
        self.positions += [0] * self.n_harmonics
        self.mass = 1
        return self.pes.V(self) 


    # Plot the potential energy surface
    def plot_2d_pes(self, range_x, range_y, dim_y=1):
        X, Y = np.meshgrid(range_x, range_y)
        pes_points = np.vectorize(self.pes_2d_pot)(X, Y, dim_y=dim_y)
        return X,Y,pes_points
    
    # Plot the potential energy surface in along 1 coordinate
    def plot_1d_pes(self, range_x, y_slice=0):
        pes_points = np.vectorize(self.pes_1d_pot)(range_x, y_slice)
        return range_x,pes_points

    def get_2d_pes_F(self, range_x, range_y, dim_y=1):
        X, Y = np.meshgrid(range_x, range_y)
        pes_F_x, pes_F_y = np.vectorize(self.pes_2d_F_x)(X, Y, dim_y=dim_y), np.vectorize(self.pes_2d_F_y)(X, Y, dim_y=dim_y)
        return X, Y, pes_F_x, pes_F_y

class potential_0(higher_d_harmonic):
    def __init__(self, A, x0, n_harmonics=20, seed=123):
         # Initialize potential parameters and create the PES
        self.n_harmonics = n_harmonics
        self.seed = seed
        self.n_dims_pot = 1
        pes_list = [DoubleWell(A,x0)]
        self.create_pes(pes_list)
        self.create_topology()
        self.state_A = [-x0]
        self.state_B = [x0]
        self.state_boundary = 0.1
        self.extent = [-1.5*x0,1.5*x0, -2.5 ,2.5]
        self.levels = np.arange(0,A*x0**4, A*x0**4/10)

    
    def __repr__(self):
        return f"One_d_potential"
        
    def to_dict(self):
        dct = super().to_dict()
        return dct
    
    def V(self, sys):

        return self.pes.V(sys)
    
    def dVdx(self, sys):
        """
        -F = [dV(x, y) / dx, dV(x,y) / dy]
        """
        return self.pes.dVdx(sys)

    def stable_interface_function(self, snapshot, center):
        import math
        return math.sqrt((snapshot.xyz[0][0]-center[0])**2)

# Define a subclass potential_1 that inherits from higher_d_harmonic
class potential_1(higher_d_harmonic):
    def __init__(self, n_harmonics=20, seed=123):
         # Initialize potential parameters and create the PES
        self.n_harmonics = n_harmonics
        self.n_dims_pot = 2
        self.seed = seed
        pes_list = []
        pes_list += [toys.OuterWalls(sigma=[3.0, 3.0]+ [0. for _ in range(n_harmonics)],
                                    x0= [0.0, 0.0]+ [0. for _ in range(n_harmonics)])
                    ]

        pes_list += [toys.Gaussian(A = -1.0, 
                                    alpha=[12.0, 12.0]+ [0. for _ in range(n_harmonics)],
                                    x0= [-0.5, 0.5]+ [0. for _ in range(n_harmonics)])
                    ]

        pes_list += [toys.Gaussian(A = -1.0, 
                                    alpha=[12.0, 12.0]+ [0. for _ in range(n_harmonics)],
                                    x0= [0.5, -0.5]+ [0. for _ in range(n_harmonics)])
                    ]

        pes_list += [toys.Gaussian(A = 2.0, 
                                    alpha=[15.0, 5.0]+ [0. for _ in range(n_harmonics)],
                                    x0= [0.0, 0.0]+ [0. for _ in range(n_harmonics)])
                    ]

        pes_list += [toys.Gaussian(A = 0.5, 
                                    alpha=[15.0, 5.0]+ [0. for _ in range(n_harmonics)],
                                    x0= [0.8, 0.8]+ [0. for _ in range(n_harmonics)])
                    ]

        pes_list += [toys.Gaussian(A = 0.5, 
                                    alpha=[15.0, 5.0]+ [0. for _ in range(n_harmonics)],
                                    x0= [-0.8, -0.8]+ [0. for _ in range(n_harmonics)])
                    ]
        self.create_pes(pes_list)
        self.create_topology()
        self.state_A = [-0.5, 0.5]
        self.state_B = [0.5, -0.5]
        self.state_boundary = 0.075
        self.levels = np.arange(-1,2.5, 0.2)

    def __repr__(self):
        return f"potential_1"
        
    def to_dict(self):
        dct = super().to_dict()
        return dct

    def V(self, sys):
        return self.pes.V(sys)
    
    def dVdx(self, sys):
        """
        -F = [dV(x, y) / dx, dV(x,y) / dy]
        """
        return self.pes.dVdx(sys)
    
    def stable_interface_function(self, snapshot, center):
        import math
        return math.sqrt((snapshot.xyz[0][0]-center[0])**2 + (snapshot.xyz[0][1]-center[1])**2)
        
# Define a subclass potential_2 that inherits from higher_d_harmonic
class potential_2(higher_d_harmonic):
    def __init__(self, n_harmonics=20, a=1, b=0,seed=123):
         # Initialize potential parameters and create the PES
        self.n_harmonics = n_harmonics
        self.n_dims_pot = 2
        self.seed = seed
        pes_list = []


        pes_list += [toys.OuterWalls(sigma= [1/1800*0.0625, 1/1800]+ [0. for _ in range(n_harmonics)],
                                    x0= [0.0, 0.0]+ [0. for _ in range(n_harmonics)])
                    ]

        pes_list += [toys.Gaussian(A = -5.0, 
                                    alpha=[0.5, 1]+ [0. for _ in range(n_harmonics)],
                                    x0= [-4, 0]+ [0. for _ in range(n_harmonics)])
                    ]

        pes_list += [toys.Gaussian(A = -5.0, 
                                    alpha=[0.5, 1]+ [0. for _ in range(n_harmonics)],
                                    x0= [4, 0]+ [0. for _ in range(n_harmonics)])
                    ]

        pes_list += [toys.Gaussian(A = 5.0, 
                                    alpha=[0.25, 4.0] + [0. for _ in range(n_harmonics)],
                                    x0= [0.0, 0.0]+ [0. for _ in range(n_harmonics)])
                    ]

        pes_list += [toys.Gaussian(A = 2.0, 
                                    alpha=[4, 0.5] + [0. for _ in range(n_harmonics)],
                                    x0= [-b, -1] + [0. for _ in range(n_harmonics)])
                    ]

        pes_list += [toys.Gaussian(A = 2.0*a, 
                                    alpha=[4, 0.5]+ [0. for _ in range(n_harmonics)],
                                    x0= [b, 1]+ [0. for _ in range(n_harmonics)])
                    ]
        self.create_pes(pes_list)
        self.create_topology()
        self.state_A = [-4.0, 0]
        self.state_B = [4.0, 0]
        self.state_boundary = 1 
        self.extent = [-8, 8, -5, 5]
        self.levels = np.arange(-10,10, 1.0)

    def __repr__(self):
        return f"potential_2"
        
    def to_dict(self):
        dct = super().to_dict()
        return dct

    def V(self, sys):
        return self.pes.V(sys)
    
    def dVdx(self, sys):
        """
        -F = [dV(x, y) / dx, dV(x,y) / dy]
        """
        return self.pes.dVdx(sys)

    def stable_interface_function(self, snapshot, center):
        import math
        return math.sqrt((snapshot.xyz[0][0]-center[0])**2 + (snapshot.xyz[0][1]-center[1])**2)
    
# Define a subclass potential_3 that inherits from higher_d_harmonic
class potential_3(higher_d_harmonic):
    def __init__(self, n_harmonics=20,a=1,b=0,seed=123):
         # Initialize potential parameters and create the PES
        self.n_harmonics = n_harmonics
        self.n_dims_pot = 2
        self.seed = seed
        pes_list = []

        pes_list += [toys.OuterWalls(sigma= [2.5, 2.5]+ [0. for _ in range(n_harmonics)],
                                    x0= [0.0, 0.0]+ [0. for _ in range(n_harmonics)])
                    ]

        pes_list += [toys.Gaussian(A = -1.0, 
                                    alpha=[50, 500]+ [0. for _ in range(n_harmonics)],
                                    x0= [-0.9, 0]+ [0. for _ in range(n_harmonics)])
                    ]

        pes_list += [toys.Gaussian(A = -1.0, 
                                    alpha=[50, 500]+ [0. for _ in range(n_harmonics)],
                                    x0= [0.9, 0]+ [0. for _ in range(n_harmonics)])
                    ]

        pes_list += [toys.Gaussian(A = 2.0, 
                                    alpha=[10, 100] + [0. for _ in range(n_harmonics)],
                                    x0= [0.0, 0.0]+ [0. for _ in range(n_harmonics)])
                    ]

        pes_list += [toys.Gaussian(A = 0.5, 
                                    alpha=[50, 10] + [0. for _ in range(n_harmonics)],
                                    x0= [-b, -1/8] + [0. for _ in range(n_harmonics)])
                    ]

        pes_list += [toys.Gaussian(A = 0.5*a, 
                                    alpha=[50, 10]+ [0. for _ in range(n_harmonics)],
                                    x0= [b, 1/8]+ [0. for _ in range(n_harmonics)])
                    ]

        pes_list += [toys.HarmonicOscillator(A=[0., 0.] + [1./2. for _ in range(n_harmonics)],
                                            omega=[0., 0.] + [0.2, 0.5] + [10.*np.random.ranf() for _ in range(n_harmonics-2)],
                                            x0=[0. for _ in range(n_harmonics + 2)])
                    ]
        self.create_pes(pes_list)
        self.create_topology()
        self.state_A = [-0.7, 0.0]
        self.state_B = [0.7, 0.0]
        self.state_boundary = 0.1
        self.levels = np.arange(-1,2, 0.25)

    def __repr__(self):
        return f"potential_3"
        
    def to_dict(self):
        dct = super().to_dict()
        return dct

    def V(self, sys):
        return self.pes.V(sys)
    
    def dVdx(self, sys):
        """
        -F = [dV(x, y) / dx, dV(x,y) / dy]
        """
        return self.pes.dVdx(sys)

    def stable_interface_function(self, snapshot, center):
        import math
        return math.sqrt((snapshot.xyz[0][0]-center[0])**2 + (snapshot.xyz[0][1]-center[1])**2)

# Define a subclass potential_4 that inherits from higher_d_harmonic
class potential_4_linear(higher_d_harmonic):
    def __init__(self, n_harmonics=20, seed=123):
         # Initialize potential parameters and create the PES
        self.n_harmonics = n_harmonics
        self.n_dims_pot = 2
        self.seed = seed
        pes_list = []

        pes_list += [toys.OuterWalls(sigma= [1, 1]+ [0. for _ in range(n_harmonics)],
                                x0= [0.0, 0.0]+ [0. for _ in range(n_harmonics)])
                ]

        pes_list += [toys.Gaussian(A = -2.0, 
                            alpha=[12, 12]+ [0. for _ in range(n_harmonics)],
                                x0= [-0.75, -0.5]+ [0. for _ in range(n_harmonics)])
                ]

        pes_list += [toys.Gaussian(A = -2.0, 
                            alpha=[12, 12]+ [0. for _ in range(n_harmonics)],
                                x0= [0.75, 0.5]+ [0. for _ in range(n_harmonics)])
                ]
        self.create_pes(pes_list)
        self.create_topology()
        self.state_A = [-0.75, -0.5]
        self.state_B = [0.75, 0.5]
        self.state_boundary = 0.075
        self.levels = np.arange(-2,2, 0.5)

    def __repr__(self):
        return f"potential_4_Linear."
        
    def to_dict(self):
        dct = super().to_dict()
        return dct

    def V(self, sys):
        return self.pes.V(sys)
    
    def dVdx(self, sys):
        """
        -F = [dV(x, y) / dx, dV(x,y) / dy]
        """
        return self.pes.dVdx(sys)
    
    def stable_interface_function(self, snapshot, center):
        import math
        return math.sqrt((snapshot.xyz[0][0]-center[0])**2 + (snapshot.xyz[0][1]-center[1])**2)
    
# Define a subclass potential_5_Z_pot that inherits from higher_d_harmonic
class potential_5_Z_pot(higher_d_harmonic):
    def __init__(self, n_harmonics = 20, seed=123):
         # Initialize potential parameters and create the PES
        self.n_dims_pot=2
        self.n_harmonics = n_harmonics
        self.seed = seed
        pes_list = [Z_Pot()]
        self.create_pes(pes_list)
        self.create_topology()
        self.state_A = [-8,-5]
        self.state_B = [8, 5]
        self.state_boundary = 2.0
        self.extent = [-15, 15, -10, 10]
        self.levels = np.arange(-2,8, 0.5)


    
    def __repr__(self):
        return f"Z_potential."
        
    def to_dict(self):
        dct = super().to_dict()
        return dct
    
    def V(self, sys):

        return self.pes.V(sys)
    
    def dVdx(self, sys):
        """
        -F = [dV(x, y) / dx, dV(x,y) / dy]
        """
        return self.pes.dVdx(sys)

    def stable_interface_function(self, snapshot, center):
        import math
        return math.sqrt((snapshot.xyz[0][0]-center[0])**2 + (snapshot.xyz[0][1]-center[1])**2)

    def simple_initial_path(self, steps, toy_eng):
        end_1 = [12,-5]
        end_2 = [-12,5]
        part_1 = self.linear_path(self.state_A, end_1, int(1/12*steps), toy_eng)
        part_2 = self.linear_path(end_1, end_2, int(5/6*steps), toy_eng)
        part_3 = self.linear_path(end_2, self.state_B, int(1/12*steps), toy_eng)
        path = []
        path += part_1
        path += part_2
        path += part_3
        return path
    

class potential_linear_q(higher_d_harmonic):
    def __init__(self, n_harmonics=20, seed=123):
         # Initialize potential parameters and create the PES
        self.n_harmonics = n_harmonics
        self.seed = seed
        self.n_dims_pot = 1
        pes_list = [Theoretical_linear_q()]
        self.create_pes(pes_list)
        self.create_topology()
        self.state_A = [-10]
        self.state_B = [10]
        self.state_boundary = 0.1
        self.extent = [-1.5*10,1.5*10, -10 ,10]
        self.levels = np.arange(-20, 2, 20/10)
    
    def __repr__(self):
        return f"One_d_potential_linear_q"
        
    def to_dict(self):
        dct = super().to_dict()
        return dct
    
    def V(self, sys):

        return self.pes.V(sys)
    
    def dVdx(self, sys):
        """
        -F = [dV(x, y) / dx, dV(x,y) / dy]
        """
        return self.pes.dVdx(sys)

    def stable_interface_function(self, snapshot, center):
        import math
        return math.sqrt((snapshot.xyz[0][0]-center[0])**2)
    
class potential_WQ(higher_d_harmonic):
    def __init__(self, n_harmonics=20, seed=123,rotation_degrees=45, d=1, scale=2):
        rotation = rotation_degrees/180*np.pi
        self.n_harmonics = n_harmonics
        self.seed = seed
        self.n_dims_pot = 2
        pes_list = [WolfeQuapp(rotation=rotation, d=d, scale=scale)]
        self.create_pes(pes_list)
        self.create_topology()
        self.cos_rot = np.cos(-rotation)/d
        self.sin_rot = np.sin(-rotation)/d
        self.rot = np.array([[self.cos_rot,self.sin_rot],[-self.sin_rot,self.cos_rot]])
        state_A_tilde = np.array([-1.15, 1.5])
        state_B_tilde = np.array([1.15, -1.5])
        self.state_A = state_A_tilde @ self.rot.T
        self.state_B = state_B_tilde @ self.rot.T
        self.state_boundary = 0.25

        self.extent = np.array([-2,2, -2 ,2])
        self.levels = np.arange(0,8, 1)*scale
    
    def __repr__(self):
        return f"WolfeQuapp"
        
    def to_dict(self):
        dct = super().to_dict()
        return dct
    
    def V(self, sys):

        return self.pes.V(sys)
    
    def dVdx(self, sys):
        """
        -F = [dV(x, y) / dx, dV(x,y) / dy]
        """
        return self.pes.dVdx(sys)

    def stable_interface_function(self, snapshot, center):
        import math
        return math.sqrt((snapshot.xyz[0][0]-center[0])**2 + (snapshot.xyz[0][1]-center[1])**2)


class potential_Face(higher_d_harmonic):
    def __init__(self, n_harmonics=20, seed=123, rotation_degrees=0, d=1, scale=1):
        rotation = rotation_degrees / 180 * np.pi
        self.n_harmonics = n_harmonics
        self.seed = seed
        self.n_dims_pot = 2
        pes_list = [FacePotential(rotation=rotation, d=d, scale=scale)]
        self.create_pes(pes_list)  # Assuming this initializes the potential energy surface
        self.create_topology()  # Assuming this creates the topology of the system
        self.cos_rot = np.cos(-rotation) / d
        self.sin_rot = np.sin(-rotation) / d
        self.rot = np.array([[self.cos_rot, self.sin_rot], [-self.sin_rot, self.cos_rot]])

        # Define states in the rotated frame
        state_A_tilde = np.array([1.0, -1.0]) 
        state_B_tilde = np.array([1.0, 1.0])
        self.state_A = state_A_tilde @ self.rot.T
        self.state_B = state_B_tilde @ self.rot.T
        self.state_boundary = 0.125 
        self.extent = np.array([-1.5, 1.5, -1.5, 1.5]) 
        self.levels = np.arange(-2, 16, 1)

    def __repr__(self):
        return f"FacePotential"
    
    def to_dict(self):
        dct = super().to_dict()  # Assuming this calls the parent class to_dict
        return dct
    
    def V(self, sys):
        # Compute potential energy for the "Face" potential
        return self.pes.V(sys)
    
    def dVdx(self, sys):
        """
        -F = [dV(x, y) / dx, dV(x, y) / dy]
        """
        # Compute forces (gradient of the potential)
        return self.pes.dVdx(sys)

    def stable_interface_function(self, snapshot, center):
        import math
        return math.sqrt((snapshot.xyz[0][0]-center[0])**2 + (snapshot.xyz[0][1]-center[1])**2)
    
    def simple_initial_path(self, steps, toy_eng):
        end_1 = [-1,-1]
        end_2 = [-1,1]
        part_1 = self.linear_path(self.state_A, end_1, int(1/3*steps), toy_eng)
        part_2 = self.linear_path(end_1, end_2, int(1/3*steps), toy_eng)
        part_3 = self.linear_path(end_2, self.state_B, int(1/3*steps), toy_eng)
        path = []
        path += part_1
        path += part_2
        path += part_3
        return path

# simple toys.PES subclasses which are used inside the high-d potentials:

#from the openpathsampling github:
class DoubleWell(toys.PES):
    r"""Simple double-well potential. Independent in each degree of freedom.

    V(x) = \sum_i A_i * (x_i**2 - x0_i**2)**2

    WARNING: Two minima only in one dimension, otherwise there are more!

    Parameters
    ----------
    A : list of float
        potential prefactor for in each degree of freedom.
    x0 : list of float
        minimum position in each degree of freedom.
    """
    def __init__(self, A, x0):
        super(DoubleWell, self).__init__()
        self.A = np.array(A)
        self.x0 = np.array(x0)
        self._local_dVdx= None

    def __repr__(self):  # pragma: no cover
        repr_str = "DoubleWell({obj.A}, {obj.x0})"
        return repr_str.format(obj=self)

    def to_dict(self):
        dct = super(DoubleWell, self).to_dict()
        dct['A'] = dct['A'].tolist()
        dct['x0'] = dct['x0'].tolist()
        return dct

    def V(self, sys):
        """Potential energy

        Parameters
        ----------
        sys : :class:`.ToyEngine`
            engine contains its state, including velocities and masses

        Returns
        -------
        float
            the potential energy
        """
        dx2 = sys.positions[0] * sys.positions[0] - self.x0 * self.x0
        return np.dot(self.A, dx2 * dx2)

    def dVdx(self, sys):
        """Derivative of potential energy (-force)

        Parameters
        ----------
        sys : :class:`.ToyEngine`
            engine contains its state, including velocities and masses

        Returns
        -------
        np.array
            the derivatives of the potential at this point
        -F = [dV(x, y) / dx, dV(x,y) / dy]
        """
        if self._local_dVdx is None:
            self._local_dVdx = np.zeros_like(sys.positions)
        dx2 = sys.positions[0] * sys.positions[0] - self.x0 * self.x0
        self._local_dVdx[0] = 4 * self.A * sys.positions[0] * dx2
        return self._local_dVdx   


class Theoretical_linear_q(toys.PES):
    r"""Double well potential with linear committor q 


    \beta V(x) = -Ln(2+e^{-2x} + e^{2x}) 

    """
    def __init__(self):
        super(Theoretical_linear_q, self).__init__()
        self._local_dVdx= None

    def __repr__(self):  # pragma: no cover
        repr_str = "LinearQ"
        return repr_str.format(obj=self)

    def to_dict(self):
        dct = super(Theoretical_linear_q, self).to_dict()
        return dct

    def V(self, sys):
        """Potential energy

        Parameters
        ----------
        sys : :class:`.ToyEngine`
            engine contains its state, including velocities and masses

        Returns
        -------
        float
            the potential energy
        """
        x = sys.positions[0] 
        return  -np.log(2+np.exp(-2*x)+np.exp(2*x))

    def dVdx(self, sys):
        """Derivative of potential energy (-force)

        Parameters
        ----------
        sys : :class:`.ToyEngine`
            engine contains its state, including velocities and masses

        Returns
        -------
        np.array
            the derivatives of the potential at this point
        -F = [dV(x, y) / dx, dV(x,y) / dy]
        """
        if self._local_dVdx is None:
            self._local_dVdx = np.zeros_like(sys.positions)
        x = sys.positions[0] 
        self._local_dVdx[0] = -(-2*np.exp(-2*x)+2*np.exp(2*x))/(2+np.exp(-2*x)+np.exp(2*x))
        return self._local_dVdx   


# Define a Z_pot toy potential class which describes the 2d Z potential.
class Z_Pot(toys.PES):
    def __init__(self):
        self._local_dVdx = None
        self._dWalldx = None
        self._dgausdx = None
        self._dzpartdx = None  
    
    def __repr__(self):
        return f"Z potential."
        
    def to_dict(self):
        dct = super().to_dict()
        return dct
    
    def wall(self, x, y, sigma, x0):
        return sigma[0]*(x - x0[0])**4 + sigma[1]*(y - x0[1])**4
    
    def dWalldx(self, sys, sigma, x0):
        if self._dWalldx is None:
            self._dWalldx = np.zeros_like(sys.positions)
        x = sys.positions[0]
        y = sys.positions[1]
        self._dWalldx[0] = 4*sigma[0]*(x - x0[0])**3
        self._dWalldx[1]  = 4*sigma[1]*(y - x0[1])**3
        return self._dWalldx

    def gaus(self, x,y, A, sigma, x0):
        # Calculate the Gaussian component for each point in x_coords
        gaussian_component = A * np.exp(-sigma[0]*(x - x0[0])**2- sigma[1]*(y-x0[1])**2)   
        return gaussian_component 
    
    def dgausdx(self, sys, A, sigma, x0):
        if self._dgausdx is None:
            self._dgausdx = np.zeros_like(sys.positions)
        x = sys.positions[0]
        y = sys.positions[1]
        self._dgausdx[0] = -2*sigma[0]*(x - x0[0])*self.gaus(x,y,A,sigma,x0)
        self._dgausdx[1]  = -2*sigma[1]*(y - x0[1])*self.gaus(x,y,A,sigma,x0)
        return self._dgausdx

    def z_part(self, x, y, A, sigma, y0, beta, x0, alpha):
        gaussian_component = A * np.exp(-sigma*(x +beta*(y-y0))**2)/(1+np.exp(alpha*x-x0))   
        return gaussian_component 

    def dz_partdx(self, sys, A, sigma, y0, beta, x0, alpha):
        if self._dzpartdx is None:
            self._dzpartdx = np.zeros_like(sys.positions)
        x = sys.positions[0]
        y = sys.positions[1]
        self._dzpartdx[0] = (-2*sigma*(x +beta*(y-y0)) - alpha*np.exp(alpha*x-x0)/(1+np.exp(alpha*x-x0)))\
                            *self.z_part(x,y,A,sigma,y0,beta,x0,alpha)
        self._dzpartdx[1] = -2*beta*sigma*(x +beta*(y-y0))\
                            *self.z_part(x,y,A,sigma,y0,beta,x0,alpha)
        return self._dzpartdx


    def V(self, sys):
        x = sys.positions[0]
        y = sys.positions[1]
        U =self.wall(x, y, [1/20480, 1/20480], [0.0, 0.0]) \
            + self.gaus(x, y, -3, [0.01, 0.2], [5, 5])  \
            + self.gaus(x, y, -3, [0.01, 0.2], [-5, -5]) \
            + self.z_part(x, y, 5, 0.2, -3, 3 ,3, 1)  \
            + self.z_part(x, y, 5, 0.2, 3, 3 ,3, -1)  \
            + self.gaus(x, y, 3, [0.01, 0.01], [0, 0])

        return U
    
    def dVdx(self, sys):
        """
        -F = [dV(x, y) / dx, dV(x,y) / dy]
        """
        if self._local_dVdx is None:
            self._local_dVdx = np.zeros_like(sys.positions)
        self._local_dVdx = self.dWalldx(sys, [1/20480, 1/20480], [0.0, 0.0]) \
            + self.dgausdx(sys, -3, [0.01, 0.2], [5, 5])  \
            + self.dgausdx(sys, -3, [0.01, 0.2], [-5, -5]) \
            + self.dz_partdx(sys, 5, 0.2, -3, 3 ,3, 1)  \
            + self.dz_partdx(sys, 5, 0.2, 3, 3 ,3, -1)  \
            + self.dgausdx(sys, 3, [0.01, 0.01], [0, 0])
        return self._local_dVdx    

class WolfeQuapp(toys.PES):
    def __init__(self, rotation=45/180*np.pi, d=1, scale=2):

        self.scale= scale
        self.cos_rot = np.cos(-rotation)/d
        self.sin_rot = np.sin(-rotation)/d
        self.rot = np.array([[self.cos_rot,self.sin_rot],[-self.sin_rot,self.cos_rot]])
        self._local_dVdx = None

    def __repr__(self):
        return f"WolfeQuapp"
    
    def to_dict(self):
        dct = super().to_dict()
        return dct
    
    
    def V(self, sys):
        x = sys.positions[0]
        y = sys.positions[1]
        position_vec= np.array([x,y])

        r = position_vec @ self.rot.T
        U = self.scale * (r[0] ** 4 + r[1] ** 4 - 2. * r[0] ** 2 
                      - 4. *r[1] ** 2 + r[0] * r[1] + .3 * r[0] + .1 * r[1] +6.5)
        return U

    
    def dVdx(self, sys):
        """
        -F = [dV(x, y) / dx, dV(x,y) / dy]
        """
        if self._local_dVdx is None:
            self._local_dVdx = np.zeros_like(sys.positions)

        x = sys.positions[0]
        y = sys.positions[1]
        position_vec= np.array([x,y])
        r = position_vec @ self.rot.T
        xr = r[0]
        yr = r[1] 

        self._local_dVdx[0] = self.scale * (
                     4 * xr ** 3 * self.cos_rot - 4 * yr ** 3 * self.sin_rot
                    - 4 * xr * self.cos_rot + 8 * yr * self.sin_rot
                    + self.cos_rot * yr - self.sin_rot * xr + .3 * self.cos_rot - .1 * self.sin_rot
                    )

        self._local_dVdx[1] = self.scale * (
                    4 * xr ** 3 * self.sin_rot + 4 * yr ** 3 * self.cos_rot
                    - 4 * xr * self.sin_rot - 8 * yr * self.cos_rot
                    + self.sin_rot * yr + self.cos_rot * xr + .3 * self.sin_rot + .1 * self.cos_rot
                    )
        
        return self._local_dVdx    

#Define a Face Potential which has stable states at the same x coordinate but are oposite of a barrier allong y
class FacePotential(toys.PES):
    def __init__(self, rotation=0, d=1, scale=1):
        self.scale = scale
        self.cos_rot = np.cos(-rotation)/d
        self.sin_rot = np.sin(-rotation)/d
        self.rot = np.array([[self.cos_rot, self.sin_rot], [-self.sin_rot, self.cos_rot]])
        self._local_dVdx = None

    def __repr__(self):
        return f"FacePotential"
    
    def to_dict(self):
        dct = super().to_dict()
        return dct
    
    def V(self, sys):
        x = sys.positions[0]
        y = sys.positions[1]
        position_vec = np.array([x, y])
        
        # Apply rotation
        r = position_vec @ self.rot.T
        rx = r[0]
        ry = r[1]
        
        # Calculate potential energy
        U = self.scale * ((2.5 * (rx ** 2 - 1) ** 2 + 4 * (2.5 + rx) * (ry ** 2 - 1) ** 2
                          - 2.5 * rx + 0.5 * rx ** 4) + 4 * (rx ** 2 - 1) ** 2 * (ry ** 2 - 1) ** 2 + 1.5)
        return U

    def dVdx(self, sys):
        """
        -F = [dV(x, y) / dx, dV(x, y) / dy]
        """
        if self._local_dVdx is None:
            self._local_dVdx = np.zeros_like(sys.positions)

        x = sys.positions[0]
        y = sys.positions[1]
        position_vec = np.array([x, y])
        
        # Apply rotation
        r = position_vec @ self.rot.T
        rx = r[0]
        ry = r[1]

        # Derivatives for x
        self._local_dVdx[0] = self.scale * (
            16 * rx * (rx ** 2 - 1) * (ry ** 2 - 1) ** 2
            + (-2.5+ 2 * rx ** 3 + 10 * rx * (rx ** 2 - 1) + 4 * (ry ** 2 - 1) ** 2)
        )
        
        # Derivatives for y
        self._local_dVdx[1] = self.scale * (
            16 * (2.5 + rx) * ry * (ry ** 2 - 1)
            + 16 * (rx ** 2 - 1) ** 2 * ry * (ry ** 2 - 1)
        )
        
        return self._local_dVdx  
# Define a MuellerBrown toy potential class which describes the 2d Mueller Brown potential.
class MuellerBrown(toys.PES):
    # TODO if looking at this potential modify dvdx to only act on first two dimensions.

    def __init__(self, A, alpha, beta, gamma, a, b, max_u, scale):
        super(MuellerBrown, self).__init__()
        self.scale = scale
        self.A = np.array(A)
        self.alpha = np.array(alpha)
        self.beta = np.array(beta)
        self.gamma = np.array(gamma)
        self.a = np.array(a)
        self.b = np.array(b)
        self.max_u = max_u
        self._local_dVdx = None

    def to_dict(self):
        dct = super(MuellerBrown, self).to_dict()
        dct['alpha'] = dct['alpha'].tolist()
        dct['beta'] = dct['beta'].tolist()
        dct['gamma'] = dct['gamma'].tolist()
        dct['a'] = dct['a'].tolist()
        dct['b'] = dct['b'].tolist()
        # dct['max_u'] = dct['max_u'].tolist()
        # dct['scale'] = dct['scale'].tolist()
        return dct

    def __repr__(self):
        return "MuellerBrown({o.A}, {o.alpha}, {o.beta}, {o.gamma}, {o.a}, {o.b}, {o.max_u}, {o.scale})".format(o=self)

    def V(self, sys):
        x, y = sys.positions
        V = 0.0
        for k in range(0, len(self.a)):
            V += self.A[k] * np.exp(self.alpha[k] * np.power((x - self.a[k]), 2)
                                    + self.beta[k] * (x - self.a[k]) * (y - self.b[k])
                                    + self.gamma[k] * np.power((y - self.b[k]), 2))

        return self.scale * np.where(V > self.max_u, self.max_u, V)

    def dVdx(self, sys):
        x, y = sys.positions
        dVdx, dVdy = 0.0, 0.0
        for k in range(0, len(self.a)):
            dVdx += self.A[k] * np.exp(self.alpha[k] * np.power((x - self.a[k]), 2)
                                       + self.beta[k] * (x - self.a[k]) * (y - self.b[k])
                                       + self.gamma[k] * np.power((y - self.b[k]), 2)) \
                    * (2 * self.alpha[k] * (x - self.a[k]) + self.beta[k] * (y - self.b[k]))
        for k in range(0, len(self.a)):
            dVdy += self.A[k] * np.exp(self.alpha[k] * np.power((x - self.a[k]), 2)
                                       + self.beta[k] * (x - self.a[k]) * (y - self.b[k])
                                       + self.gamma[k] * np.power((y - self.b[k]), 2)) \
                    * (self.beta[k] * (x - self.a[k]) + 2 * self.gamma[k] * (y - self.b[k]))

        self._local_dVdx = np.array([self.scale * dVdx, self.scale * dVdy])
        return self._local_dVdx


class XYDiagpot(toys.PES):
    def __init__(self, b):
        self.b = b
        self._local_dVdx = None
    
    def __repr__(self):
        return f"XYDiagpot with barrier height {self.b}."
        
    def to_dict(self):
        dct = super().to_dict()
        dct["b"] = self.b
        return dct
    
    def V(self, sys):
        """
        V(x,y) = b ((x^2 - 1)^2 + (x - y)^2)
        """
        x = sys.positions[0]
        y = sys.positions[1]
        return self.b * ((x**2 - 1)**2 + (x - y)**2)
    
    def dVdx(self, sys):
        """
        -F = [dV(x, y) / dx, dV(x,y) / dy]
        """
        if self._local_dVdx is None:
            self._local_dVdx = np.zeros_like(sys.positions)
        x = sys.positions[0]
        y = sys.positions[1]
        self._local_dVdx[0] = 2 * self.b * (2 * x**3 - x - y)
        self._local_dVdx[1] = - 2 * self.b * (x - y)
        return self._local_dVdx    

class CallableVolume(object):
    def __init__(self, vol):
        self.vol = vol

    def __call__(self, x, y):
        snapshot = toys.Snapshot(coordinates=np.array([[x,y,0.0]]))
        return 1.0 if self.vol(snapshot) else 0.0

