import numpy as np
from openmm import app
import openmm as mm
from openmmtools.integrators import VVVRIntegrator
from simtk import unit
import openpathsampling as paths
import openpathsampling.engines.openmm as eng
from collections import namedtuple
import mdtraj as md 
mdtraj_host_guest = md.load("/Users/rbreeba/Documents/PhD/-RE-TIS-AIMMD/TIS_AIMMD_biosystems/AIMMD_TIS_openmm_run_scripts/HostGuest.pdb")
mdtraj_topology = mdtraj_host_guest.topology

guest_head_indices = np.array(mdtraj_topology.select("name C9 H15"))
guest_tail_indices = np.array(mdtraj_topology.select("name C10 H18"))
guest_center_indices = np.array(mdtraj_topology.select("resname B2 and type C and not (name C9 or name C10)"))
guest_indices_hbonds = np.array(mdtraj_topology.select("resname B2"))
guest_indices = np.array(mdtraj_topology.select("resname B2"))

dihedral_indices_bottom = np.array(mdtraj_topology.select("name H18 O2 C10 C3 and resname B2"))
dihedral_indices_top =  np.array(mdtraj_topology.select("name H15 O1 C9 C4 and resname B2"))

Guest_indices_orientation_top = np.array(mdtraj_topology.select("resname B2 and name C4"))
Guest_indices_orientation_bottom = np.array(mdtraj_topology.select("resname B2 and name C3"))


host_center_indices = np.array(mdtraj_topology.select("resname CUC and (type N or (name C3 C4 C9 C10 C15 C16 C21 C22 C27 C28 C33 C34 C39 C40))"))
host_indices_hbonds = np.array(mdtraj_topology.select("resname CUC and type O and type N and type H"))
host_indices = np.array(mdtraj_topology.select("resname CUC"))
# indices for groups to define normal plane vectors
indexGroup_1_Host = np.array(mdtraj_topology.select("resname CUC and name N1 N2 N3 N4 N5 N6"))
indexGroup_2_Host = np.array(mdtraj_topology.select("resname CUC and name N7 N8 N9 N10 N11 N12"))
indexGroup_3_Host = np.array(mdtraj_topology.select("resname CUC and name N13 N14 N15 N16 N17 N18"))

# only take the O to count a full water residue
water_indices = np.array(mdtraj_topology.select("water and type O"))
# take all atoms of the water molecules to count a water and count by deviding by 3 (smoother variable)
water_indices_all_atoms = np.array(mdtraj_topology.select("water and type O"))

def group_CM_vector(snapshot, group_A_atoms, group_B_atoms):
    import numpy as np

    A_com = np.mean(snapshot.xyz[group_A_atoms, :], axis=0)
    B_com = np.mean(snapshot.xyz[group_B_atoms, :], axis=0)

    box_lengths = np.diagonal(snapshot.box_vectors)  # shape (3,)

    delta = A_com - B_com
    delta -= box_lengths * np.round(delta / box_lengths)  # minimum image convention

    return delta


cv_hg_vec = paths.CoordinateFunctionCV(name="hg_vec",
    f=group_CM_vector,
    group_A_atoms=np.arange(0,126),
    group_B_atoms=np.arange(126,156)
).with_diskcache()


def compute_volume_square_box(snapshot):
    box_vector = snapshot.box_vectors
    x = box_vector[0,0]
    y = box_vector[1,1]
    z = box_vector[2,2]
    return x*y*z

cv_volume_square_box = paths.collectivevariables.FunctionCV("volume",f=compute_volume_square_box)

def distance(snapshot, vector):
    import numpy as np
    v = vector(snapshot)
    out = np.linalg.norm(v)
    return out

cv_hg_distance = paths.CoordinateFunctionCV(
    name="distance",
    f=distance,
    vector = cv_hg_vec
).with_diskcache()

# IndexGroup_1 = np.arange(0,6,1)
# IndexGroup_2 = np.arange(6,12,1)
# IndexGroup_3 = np.arange(18,24,1)
# host_indices = np.arange(0,126,1)
# guest_indices = np.arange(126,156,1)
# water_indices = np.arange(156,4491,1)

cv_host_vec_1 = paths.CoordinateFunctionCV(name="host_vec_1",
    f=group_CM_vector,
    group_A_atoms=indexGroup_1_Host,
    group_B_atoms=indexGroup_2_Host
).with_diskcache()

cv_host_vec_2 = paths.CoordinateFunctionCV(name="host_vec_2",
    f=group_CM_vector,
    group_A_atoms=indexGroup_1_Host,
    group_B_atoms=indexGroup_3_Host
).with_diskcache()


def norm_cross(snapshot, cv_vec_1, cv_vec_2):
    import numpy as np
    V_1 = cv_vec_1(snapshot)
    V_2 = cv_vec_2(snapshot)
    N = np.cross(V_1,V_2)
    return N/np.linalg.norm(N)

cv_normal_host =  paths.CoordinateFunctionCV(
    name="normal_host",
    f=norm_cross,
    cv_vec_1 = cv_host_vec_1,
    cv_vec_2 = cv_host_vec_2
).with_diskcache()

def host_guest_angle(snapshot, cv_hg_vec, cv_normal_host, cv_hg_distance):
    import numpy as np
    hg = cv_hg_vec(snapshot)/cv_hg_distance(snapshot)
    normal = cv_normal_host(snapshot)
    theta = np.arccos(np.clip(np.dot(hg, normal), -1.0, 1.0))
    return theta 

cv_hg_angle =  paths.CoordinateFunctionCV(
    name="hg_angle",
    f=host_guest_angle,
    cv_hg_vec = cv_hg_vec, 
    cv_normal_host = cv_normal_host, 
    cv_hg_distance= cv_hg_distance
).with_diskcache()

def host_guest_position_angle_symmetriced(snapshot,cv_hg_angle):
    import numpy as np
    theta = cv_hg_angle(snapshot)
    return np.min([theta, np.pi-theta])    

cv_hg_position_angle_symmetric =  paths.CoordinateFunctionCV(
    name="hg_position_angle",
    f=host_guest_position_angle_symmetriced,
    cv_hg_angle = cv_hg_angle
).with_diskcache()

cv_guest_vec = paths.CoordinateFunctionCV(
    name="guest_vec",
    f=group_CM_vector,
    group_A_atoms=[126+2], #C3 in guest
    group_B_atoms=[126+3] #C$ in guest
).with_diskcache()

def angle_guest_orientation(snapshot, cv_guest_vec, cv_normal_host):
    import numpy as np
    guest_vec = cv_guest_vec(snapshot)
    guest = guest_vec/np.linalg.norm(guest_vec)
    normal = cv_normal_host(snapshot)
    phi = np.arccos(np.dot(guest,normal))
    return phi

cv_angle_guest_orientation = paths.CoordinateFunctionCV(
    name="angle_guest_orientation",
    f=angle_guest_orientation,
    cv_guest_vec = cv_guest_vec,
    cv_normal_host = cv_normal_host
    # cv_scalarize_numpy_singletons=False,  # to make sure it always returns a 2d array, even if called on trajectories with a single frame
).with_diskcache()

def host_guest_angle_guest_orientation_symmetriced(snapshot,cv_angle_guest_orientation):
    import numpy as np
    theta = cv_angle_guest_orientation (snapshot)
    return np.min([theta, np.pi-theta])    

cv_hg_angle_guest_orientation_symmetric =  paths.CoordinateFunctionCV(
    name="angle_guest_orientation",
    f=host_guest_angle_guest_orientation_symmetriced,
    cv_angle_guest_orientation=cv_angle_guest_orientation
).with_diskcache()


def count_waters_in_cavity(snapshot, host_indices, water_indices, cavity_cutoff=0.4):
    import numpy as np
    host_com = np.mean(snapshot.xyz[host_indices], axis=0)  # Center of mass of the host
    box_lengths = snapshot.box_vectors.diagonal()  # Box lengths for periodic boundary conditions

    # Get water positions
    water_positions = snapshot.xyz[water_indices]

    # Calculate displacements using minimum image convention
    displacements = water_positions - host_com
    displacements -= box_lengths * np.round(displacements / box_lengths)

    # Compute distances
    distances = np.linalg.norm(displacements, axis=1)

    # Count waters within the cavity cutoff
    water_in_cavity_count = np.sum(distances < cavity_cutoff)

    return int(water_in_cavity_count)  # Ensure the return is an integer

# Create the CV in OpenPathSampling
cv_water_cavity = paths.FunctionCV(
    name="water_in_cavity",
    f=count_waters_in_cavity,
    host_indices=host_indices,
    water_indices=water_indices,
    cavity_cutoff=0.4
).with_diskcache()


# Define the hydrogen bond counting function
def count_hydrogen_bonds(snapshot,host_indices,guest_indices):
    import mdtraj as md
    import openpathsampling as paths
    # Verify that traj is an MDTraj Trajectory object
    traj = paths.Trajectory([snapshot]).to_mdtraj()

    if not isinstance(traj, md.Trajectory):
        raise TypeError("Expected MDTraj Trajectory object, got:", type(traj))
    # Calculate hydrogen bonds
    # using either the baker hubbard or wernet nilsson hbond calculation
    hbonds = md.wernet_nilsson(traj, exclude_water=True, periodic=True)[0]
    # hbonds = md.baker_hubbard(traj, exclude_water=True, periodic=True)
    # Process only the bonds for the first frame (assuming single snapshot trajectories)
    # Convert host and guest indices to sets for faster lookup
    host_indices_set = set(host_indices)
    guest_indices_set = set(guest_indices)

    # Filter hydrogen bonds for those between host and guest only checking acceptor and donor
    hbonds_host_guest = [
        hb for hb in hbonds
        if (hb[0] in host_indices_set and hb[2] in guest_indices_set) or 
           (hb[2] in host_indices_set and hb[0] in guest_indices_set)
    ]
    # Return the number of hydrogen bonds between host and guest
    return len(hbonds_host_guest)

# Create the OPS CV using MDTrajFunctionCV
cv_hydrogen_bond = paths.FunctionCV(
    name="hydrogen_bond_count",
    f=count_hydrogen_bonds,
    host_indices=host_indices,  # Indices of host atoms
    guest_indices=guest_indices # Indices of guest atoms
).with_diskcache()

def spherical_relative_to_COM_host(snapshot,host_indices, atom_indices):
    import numpy as np
    # Function to convert Cartesian coordinates to spherical coordinates
    def cartesian_to_spherical(x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(y, x)  # Azimuthal angle in the xy-plane
        phi = np.arccos(z / r) if r != 0 else 0  # Polar angle from the z-axis
        return r, theta, phi

    host_com = np.mean(snapshot.xyz[host_indices], axis=0)  # Compute COG in each call (the first dim is the atoms)
    box_lengths = snapshot.box_vectors.diagonal()  # Box lengths for PBC handling

    # Calculate spherical coordinates for each guest atom with PBC
    frame_spherical = np.zeros((len(atom_indices),3))
    for i, guest_idx in enumerate(atom_indices):
        # Calculate displacement with PBC
        guest_pos = snapshot.xyz[ guest_idx]
        displacement = guest_pos - host_com
        
        # Apply minimum image convention
        displacement -= box_lengths * np.round(displacement / box_lengths)
        
        # Convert to spherical coordinates
        r, theta, phi = cartesian_to_spherical(*displacement)
        frame_spherical[i,:] = np.array((r, theta, phi))

    return frame_spherical.flatten()

def distance_matrix_flattened(snapshot, atom_pairs):
    distance_matrix = mdtraj.compute_distances(snapshot, atom_pairs)

cv_spherical_guest = paths.CoordinateFunctionCV("spherical",
                                          spherical_relative_to_COM_host,
                                          host_indices=host_indices,
                                          atom_indices=guest_indices).with_diskcache()

cv_spherical_water = paths.CoordinateFunctionCV("spherical",
                                          spherical_relative_to_COM_host,
                                          host_indices=host_indices,
                                          atom_indices=water_indices).with_diskcache()

def hbond_onehot_descriptor(snapshot, host_indices, guest_indices):
    import mdtraj as md

    traj = paths.Trajectory([snapshot]).to_mdtraj()
    hbonds = md.wernet_nilsson(traj, exclude_water=True, periodic=True)[0]

    # Convert to set of unordered (donor, acceptor) pairs
    hbond_pairs = set(frozenset((d, a)) for d, h, a in hbonds)

    # Build unique host-guest pairs (no duplication)
    host_indices = np.array(host_indices)
    guest_indices = np.array(guest_indices)
    pair_list = [(i, j) for i in host_indices for j in guest_indices if i < j]

    # One-hot vector: 1 if pair in hbond set (symmetric)
    one_hot = np.array([frozenset((i, j)) in hbond_pairs for (i, j) in pair_list], dtype=int)

    return one_hot

def host_guest_distance_matrix(snapshot, host_indices, guest_indices):
    import mdtraj as md

    # Convert to MDTraj trajectory
    traj = paths.Trajectory([snapshot]).to_mdtraj()

    # Build unique, asymmetric pair list: i < j to avoid symmetric duplicates
    pair_list = [(i, j) for i in host_indices for j in guest_indices if i < j]

    # Use md.compute_distances to compute all distances at once
    distances = md.compute_distances(traj, pair_list, periodic=True)[0]  # shape: (n_pairs,)

    return distances

def make_pair_list(host_indices, guest_indices):
    return [(i, j) for i in host_indices for j in guest_indices if i < j]

def interpret_hbond_pairs_from_topology(top, pair_list, one_hot_vector):
    readable = []

    for (i, j), hot in zip(pair_list, one_hot_vector):
        a1 = top.atom(i)
        a2 = top.atom(j)
        status = "1" if hot else "0"
        if status == "1":
            readable.append({
                'status': status,
                'atom1_index': i,
                'atom1': f"{a1.name}-{a1.residue.name}{a1.residue.index}",
                'atom2_index': j,
                'atom2': f"{a2.name}-{a2.residue.name}{a2.residue.index}"
            })

    return readable

cv_one_hot_hydrogen_matrix = paths.FunctionCV("hbond_onehot_descriptor", hbond_onehot_descriptor, host_indices=host_indices, guest_indices=guest_indices)
cv_distance_matrix = paths.FunctionCV("host_guest_distance_matrix", host_guest_distance_matrix, host_indices=host_indices, guest_indices=guest_indices)




def transform_function_HG_v1(snapshot, receptor_atoms, ligand_atoms):
    import numpy as np
    dims = np.shape(snapshot.xyz)
    receptor_com = snapshot.xyz[:,receptor_atoms,:].mean(1)
    ligand_com = snapshot.xyz[:,ligand_atoms,:].mean(1)
    coordinates = snapshot.xyz[:,0:156,:]
    distances = np.sqrt(((receptor_com - ligand_com)**2).sum(1))
    return np.concatenate((coordinates.reshape(dims[0],156*3),distances.reshape(dims[0],1)) , axis=1)

descriptor_transform_v1 = paths.FunctionCV('descriptor_transform',  # name in OPS
                                            transform_function_HG_v1,  # the function we just defined
                                            receptor_atoms=host_indices,
                                            ligand_atoms = guest_indices
                                            ).with_diskcache()  # enable caching of values

def transform_function_HG_v2(snapshot, cv_hg_distance, cv_guest_angle_orientation, cv_hg_angle, cv_hydrogen_bond,cv_water_cavity,cv_spherical_guest):
    import numpy as np
    #devide water in cavity by 25 to roughly scale it between 0 and 1
    return np.concatenate((np.array((cv_hg_distance(snapshot), cv_guest_angle_orientation(snapshot),cv_hg_angle(snapshot),cv_hydrogen_bond(snapshot),cv_water_cavity(snapshot)/25)),cv_spherical_guest(snapshot)))
# wrap the function
descriptor_transform_v2 = paths.FunctionCV('descriptor_transform',  # name in OPS
                                            transform_function_HG_v2,  # the function we just defined
                                            cv_hg_distance = cv_hg_distance,
                                            cv_guest_angle_orientation=cv_angle_guest_orientation ,
                                            cv_hg_angle=cv_hg_angle,
                                            cv_hydrogen_bond=cv_hydrogen_bond,
                                            cv_water_cavity=cv_water_cavity,
                                            cv_spherical_guest=cv_spherical_guest,
                                            cv_wrap_numpy_array=True,
                                            cv_scalarize_numpy_singletons=False,  # to make sure it always returns a 2d array, even if called on trajectories with a single frame
                                            ).with_diskcache()  # enable caching of values   

def transform_function_HG_simple(snapshot, cv_hg_distance, cv_guest_angle_orientation, cv_host_guest_angle, cv_hydrogen_bond,cv_water_cavity):
    import numpy as np
    #devide water in cavity by 25 to roughly scale it between 0 and 1
    out = np.array((cv_hg_distance(snapshot), cv_guest_angle_orientation(snapshot),cv_host_guest_angle(snapshot),cv_hydrogen_bond(snapshot),cv_water_cavity(snapshot)/25))
    return out

descriptor_transform_HG_simple = paths.FunctionCV('descriptor_transform',  # name in OPS
                                            transform_function_HG_simple, 
                                            cv_hg_distance = cv_hg_distance,
                                            cv_guest_angle_orientation=cv_angle_guest_orientation ,
                                            cv_host_guest_angle=cv_hg_angle,
                                            cv_hydrogen_bond=cv_hydrogen_bond,
                                            cv_water_cavity=cv_water_cavity,
                                            cv_wrap_numpy_array=True,
                                            cv_scalarize_numpy_singletons=False,  # to make sure it always returns a 2d array, even if called on trajectories with a single frame
                                            ).with_diskcache()  # enable caching of values

descriptor_transform_HG_simple_symmetriced = paths.FunctionCV('descriptor_transform',  # name in OPS
                                            transform_function_HG_simple, 
                                            cv_hg_distance = cv_hg_distance,
                                            cv_guest_angle_orientation=cv_angle_guest_orientation ,
                                            cv_host_guest_angle=cv_hg_position_angle_symmetric,
                                            cv_hydrogen_bond=cv_hydrogen_bond,
                                            cv_water_cavity=cv_water_cavity,
                                            cv_wrap_numpy_array=True,
                                            cv_scalarize_numpy_singletons=False,  # to make sure it always returns a 2d array, even if called on trajectories with a single frame
                                            ).with_diskcache()  # enable caching of values

dim_label_HG_simple_symmetriced =  [
    "Center Distance",
    r"Guest Orientation Angle",
    r"Guest Position Angle",
    "H-Bonds",
    "Cavity Waters"
]
dim_scaled_unit_HG_simple_symmetriced = [
    "[nm)",
    r"[$\degree$) ",
    r"[$\degree$) ",
    "",
    ""
]
scaling_descriptors_HG_simple_symmetriced = [1,180/np.pi,180/np.pi,1,25]


def transform_function_HG_distance_hydrogen_matrix(
    snapshot,
    cv_distance_matrix,
    cv_one_hot_hydrogen_matrix,
    cv_guest_angle_orientation,
    cv_hg_position_angle,
    cv_water_cavity
):
    import numpy as np

    # Ensure everything is a flat 1D array before concatenating
    features = np.concatenate((
        np.ravel(cv_distance_matrix(snapshot)),
        np.ravel(cv_hydrogen_bond(snapshot)),
        np.ravel(cv_one_hot_hydrogen_matrix(snapshot)),
        np.ravel(cv_guest_angle_orientation(snapshot)),
        np.ravel(cv_hg_angle(snapshot)),
        np.ravel(cv_water_cavity(snapshot))
    ))

    return features

descriptor_transform_HG_distance_one_hot_hydrogen =  paths.FunctionCV('descriptor_transform',  # name in OPS
                                            transform_function_HG_distance_hydrogen_matrix,
                                            cv_distance_matrix=cv_distance_matrix,
                                            cv_one_hot_hydrogen_matrix=cv_one_hot_hydrogen_matrix,
                                            cv_guest_angle_orientation=cv_angle_guest_orientation ,
                                            cv_hg_position_angle= cv_hg_position_angle_symmetric,
                                            cv_water_cavity=cv_water_cavity,
                                            cv_wrap_numpy_array=True,
                                            cv_scalarize_numpy_singletons=False,  # to make sure it always returns a 2d array, even if called on trajectories with a single frame
                                            ).with_diskcache()  # enable caching of values



#### ------------------[ New CVs ] ------------------ ####

def hydrophobic_contact_score(snapshot, guest_indices, host_indices, r0=0.6, n=6, m=12):
    import numpy as np

    guest_coords = snapshot.xyz[guest_indices]
    host_coords = snapshot.xyz[host_indices]
    box_lengths = np.diagonal(snapshot.box_vectors)  # assuming square/orthorhombic box

    contact_sum = 0.0

    for g in guest_coords:
        for h in host_coords:
            # Apply minimum image convention
            delta = g - h
            delta -= box_lengths * np.round(delta / box_lengths)
            r = np.linalg.norm(delta)

            if r < r0:
                contact = (1 - (r / r0) ** n) / (1 - (r / r0) ** m)
                contact_sum += contact

    return contact_sum

cv_hydrophobic_contact_score = paths.FunctionCV(
    name="hydrophobic_contact_score",
    f=hydrophobic_contact_score,
    guest_indices=guest_center_indices,
    host_indices=host_center_indices,
    r0=0.5,
    n=6,
    m=12
).with_diskcache()


def compute_guest_OH_dihedral(snapshot, dihedral_indices):
    import mdtraj as md
    import openpathsampling as paths
    traj = paths.Trajectory([snapshot]).to_mdtraj()
    angle = md.compute_dihedrals(traj, [dihedral_indices])[0][0]  # single value
    return angle  # in radians


cv_guest_OH_dihedral_bottom = paths.FunctionCV(
    name="guest_OH_dihedral",
    f=compute_guest_OH_dihedral,
    dihedral_indices=dihedral_indices_bottom
).with_diskcache()

cv_guest_OH_dihedral_top = paths.FunctionCV(
    name="guest_OH_dihedral",
    f=compute_guest_OH_dihedral,
    dihedral_indices=dihedral_indices_top
).with_diskcache()


def water_near_guest_region(snapshot, region_indices, water_indices, cutoff=0.6):
    import numpy as np

    # Get region center (e.g., COM of guest head/tail)
    region_com = np.mean(snapshot.xyz[region_indices], axis=0)  # shape: (3,)
    water_positions = snapshot.xyz[water_indices]  # shape: (N, 3)

    # Get box lengths from diagonal of box_vectors (assumes square box)
    box_lengths = np.diagonal(snapshot.box_vectors)  # shape: (3,)

    # Apply minimum image convention to all displacements
    displacement = water_positions - region_com  # shape: (N, 3)
    displacement -= box_lengths * np.round(displacement / box_lengths)

    distances = np.linalg.norm(displacement, axis=1)
    return int(np.sum(distances < cutoff))

cutoff_waters_guest = 0.35
cv_water_guest_top = paths.FunctionCV(
    name="water_near_guest_top",
    f=water_near_guest_region,
    region_indices=guest_head_indices,
    water_indices=water_indices,
    cutoff=cutoff_waters_guest
).with_diskcache()

cv_water_guest_center = paths.FunctionCV(
    name="water_near_guest_center",
    f=water_near_guest_region,
    region_indices=guest_center_indices,
    water_indices=water_indices,
    cutoff=cutoff_waters_guest
).with_diskcache()

cv_water_guest_bottom = paths.FunctionCV(
    name="water_near_guest_bottom",
    f=water_near_guest_region,
    region_indices=guest_tail_indices,
    water_indices=water_indices,
    cutoff=cutoff_waters_guest
).with_diskcache()


def group_distance_to_host_com(snapshot, host_group_indices, atom_groups):
    import numpy as np

    # host_group_indices: all indices of atoms making up the full host (for host COM)
    # atom_groups: list of lists, each sublist contains atom indices of a group on the ring

    box_lengths = np.diagonal(snapshot.box_vectors)

    # Compute host COM
    host_com = np.mean(snapshot.xyz[host_group_indices], axis=0)

    # Distance from each group COM to host COM (with PBC)
    distances = []
    for group in atom_groups:
        group_com = np.mean(snapshot.xyz[group], axis=0)
        delta = group_com - host_com
        delta -= box_lengths * np.round(delta / box_lengths)
        dist = np.linalg.norm(delta)
        distances.append(dist)

    return np.array(distances)  # shape: (7,)

def host_mouth_deformation_from_cv(snapshot, cv_host_ring_deformation):
    import numpy as np
    distances = cv_host_ring_deformation(snapshot)
    return np.max(distances) - np.min(distances)


atom_groups_top = [
    np.array(mdtraj_topology.select("resname CUC and name N{} N{} C{} O{}".format(1,3,1,2))),
    np.array(mdtraj_topology.select("resname CUC and name N{} N{} C{} O{}".format(5,7,7,3))),
    np.array(mdtraj_topology.select("resname CUC and name N{} N{} C{} O{}".format(9,11,13,6))),
    np.array(mdtraj_topology.select("resname CUC and name N{} N{} C{} O{}".format(13,15,19,8))),
    np.array(mdtraj_topology.select("resname CUC and name N{} N{} C{} O{}".format(17,19,25,10))),
    np.array(mdtraj_topology.select("resname CUC and name N{} N{} C{} O{}".format(21,23,31,12))),
    np.array(mdtraj_topology.select("resname CUC and name N{} N{} C{} O{}".format(25,27,37,14)))
]

atom_groups_bottom = [
    np.array(mdtraj_topology.select("resname CUC and name N{} N{} C{} O{}".format(2,4,2,1))),
    np.array(mdtraj_topology.select("resname CUC and name N{} N{} C{} O{}".format(6,8,8,4))),
    np.array(mdtraj_topology.select("resname CUC and name N{} N{} C{} O{}".format(10,12,14,7))),
    np.array(mdtraj_topology.select("resname CUC and name N{} N{} C{} O{}".format(14,16,20,9))),
    np.array(mdtraj_topology.select("resname CUC and name N{} N{} C{} O{}".format(18,20,26,11))),
    np.array(mdtraj_topology.select("resname CUC and name N{} N{} C{} O{}".format(22,24,32,13))),
    np.array(mdtraj_topology.select("resname CUC and name N{} N{} C{} O{}".format(26,28,38,5)))
]

atom_groups_combined = [
    np.concatenate([top, bottom])
    for top, bottom in zip(atom_groups_top, atom_groups_bottom)
]

cv_host_ring_deformation_top = paths.FunctionCV(
    name="host_group_distances",
    f=group_distance_to_host_com,
    host_group_indices=np.array(atom_groups_top).flatten(),
    atom_groups=atom_groups_top,
    cv_wrap_numpy_array=True

).with_diskcache()
cv_host_ring_deformation_bottom = paths.FunctionCV(
    name="host_group_distances",
    f=group_distance_to_host_com,
    host_group_indices=np.array(atom_groups_bottom).flatten(),
    atom_groups=atom_groups_bottom,
    cv_wrap_numpy_array=True

).with_diskcache()
cv_host_ring_deformation = paths.FunctionCV(
    name="host_group_distances",
    f=group_distance_to_host_com,
    host_group_indices=host_indices,
    atom_groups=atom_groups_combined,
    cv_wrap_numpy_array=True
).with_diskcache()


cv_mouth_deformation_top = paths.FunctionCV(
    name="host_mouth_deformation",
    f=host_mouth_deformation_from_cv,
    cv_host_ring_deformation=cv_host_ring_deformation_top
).with_diskcache()
cv_mouth_deformation_bottom = paths.FunctionCV(
    name="host_mouth_deformation",
    f=host_mouth_deformation_from_cv,
    cv_host_ring_deformation=cv_host_ring_deformation_bottom,
).with_diskcache()
cv_mouth_deformation = paths.FunctionCV(
    name="host_mouth_deformation",
    f=host_mouth_deformation_from_cv,
    cv_host_ring_deformation=cv_host_ring_deformation
).with_diskcache()


def hydrogen_bond_asymmetry(snapshot, top_host_indices, bottom_host_indices, guest_indices):
    import mdtraj as md
    import openpathsampling as paths
    traj = paths.Trajectory([snapshot]).to_mdtraj()
    hbonds = md.wernet_nilsson(traj, exclude_water=True, periodic=True)[0]

    guest = set(guest_indices)
    top = set(top_host_indices)
    bottom = set(bottom_host_indices)

    top_hb = sum(1 for d,h,a in hbonds if ((d in guest and a in top) or (a in guest and d in top)))
    bot_hb = sum(1 for d,h,a in hbonds if ((d in guest and a in bottom) or (a in guest and d in bottom)))

    return top_hb - bot_hb  # signed difference

def list_Os(list):
    string = ""
    for i in list:
        string += f"O{i} "
    return string

top_host_indices = np.array(mdtraj_topology.select("resname CUC and name {}".format(list_Os([2,3,6,8,10,12,14]))))
bottom_host_indices = np.array(mdtraj_topology.select("resname CUC and name {}".format(list_Os([1,4,7,9,11,13,5]))))

cv_hbond_asymmetry = paths.FunctionCV(
    name="hbond_asymmetry",
    f=hydrogen_bond_asymmetry,
    top_host_indices=top_host_indices,
    bottom_host_indices=bottom_host_indices,
    guest_indices=guest_indices
).with_diskcache()



def shared_waters_between_host_guest(snapshot, host_indices, guest_indices, water_indices, cutoff=0.5):
    import numpy as np

    box_lengths = np.diagonal(snapshot.box_vectors)

    # Compute COMs of host and guest
    host_com = np.mean(snapshot.xyz[host_indices], axis=0)
    guest_com = np.mean(snapshot.xyz[guest_indices], axis=0)

    water_positions = snapshot.xyz[water_indices]

    # Minimum image convention: distance between water and COMs
    def pbc_distance(water_pos, com):
        delta = water_pos - com
        delta -= box_lengths * np.round(delta / box_lengths)
        return np.linalg.norm(delta, axis=1)

    d_host = pbc_distance(water_positions, host_com)
    d_guest = pbc_distance(water_positions, guest_com)

    shared = (d_host < cutoff) & (d_guest < cutoff)
    return int(np.sum(shared))


cv_water_shared_host_guest = paths.FunctionCV(
    name="water_shared_host_guest",
    f=shared_waters_between_host_guest,
    host_indices=host_indices,
    guest_indices=guest_indices,
    water_indices=water_indices,
    cutoff=0.6  # tighter cutoff to capture shared bridging waters
).with_diskcache()


def transform_function_HG_additional_CVs_symmetric(
    snapshot,
    cv_hg_distance,
    cv_guest_angle_orientation,
    cv_host_guest_angle,
    cv_hydrogen_bond,
    cv_water_cavity,
    cv_hydrophobic_contact_score,
    cv_guest_OH_dihedral_top,
    cv_guest_OH_dihedral_bottom,
    cv_water_near_guest_top,
    cv_water_near_guest_center,
    cv_water_near_guest_bottom,
    cv_water_shared_host_guest,
    cv_host_mouth_deformation,
    cv_hbond_asymmetry
):
    import numpy as np

    values = np.array([
        cv_hg_distance(snapshot),
        cv_guest_angle_orientation(snapshot),
        cv_host_guest_angle(snapshot),
        cv_hydrogen_bond(snapshot),
        cv_water_cavity(snapshot),
        cv_hydrophobic_contact_score(snapshot),
        cv_guest_OH_dihedral_top(snapshot),
        cv_guest_OH_dihedral_bottom(snapshot),
        cv_water_near_guest_top(snapshot),
        cv_water_near_guest_center(snapshot),
        cv_water_near_guest_bottom(snapshot),
        cv_water_shared_host_guest(snapshot),
        cv_host_mouth_deformation(snapshot),
        cv_hbond_asymmetry(snapshot)
    ], dtype=float)

    return values

descriptor_transform_HG_new_symmetric = paths.FunctionCV(
    name='descriptor_transform_scaled',
    f=transform_function_HG_additional_CVs_symmetric,
    cv_hg_distance=cv_hg_distance,
    cv_guest_angle_orientation=cv_angle_guest_orientation,
    cv_host_guest_angle=cv_hg_position_angle_symmetric,
    cv_hydrogen_bond=cv_hydrogen_bond,
    cv_water_cavity=cv_water_cavity,
    cv_hydrophobic_contact_score=cv_hydrophobic_contact_score,
    cv_guest_OH_dihedral_top=cv_guest_OH_dihedral_top,
    cv_guest_OH_dihedral_bottom=cv_guest_OH_dihedral_bottom,
    cv_water_near_guest_top=cv_water_guest_top,
    cv_water_near_guest_center=cv_water_guest_center,
    cv_water_near_guest_bottom=cv_water_guest_bottom,
    cv_water_shared_host_guest=cv_water_shared_host_guest,
    cv_host_mouth_deformation=cv_mouth_deformation,
    cv_hbond_asymmetry=cv_hbond_asymmetry,
    cv_wrap_numpy_array=True,
    cv_scalarize_numpy_singletons=False
).with_diskcache()


descriptor_labels_new = [
    "Host-Guest Distance",
    "Guest Orientation Angle",
    "Guest Position Angle",
    "Hydrogen Bonds",
    "Waters in Cavity",
    "Hydrophobic Contact Score",
    "Guest OH Dihedral (Top)",
    "Guest OH Dihedral (Bottom)",
    "Waters Near Guest Head",
    "Waters Near Guest Center",
    "Waters Near Guest Tail",
    "Shared Waters",
    "Mouth Deformation",
    "H-Bond Asymmetry"
]

descriptor_units_new = [
    "[nm]",
    "[rad]",
    "[rad]",
    "",
    "",
    "",
    "[rad]",
    "[rad]",
    "",
    "",
    "",
    "",
    "[nm]",
    ""
]



descriptor_new_min = np.array([
    0.0,      # Host-Guest Distance
    0.0,      # Guest Orientation
    0.0,      # Guest Position
    0.0,      # H-bonds
    0.0,      # Waters in cavity
    0.0,      # Hydrophobic contact score
    -np.pi,      # OH dihedral top
    -np.pi,      # OH dihedral bottom
    0.0,      # Water guest head
    0.0,      # Water guest center
    0.0,      # Water guest tail
    0.0,      # Shared waters
    0.0,      # Mouth deformation 
    -2.0      # Hydrogen bond asymmetry (can be negative)
])

descriptor_new_max = np.array([
    1.0,       # Host-Guest Distance
    np.pi,     # Guest Orientation
    np.pi,     # Guest Position
    4.0,       # H-bonds
    10.0,      # Waters in cavity
    60.0,      # Hydrophobic contact score
    np.pi,     # OH dihedral top
    np.pi,     # OH dihedral bottom
    10.0,      # Water guest head
    10.0,      # Water guest center
    10.0,      # Water guest tail
    10.0,      # Shared waters
    0.2,       # Mouth deformation
    2.0        # Hydrogen bond asymmetry
])


def scale_descriptors(snapshot, descriptor_transform, descriptor_min, descriptor_max):
    values = descriptor_transform(snapshot)
    scaled = (values - descriptor_min) / (descriptor_max - descriptor_min)
    return scaled

def unscale_descriptors(scaled_values, descriptor_min, descriptor_max):
    return scaled_values * (descriptor_max - descriptor_min) + descriptor_min


                               
descriptor_transform_HG_new_symmetric_scaled= paths.FunctionCV(
    name='descriptor_transform_scaled',
    f=scale_descriptors,
    descriptor_transform = descriptor_transform_HG_new_symmetric,
    descriptor_min= descriptor_new_min,
    descriptor_max= descriptor_new_max,
    cv_wrap_numpy_array=True,
    cv_scalarize_numpy_singletons=False
).with_diskcache()


### Smoothned descriptor functions 
def count_waters_in_cavity_continuous(snapshot, host_indices, water_indices, cavity_cutoff=0.4):
    import numpy as np
    host_com = np.mean(snapshot.xyz[host_indices], axis=0)  # Center of mass of the host
    box_lengths = snapshot.box_vectors.diagonal()  # Box lengths for periodic boundary conditions

    # Get water positions
    water_positions = snapshot.xyz[water_indices]

    # Calculate displacements using minimum image convention
    displacements = water_positions - host_com
    displacements -= box_lengths * np.round(displacements / box_lengths)

    # Compute distances
    distances = np.linalg.norm(displacements, axis=1)

    # Count waters within the cavity cutoff
    continuous_water_metric = 1/(1+np.exp(30*(distances-cavity_cutoff)))
    water_in_cavity_count = np.sum(continuous_water_metric)

    return water_in_cavity_count  # Ensure the return is an integer

# Create the CV in OpenPathSampling
cv_water_cavity_continuous = paths.FunctionCV(
    name="water_in_cavity",
    f=count_waters_in_cavity_continuous,
    host_indices=host_indices,
    water_indices=water_indices,
    cavity_cutoff=0.5
).with_diskcache()


def water_near_guest_region_continuous(snapshot, region_indices, water_indices, cutoff=0.6):
    import numpy as np

    # Get region center (e.g., COM of guest head/tail)
    region_com = np.mean(snapshot.xyz[region_indices], axis=0)  # shape: (3,)
    water_positions = snapshot.xyz[water_indices]  # shape: (N, 3)

    # Get box lengths from diagonal of box_vectors (assumes square box)
    box_lengths = np.diagonal(snapshot.box_vectors)  # shape: (3,)

    # Apply minimum image convention to all displacements
    displacement = water_positions - region_com  # shape: (N, 3)
    displacement -= box_lengths * np.round(displacement / box_lengths)

    distances = np.linalg.norm(displacement, axis=1)

    continuous_water_metric = 1/(1+np.exp(30*(distances-cutoff)))
    water_near_region = np.sum(continuous_water_metric)
    return water_near_region

cutoff_waters_guest = 0.35
cv_water_guest_top_continuous = paths.FunctionCV(
    name="water_near_guest_top",
    f=water_near_guest_region_continuous,
    region_indices=guest_head_indices,
    water_indices=water_indices,
    cutoff=cutoff_waters_guest
).with_diskcache()

cv_water_guest_center_continuous = paths.FunctionCV(
    name="water_near_guest_center",
    f=water_near_guest_region_continuous,
    region_indices=guest_center_indices,
    water_indices=water_indices,
    cutoff=cutoff_waters_guest
).with_diskcache()

cv_water_guest_bottom_continuous = paths.FunctionCV(
    name="water_near_guest_bottom",
    f=water_near_guest_region_continuous,
    region_indices=guest_tail_indices,
    water_indices=water_indices,
    cutoff=cutoff_waters_guest
).with_diskcache()



def shared_waters_between_host_guest_continuous(snapshot, host_indices, guest_indices, water_indices, cutoff=0.5):
    import numpy as np

    box_lengths = np.diagonal(snapshot.box_vectors)

    # Compute COMs of host and guest
    host_com = np.mean(snapshot.xyz[host_indices], axis=0)
    guest_com = np.mean(snapshot.xyz[guest_indices], axis=0)

    water_positions = snapshot.xyz[water_indices]

    # Minimum image convention: distance between water and COMs
    def pbc_distance(water_pos, com):
        delta = water_pos - com
        delta -= box_lengths * np.round(delta / box_lengths)
        return np.linalg.norm(delta, axis=1)

    d_host = pbc_distance(water_positions, host_com)
    d_guest = pbc_distance(water_positions, guest_com)
    shared = 1/(1+np.exp(30*(d_host-cutoff))*(1+np.exp(30*(d_guest-cutoff))))
    return np.sum(shared)


cv_water_shared_host_guest_continuous = paths.FunctionCV(
    name="water_shared_host_guest",
    f=shared_waters_between_host_guest_continuous,
    host_indices=host_indices,
    guest_indices=guest_indices,
    water_indices=water_indices,
    cutoff=0.5  
).with_diskcache()



descriptor_transform_HG_new_symmetric_continuous_waters = paths.FunctionCV(
    name='descriptor_transform_scaled',
    f=transform_function_HG_additional_CVs_symmetric,
    cv_hg_distance=cv_hg_distance,
    cv_guest_angle_orientation=cv_angle_guest_orientation,
    cv_host_guest_angle=cv_hg_position_angle_symmetric,
    cv_hydrogen_bond=cv_hydrogen_bond,
    cv_water_cavity=cv_water_cavity_continuous,
    cv_hydrophobic_contact_score=cv_hydrophobic_contact_score,
    cv_guest_OH_dihedral_top=cv_guest_OH_dihedral_top,
    cv_guest_OH_dihedral_bottom=cv_guest_OH_dihedral_bottom,
    cv_water_near_guest_top=cv_water_guest_top_continuous,
    cv_water_near_guest_center=cv_water_guest_center_continuous,
    cv_water_near_guest_bottom=cv_water_guest_bottom_continuous,
    cv_water_shared_host_guest=cv_water_shared_host_guest_continuous,
    cv_host_mouth_deformation=cv_mouth_deformation,
    cv_hbond_asymmetry=cv_hbond_asymmetry,
    cv_wrap_numpy_array=True,
    cv_scalarize_numpy_singletons=False
).with_diskcache()

descriptor_transform_HG_new_symmetric_continuous_waters_scaled= paths.FunctionCV(
    name='descriptor_transform_scaled',
    f=scale_descriptors,
    descriptor_transform = descriptor_transform_HG_new_symmetric_continuous_waters,
    descriptor_min= descriptor_new_min,
    descriptor_max= descriptor_new_max,
    cv_wrap_numpy_array=True,
    cv_scalarize_numpy_singletons=False
).with_diskcache()



descriptors_index_7_set_from_14= [0,1,2,3,4,5,11]
descriptors_7_set_min = np.array(descriptor_new_min)[descriptors_index_7_set_from_14]
descriptors_7_set_max = np.array(descriptor_new_max)[descriptors_index_7_set_from_14]


def transform_function_HG_7_descriptors(
    snapshot,
    cv_hg_distance,
    cv_guest_angle_orientation,
    cv_host_guest_angle,
    cv_hydrogen_bond,
    cv_water_cavity,
    cv_hydrophobic_contact_score,
    cv_water_shared_host_guest,
):
    import numpy as np

    values = np.array([
        cv_hg_distance(snapshot),
        cv_guest_angle_orientation(snapshot),
        cv_host_guest_angle(snapshot),
        cv_hydrogen_bond(snapshot),
        cv_water_cavity(snapshot),
        cv_hydrophobic_contact_score(snapshot),
        cv_water_shared_host_guest(snapshot),
    ], dtype=float)

    return values

descriptor_transform_HG_continuous_waters_7_descriptors = paths.FunctionCV(
    name='descriptor_transform_scaled',
    f=transform_function_HG_7_descriptors,
    cv_hg_distance=cv_hg_distance,
    cv_guest_angle_orientation=cv_angle_guest_orientation,
    cv_host_guest_angle=cv_hg_position_angle_symmetric,
    cv_hydrogen_bond=cv_hydrogen_bond,
    cv_water_cavity=cv_water_cavity_continuous,
    cv_hydrophobic_contact_score=cv_hydrophobic_contact_score,
    cv_water_shared_host_guest=cv_water_shared_host_guest_continuous,
    cv_wrap_numpy_array=True,
    cv_scalarize_numpy_singletons=False
).with_diskcache()

descriptor_transform_HG_continuous_waters_7_descriptors_scaled= paths.FunctionCV(
    name='descriptor_transform_scaled',
    f=scale_descriptors,
    descriptor_transform = descriptor_transform_HG_continuous_waters_7_descriptors,
    descriptor_min= descriptors_7_set_min,
    descriptor_max= descriptors_7_set_max,
    cv_wrap_numpy_array=True,
    cv_scalarize_numpy_singletons=False
).with_diskcache()

descriptor_transform_HG_continuous_waters_7_descriptors_symmetriced_or = paths.FunctionCV(
    name='descriptor_transform_scaled',
    f=transform_function_HG_7_descriptors,
    cv_hg_distance=cv_hg_distance,
    cv_guest_angle_orientation=cv_hg_angle_guest_orientation_symmetric,
    cv_host_guest_angle=cv_hg_position_angle_symmetric,
    cv_hydrogen_bond=cv_hydrogen_bond,
    cv_water_cavity=cv_water_cavity_continuous,
    cv_hydrophobic_contact_score=cv_hydrophobic_contact_score,
    cv_water_shared_host_guest=cv_water_shared_host_guest_continuous,
    cv_wrap_numpy_array=True,
    cv_scalarize_numpy_singletons=False
).with_diskcache()

descriptor_transform_HG_continuous_waters_7_descriptors_symmetriced_or_scaled= paths.FunctionCV(
    name='descriptor_transform_scaled',
    f=scale_descriptors,
    descriptor_transform = descriptor_transform_HG_continuous_waters_7_descriptors_symmetriced_or,
    descriptor_min= descriptors_7_set_min,
    descriptor_max= descriptors_7_set_max,
    cv_wrap_numpy_array=True,
    cv_scalarize_numpy_singletons=False
).with_diskcache()