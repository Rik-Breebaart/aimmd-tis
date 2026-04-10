import numpy as np

def train_test_split(data, train_size, test_size, shuffle_indexs):
    if len(data) == len(shuffle_indexs):
        data_shuffled = data[shuffle_indexs]
    else:
        raise ValueError("Incorrect shuffle_index array given")

    if len(data_shuffled) == (train_size + test_size):
        return data_shuffled[:train_size], data_shuffled[train_size : train_size + test_size]

    raise ValueError("Train and test sizes do not match data length")


def create_train_test_split(descriptors,  shot_results, weights=None, split=[4, 1], seed=None):
    total_dataset_size = len(shot_results)
    ratio_train = split[0] / sum(split)
    train_size = int(np.floor(ratio_train * total_dataset_size))
    test_size = total_dataset_size - train_size
    if seed is not None:
        np.random.seed(seed)
    shuffle_rng_index = np.random.permutation(total_dataset_size)
    if weights == None:
        weights = np.ones(len(shot_results))
    weights_train, weights_test = train_test_split(weights, train_size, test_size, shuffle_rng_index)
    descriptors_train, descriptors_test = train_test_split(
        descriptors, train_size, test_size, shuffle_rng_index
    )
    shot_results_train, shot_results_test = train_test_split(
        shot_results, train_size, test_size, shuffle_rng_index
    )
    trainset = descriptors_train, shot_results_train, weights_train
    testset = descriptors_test, shot_results_test, weights_test
    return trainset, testset


class SyntheticDataGenerator:
    def __init__(self, potential_grid, p_B, pes, beta):
        self.potential_grid = potential_grid
        self.p_B = p_B
        self.n_x, self.n_y = p_B.shape
        self.pes_dim = pes.n_dims_pot + pes.n_harmonics
        self.range_pes = [[pes.extent[0], pes.extent[1]], [pes.extent[2], pes.extent[3]]]
        self.beta = beta

    def boltzmann_distribution(self, potential):
        return np.exp(-self.beta * potential)

    def generate_data(self, num_points):
        positions = np.zeros((num_points, self.pes_dim))
        shot_result = np.zeros((num_points, 2))

        random_x = np.random.uniform(0, self.n_x, num_points)
        random_y = np.random.uniform(0, self.n_y, num_points)
        x_int = np.floor(random_x).astype(int)
        y_int = np.floor(random_y).astype(int)
        random_u = np.random.uniform(0, 1, num_points)

        p_b = self.p_B[x_int, y_int]
        shot_result[:, 0] = random_u >= p_b
        shot_result[:, 1] = random_u < p_b

        potential = self.potential_grid[x_int, y_int]
        weights = self.boltzmann_distribution(potential)
        positions[:, 0] = (random_x / self.n_x) * (self.range_pes[0][1] - self.range_pes[0][0]) + self.range_pes[0][0]
        positions[:, 1] = (random_y / self.n_y) * (self.range_pes[1][1] - self.range_pes[1][0]) + self.range_pes[1][0]
        positions[:, 2:] = np.random.normal((self.pes_dim - 2))

        return positions, shot_result, weights


def q_normalized_trainset(
    descriptors,
    shot_results,
    weights,
    q_values,
    n_bins=40,
    floor_weight=1e-12,
):
    """Reweight samples so under-populated q regions contribute more uniformly."""
    q_values = np.asarray(q_values)
    weights = np.asarray(weights)
    if q_values.shape[0] != weights.shape[0]:
        raise ValueError("q_values and weights must have same length")

    bin_edges = np.linspace(np.nanmin(q_values), np.nanmax(q_values), int(n_bins) + 1)
    bin_index = np.clip(np.digitize(q_values, bin_edges) - 1, 0, n_bins - 1)
    counts = np.bincount(bin_index, minlength=n_bins).astype(float)
    counts[counts == 0.0] = 1.0

    inv_density = 1.0 / counts[bin_index]
    new_weights = weights * inv_density
    new_weights = np.maximum(new_weights, float(floor_weight))
    new_weights = new_weights / np.sum(new_weights)
    new_weights = new_weights * np.sum(weights)

    return descriptors, shot_results, new_weights
