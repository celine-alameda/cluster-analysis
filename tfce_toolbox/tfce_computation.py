import math

# values as per doi:10.1016/j.neuroimage.2008.03.061
dh = 0.1
extend_weight = 0.5
height_weight = 2


def shuffle_t_cluster_sign(rng, t_values: list):
    """shuffle the sign of the t values. Useful when the t-values are computed between two conditions.
    In this case it is equivalent to shuffling the two conditions"""
    ints = rng.integers(low=0, high=2, size=len(t_values)) * 2 - 1
    return t_values * ints


def shuffle_t_cluster_position(rng, t_values: list):
    """shuffle the position of the t values, with replacement (as should be for bootstrapping)"""
    ints = rng.integers(low=0, high=len(t_values), size=len(t_values))
    return [t_values[index] for index in ints]


def compute_resampling(t_resamplings, start, number_to_do):
    min_values = []
    max_values = []
    for i in range(start, start + number_to_do):
        resampling = t_resamplings[i]
        tfce_list = tfce_from_distribution(resampling)
        min_values.append(min(tfce_list))
        max_values.append(max(tfce_list))
    return min_values, max_values


def tfce_from_distribution(distribution: list):
    """Given a distribution (1D list of values), computes the Threshold-Free Cluster Enhancement"""
    tfce_values = []
    for i in range(len(distribution)):
        # floor to 0.1
        # notations are similar to those in the paper
        if distribution[i] == 0:
            tfce_values.append(0)
            continue
        signum = distribution[i] / abs(distribution[i])
        h_p = math.floor(abs(distribution[i]) / dh) * dh
        height = dh
        tfce = 0
        while height <= h_p:
            # extent is how many samples have values of at least h
            # reach forward
            extend = 1  # at least this sample
            index = i + 1
            while index < len(distribution):
                if abs(distribution[index] < height):
                    break
                extend += 1
                index += 1
            # reach backward
            index = i - 1
            while index > 0:
                if abs(distribution[index] < height):
                    break
                extend += 1
                index -= 1
            tfce = tfce + math.pow(extend, extend_weight) * math.pow(height, height_weight)
            height += dh
        tfce = tfce * signum
        tfce_values.append(tfce)
    return tfce_values
