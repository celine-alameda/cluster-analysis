import math

# values as per doi:10.1016/j.neuroimage.2008.03.061
dh = 0.1
extend_weight = 0.5
height_weight = 2



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
