import concurrent
import math

# values as per doi:10.1016/j.neuroimage.2008.03.061
dh = 0.1
extend_weight = 0.5
height_weight = 2


def tfces_from_distributions_st(distributions: list):
    tfces = []
    for distribution in distributions:
        tfce = tfce_from_distribution(distribution)
        tfces.append(tfce)
    return tfces


def tfces_from_distributions_mt(distributions: list, n_workers):
    tfces = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_url = {executor.submit(tfce_from_distribution, distrib): distrib for distrib in
                         distributions}
        for future in concurrent.futures.as_completed(future_to_url):
            try:
                tfce = future.result()
                tfces.append(tfce)
            except Exception as exc:
                print('Exception: {}'.format(exc))
    return tfces


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
                signum_at_index = distribution[index] / abs(distribution[index])
                if abs(distribution[index] < height) or signum_at_index != signum:
                    break
                extend += 1
                index += 1
            # reach backward
            index = i - 1
            while index >= 0:
                signum_at_index = distribution[index] / abs(distribution[index])
                if abs(distribution[index] < height) or signum_at_index != signum:
                    break
                extend += 1
                index -= 1
            tfce = tfce + math.pow(extend, extend_weight) * math.pow(height, height_weight)
            height += dh
        tfce = tfce * signum
        tfce_values.append(tfce)
    return tfce_values
