import numpy as np

# percentile
def get_confident_interval(scores, reliability=95):
    if reliability==95:
        q = 5
    return np.percentile(scores, [q, 100-q], interpolation="nearest")
