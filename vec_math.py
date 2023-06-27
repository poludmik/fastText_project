import numpy as np
from scipy import spatial


def cosine_sim_scipy(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return 1 - spatial.distance.cosine(vec1, vec2)

def cosine_sim_numpy(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return (vec1 @ vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
