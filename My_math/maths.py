import numpy as np


def build_B(list_edges: list[tuple[int, int]], N: int) -> np.ndarray:
    """
    Generate the incidence matrix for a graph.

    Parameters
    ----------
    list_edges : list of tuple[int, int]
        List of edges, where each edge is represented as a tuple (tail, head).
    N : int
        Number of nodes in the graph.

    Returns
    -------
    np.ndarray
        Incidence matrix of shape (N, E), where E is the number of edges.
    """
    B = np.zeros((N, len(list_edges)))
    for k, (tail, head) in enumerate(list_edges):
        B[tail, k] = 1
        B[head, k] = -1
    return B

