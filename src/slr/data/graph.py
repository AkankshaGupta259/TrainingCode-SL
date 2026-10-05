"""Skeleton graph for sign language (27-node upper-body + both-hands).

Node layout (output index -> meaning), chosen so hands (the dominant SLR signal)
are well represented while staying compact:

    0  nose
    1  left_shoulder   2  right_shoulder
    3  left_elbow      4  right_elbow
    5  left_wrist      6  right_wrist
    7..16   left hand : wrist, thumb_tip, index_mcp, index_tip, middle_mcp,
                        middle_tip, ring_mcp, ring_tip, pinky_mcp, pinky_tip
    17..26  right hand: (same 10-joint layout)

``NODE_SPEC`` ties each output node to a MediaPipe Holistic source, so the pose
extractor and the graph stay in sync. Swap both together to change the skeleton.
"""
from __future__ import annotations

import numpy as np

NUM_NODES = 27

# (output_index): ("pose"|"left_hand"|"right_hand", mediapipe_landmark_index)
NODE_SPEC = {
    0: ("pose", 0),    # nose
    1: ("pose", 11),   # left_shoulder
    2: ("pose", 12),   # right_shoulder
    3: ("pose", 13),   # left_elbow
    4: ("pose", 14),   # right_elbow
    5: ("pose", 15),   # left_wrist
    6: ("pose", 16),   # right_wrist
}
_HAND_LMS = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]  # wrist, thumb/index/middle/ring/pinky
for i, lm in enumerate(_HAND_LMS):
    NODE_SPEC[7 + i] = ("left_hand", lm)
    NODE_SPEC[17 + i] = ("right_hand", lm)

# Directed "inward" edges (child -> parent); undirected set is built from these.
INWARD = [
    (1, 0), (2, 0), (3, 1), (5, 3), (4, 2), (6, 4),  # body, centered on nose
    (7, 5), (17, 6),                                 # hand-wrist to arm-wrist
    # left hand (wrist = 7): 8 thumb,9 idx_mcp,10 idx_tip,11 mid_mcp,12 mid_tip,
    #                        13 ring_mcp,14 ring_tip,15 pinky_mcp,16 pinky_tip
    (8, 7), (9, 7), (10, 9), (11, 7), (12, 11),
    (13, 7), (14, 13), (15, 7), (16, 15),
    # right hand (wrist = 17)
    (18, 17), (19, 17), (20, 19), (21, 17), (22, 21),
    (23, 17), (24, 23), (25, 17), (26, 25),
]
CENTER = 0


def _hop_distance(num_node: int, edges: list, max_hop: int = 1) -> np.ndarray:
    A = np.zeros((num_node, num_node))
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1
    hop = np.full((num_node, num_node), np.inf)
    transfer = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive = (np.stack(transfer) > 0)
    for d in range(max_hop, -1, -1):
        hop[arrive[d]] = d
    return hop


def _normalize_digraph(A: np.ndarray) -> np.ndarray:
    Dl = A.sum(0)
    Dn = np.zeros_like(A)
    for i in range(A.shape[0]):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** -1
    return A @ Dn


class Graph:
    """Builds the adjacency tensor A of shape (K, V, V) for GCN models."""

    def __init__(self, strategy: str = "spatial", max_hop: int = 1, dilation: int = 1):
        self.max_hop = max_hop
        self.dilation = dilation
        self.num_node = NUM_NODES
        self.center = CENTER

        self_link = [(i, i) for i in range(self.num_node)]
        neighbor = INWARD + [(j, i) for (i, j) in INWARD]
        self.edge = self_link + neighbor
        self.hop_dis = _hop_distance(self.num_node, self.edge, max_hop)
        self.A = self._get_adjacency(strategy)

    def _get_adjacency(self, strategy: str) -> np.ndarray:
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        norm = _normalize_digraph(adjacency)

        if strategy == "uniform":
            return norm[np.newaxis, :]

        if strategy == "distance":
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = norm[self.hop_dis == hop]
            return A

        if strategy == "spatial":
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] != hop:
                            continue
                        if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                            a_root[j, i] = norm[j, i]
                        elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                            a_close[j, i] = norm[j, i]
                        else:
                            a_further[j, i] = norm[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            return np.stack(A)

        raise ValueError(f"Unknown strategy: {strategy}")
