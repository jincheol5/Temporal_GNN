import pandas as pd
from collections import defaultdict

class NeighborSampler:
    """
    node_id=0: padding node
    """
    @staticmethod
    def find_temporal_neighbor(
            adj:defaultdict,
            num_neighbor:int,
            seed:int
        ):
        """
        Input:
        Return:
            neighbor_ids: [B,num_neighbor]
            neighbor_times: [B,num_neighbor]
            edge_ids: [B,num_neighbor]
        """