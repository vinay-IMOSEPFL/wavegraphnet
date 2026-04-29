# utils/data_loader.py
import torch
import numpy as np
import scipy.fft
from torch_geometric.data import Data as PyGData
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.loader import DataLoader as PyGDataLoader
import itertools

# Transducer physical coordinates and damage coordinates mapping
TRANSDUCER_COORDS = {
    1: [0.9, 0.94],
    2: [0.74, 0.94],
    3: [0.58, 0.94],
    4: [0.42, 0.94],
    5: [0.26, 0.94],
    6: [0.1, 0.94],
    7: [0.9, 0.06],
    8: [0.74, 0.06],
    9: [0.58, 0.06],
    10: [0.42, 0.06],
    11: [0.26, 0.06],
    12: [0.1, 0.06],
}

DAMAGE_LABELS = {
    "D1": [0.1, 0.83],
    "D2": [0.13, 0.83],
    "D3": [0.1, 0.8],
    "D4": [0.13, 0.8],
    "D5": [0.5, 0.854],
    "D6": [0.53, 0.854],
    "D7": [0.5, 0.824],
    "D8": [0.53, 0.824],
    "D9": [0.36, 0.69],
    "D10": [0.39, 0.69],
    "D11": [0.36, 0.66],
    "D12": [0.39, 0.66],
    "D13": [0.64, 0.55],
    "D14": [0.67, 0.55],
    "D15": [0.64, 0.52],
    "D16": [0.67, 0.52],
    "D17": [0.26, 0.39],
    "D18": [0.29, 0.39],
    "D19": [0.26, 0.36],
    "D20": [0.29, 0.36],
    "D21": [0.87, 0.41],
    "D22": [0.9, 0.41],
    "D23": [0.87, 0.38],
    "D24": [0.9, 0.38],
    "D25": [0.5, 0.18],
    "D26": [0.53, 0.18],
    "D27": [0.5, 0.15],
    "D28": [0.53, 0.15],
}


def get_k_graph_edge_index(num_nodes, self_loops=False):
    edges = list(itertools.combinations(range(num_nodes), 2))
    edge_list_undirected = []
    for u, v in edges:
        edge_list_undirected.append([u, v])
        edge_list_undirected.append([v, u])
    if self_loops:
        for i in range(num_nodes):
            edge_list_undirected.append([i, i])
    if not edge_list_undirected:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edge_list_undirected, dtype=torch.long).t().contiguous()


def parse_damage_label(damage_string: str) -> str:
    """Helper to extract damage ID from sample name."""
    if "baseline" in damage_string:
        return "undamaged"
    else:
        return damage_string.split("_")[0]


class CoupledModelDataset(TorchDataset):
    """Dataset for WaveGraphNet."""

    def __init__(
        self,
        data_map,
        sample_id_list,
        inv_static_edge_index,
        inv_edge_feature_col_idxs,
        fwd_propagation_col_idxs,
        fixed_fft_bin_indices,
        amp_means,
        amp_stds,
        lookback_fft,
        average_baseline_energy_profile,
        global_max_delta_e,
    ):

        self.data_map = data_map
        self.sample_id_list = sample_id_list
        self.node_coords_tensor = torch.tensor(
            np.array([TRANSDUCER_COORDS[i + 1] for i in range(12)]), dtype=torch.float
        )

        # Enforce bi-directional edges
        row, col = inv_static_edge_index
        bi_directional_edges = torch.cat(
            [torch.stack([row, col], dim=0), torch.stack([col, row], dim=0)], dim=1
        )
        self.inv_static_edge_index = torch.unique(bi_directional_edges, dim=1)

        self.inv_edge_feature_col_idxs = torch.as_tensor(
            inv_edge_feature_col_idxs, dtype=torch.long
        )
        self.lookback_fft = lookback_fft
        self.feature_dim_per_freq = 2
        self.fixed_fft_bin_indices = torch.as_tensor(
            fixed_fft_bin_indices, dtype=torch.long
        )
        self.amp_means = torch.tensor(amp_means, dtype=torch.float32)
        self.amp_stds = torch.tensor(amp_stds, dtype=torch.float32)
        self.num_attention_freqs = len(fixed_fft_bin_indices)

        self.global_max_delta_e = global_max_delta_e if global_max_delta_e > 0 else 1.0
        self.propagation_pair_indices = torch.tensor(
            sorted(list(set(fwd_propagation_col_idxs))), dtype=torch.long
        )
        self.average_baseline_energy_profile = average_baseline_energy_profile

    def __len__(self):
        return len(self.sample_id_list)

    def __getitem__(self, idx: int):
        sample_id = self.sample_id_list[idx]
        normalized_diff_signal = torch.from_numpy(self.data_map[sample_id]).float()
        num_pairs = normalized_diff_signal.shape[1]

        signal_for_fft = normalized_diff_signal[: self.lookback_fft, :].numpy()

        # Vectorized FFT
        fft_complex = scipy.fft.rfft(signal_for_fft, n=self.lookback_fft, axis=0)
        fft_complex = fft_complex[self.fixed_fft_bin_indices.numpy(), :]

        amps = torch.from_numpy(np.abs(fft_complex)).float()
        phases = torch.from_numpy(np.angle(fft_complex)).float()

        amp_means_arr = self.amp_means.view(1, num_pairs)
        amp_stds_arr = self.amp_stds.view(1, num_pairs)
        normalized_amps = (amps - amp_means_arr) / amp_stds_arr

        normalized_amps = normalized_amps.T
        phases = phases.T

        full_freq_profile = torch.stack([normalized_amps, phases], dim=-1)
        full_freq_profile_flat = full_freq_profile.view(num_pairs, -1)

        edge_attr_list = []
        for i in range(self.inv_static_edge_index.shape[1]):
            u, v = (
                self.inv_static_edge_index[0, i].item(),
                self.inv_static_edge_index[1, i].item(),
            )
            pair_idx = self.inv_edge_feature_col_idxs[
                i % len(self.inv_edge_feature_col_idxs)
            ]

            dist = (self.node_coords_tensor[u] - self.node_coords_tensor[v]).norm(
                keepdim=True
            )
            vec = self.node_coords_tensor[u] - self.node_coords_tensor[v]
            spatial_feats = torch.cat([dist, vec], dim=0)

            freq_feats = full_freq_profile_flat[pair_idx]
            edge_attr_list.append(torch.cat([spatial_feats, freq_feats], dim=0))

        edge_attr_inv = torch.stack(edge_attr_list)
        data_inv = PyGData(
            x=self.node_coords_tensor,
            edge_index=self.inv_static_edge_index,
            edge_attr=edge_attr_inv,
        )

        damage_id_str = parse_damage_label(sample_id)
        xd, yd = -0.001, -0.001
        if damage_id_str != "undamaged" and damage_id_str in DAMAGE_LABELS:
            coords = DAMAGE_LABELS[damage_id_str]
            xd, yd = float(coords[0]), float(coords[1])
        y_true = torch.tensor([[xd, yd]], dtype=torch.float)

        current_energy_profile = torch.abs(full_freq_profile[:, :, 0])
        delta_energy = (
            current_energy_profile
            - self.average_baseline_energy_profile.view(num_pairs, 1)
        )
        full_delta_e_map = torch.mean(delta_energy, dim=-1).clamp(min=0)
        unnormalized_delta_e_propagation = full_delta_e_map[
            self.propagation_pair_indices
        ]
        normalized_delta_e_true = (
            unnormalized_delta_e_propagation / self.global_max_delta_e
        )

        return {
            "data_inv": data_inv,
            "delta_e_true": normalized_delta_e_true,
            "y_true": y_true,
            "sample_id": sample_id,
        }


# Add to utils/data_loader.py


class StandardGraphDataset(TorchDataset):
    """Simplified Dataset for Baseline GNNs."""

    def __init__(
        self,
        data_map,
        sample_id_list,
        static_edge_index,
        edge_feature_col_idxs,
        fixed_fft_bin_indices,
        amp_means,
        amp_stds,
        lookback_fft=500,
    ):
        self.data_map = data_map
        self.sample_id_list = sample_id_list
        self.node_coords_tensor = torch.tensor(
            np.array([TRANSDUCER_COORDS[i + 1] for i in range(12)]), dtype=torch.float
        )

        # Enforce that all edges, including virtual ones, are bi-directional for symmetric information flow
        row, col = static_edge_index
        bi_directional_edges = torch.cat(
            [torch.stack([row, col], dim=0), torch.stack([col, row], dim=0)], dim=1
        )
        self.static_edge_index = torch.unique(bi_directional_edges, dim=1)

        self.edge_feature_col_idxs = torch.as_tensor(
            edge_feature_col_idxs, dtype=torch.long
        )
        self.lookback_fft = lookback_fft
        self.fixed_fft_bin_indices = torch.as_tensor(
            fixed_fft_bin_indices, dtype=torch.long
        )
        self.amp_means = torch.tensor(amp_means, dtype=torch.float32)
        self.amp_stds = torch.tensor(amp_stds, dtype=torch.float32)
        self.num_attention_freqs = len(fixed_fft_bin_indices)

    def __len__(self):
        return len(self.sample_id_list)

    def __getitem__(self, idx: int):
        sample_id = self.sample_id_list[idx]
        normalized_diff_signal = torch.from_numpy(self.data_map[sample_id]).float()
        num_pairs = normalized_diff_signal.shape[1]

        signal_for_fft = normalized_diff_signal[: self.lookback_fft, :].numpy()

        # VECTORIZED FFT
        fft_complex = scipy.fft.rfft(signal_for_fft, n=self.lookback_fft, axis=0)
        fft_complex = fft_complex[self.fixed_fft_bin_indices.numpy(), :]

        amps = torch.from_numpy(np.abs(fft_complex)).float()
        phases = torch.from_numpy(np.angle(fft_complex)).float()

        amp_means_arr = self.amp_means.view(1, num_pairs)
        amp_stds_arr = self.amp_stds.view(1, num_pairs)
        normalized_amps = (amps - amp_means_arr) / amp_stds_arr

        # Transpose to shape (num_pairs, num_freqs)
        normalized_amps = normalized_amps.T
        phases = phases.T

        # Stack into (num_pairs, num_freqs, 2) and flatten to (num_pairs, num_freqs * 2)
        full_freq_profile_flat = torch.stack([normalized_amps, phases], dim=-1).view(
            num_pairs, -1
        )

        # Map features to the bi-directional edge index
        edge_attr_list = []
        for i in range(self.static_edge_index.shape[1]):
            u, v = (
                self.static_edge_index[0, i].item(),
                self.static_edge_index[1, i].item(),
            )
            pair_idx = self.edge_feature_col_idxs[i % len(self.edge_feature_col_idxs)]

            dist = (self.node_coords_tensor[u] - self.node_coords_tensor[v]).norm(
                keepdim=True
            )
            vec = self.node_coords_tensor[u] - self.node_coords_tensor[v]
            spatial_feats = torch.cat([dist, vec], dim=0)

            freq_feats = full_freq_profile_flat[pair_idx]
            edge_attr_list.append(torch.cat([spatial_feats, freq_feats], dim=0))

        edge_attr = torch.stack(edge_attr_list)
        data = PyGData(
            x=self.node_coords_tensor,
            edge_index=self.static_edge_index,
            edge_attr=edge_attr,
        )

        damage_id_str = parse_damage_label(sample_id)
        xd, yd = -0.001, -0.001
        if damage_id_str != "undamaged" and damage_id_str in DAMAGE_LABELS:
            coords = DAMAGE_LABELS[damage_id_str]
            xd, yd = float(coords[0]), float(coords[1])
        data.y = torch.tensor([[xd, yd]], dtype=torch.float)

        return data
