# main_cnn.py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.fft
from torch.utils.data import Dataset, DataLoader
from utils.splits import get_train_test_ids
from utils.data_loader import parse_damage_label, DAMAGE_LABELS
from models.cnn1d import PaperCnnBaseline
import pickle


class Cnn1DDataset(Dataset):
    def __init__(
        self,
        data_map,
        sample_id_list,
        fixed_fft_bin_indices,
        amp_means,
        amp_stds,
        lookback_fft=500,
    ):
        self.data_map = data_map
        self.sample_id_list = sample_id_list
        self.lookback_fft = lookback_fft
        self.fixed_fft_bin_indices = fixed_fft_bin_indices
        self.amp_means = amp_means
        self.amp_stds = amp_stds
        self.num_attention_freqs = len(fixed_fft_bin_indices)

    def __len__(self):
        return len(self.sample_id_list)

    def __getitem__(self, idx):
        sample_id = self.sample_id_list[idx]
        normalized_diff_signal = self.data_map[sample_id]
        num_pairs = normalized_diff_signal.shape[1]

        signal_for_fft = normalized_diff_signal[: self.lookback_fft, :]

        # Output shape for 1D CNN: (in_channels, seq_len) -> (num_pairs * 2, num_freqs)
        x_tensor = torch.zeros(
            (num_pairs * 2, self.num_attention_freqs), dtype=torch.float32
        )

        for pair_idx in range(num_pairs):
            fft_complex = scipy.fft.rfft(
                signal_for_fft[:, pair_idx], n=self.lookback_fft
            )
            amps = np.abs(fft_complex[self.fixed_fft_bin_indices])
            phases = np.angle(fft_complex[self.fixed_fft_bin_indices])

            normalized_amps = (amps - self.amp_means[pair_idx]) / self.amp_stds[
                pair_idx
            ]

            # Pack amps and phases as separate channels
            x_tensor[pair_idx * 2, :] = torch.from_numpy(normalized_amps).float()
            x_tensor[pair_idx * 2 + 1, :] = torch.from_numpy(phases).float()

        # Parse ground truth
        damage_id_str = parse_damage_label(sample_id)
        xd, yd = -0.001, -0.001
        if damage_id_str != "undamaged" and damage_id_str in DAMAGE_LABELS:
            coords = DAMAGE_LABELS[damage_id_str]
            xd, yd = float(coords[0]), float(coords[1])

        y_tensor = torch.tensor([xd, yd], dtype=torch.float)

        return x_tensor, y_tensor


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="A", choices=["A", "B"])
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("data/processed/ogw_data.pkl", "rb") as f:
        data_map = pickle.load(f)
    train_ids, test_ids = get_train_test_ids(args.split, list(data_map.keys()))

    # Mock parameters - replace with your saved parameters
    fixed_fft_bin_indices = np.arange(251)
    amp_means = np.zeros(66)  # Adjust if you have 36 pairs
    amp_stds = np.ones(66)
    num_sensor_pairs = list(data_map.values())[0].shape[1]

    train_dataset = Cnn1DDataset(
        data_map, train_ids, fixed_fft_bin_indices, amp_means, amp_stds
    )
    test_dataset = Cnn1DDataset(
        data_map, test_ids, fixed_fft_bin_indices, amp_means, amp_stds
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # In_channels = number of sensor pairs * 2 (amplitude and phase)
    in_channels = num_sensor_pairs * 2
    model = PaperCnnBaseline(in_channels=in_channels, num_classes=2).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print(f"--- Training 1D CNN Baseline on Split {args.split} ---")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)

        if epoch % 10 == 0 or epoch == 1:
            test_loss = evaluate(model, test_loader, criterion, device)
            print(
                f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}"
            )


if __name__ == "__main__":
    main()
