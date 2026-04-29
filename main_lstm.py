# main_lstm.py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.fft
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.splits import get_train_test_ids
from utils.data_loader import parse_damage_label, DAMAGE_LABELS
from models.lstm import LSTM_baseline
import pickle


class LstmDataset(Dataset):
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

        # 1. VECTORIZED FFT: Compute all pairs simultaneously over axis 0 (time)
        fft_complex = scipy.fft.rfft(signal_for_fft, n=self.lookback_fft, axis=0)

        # 2. Extract only the bins we care about
        fft_complex = fft_complex[self.fixed_fft_bin_indices, :]

        amps = np.abs(fft_complex)
        phases = np.angle(fft_complex)

        # 3. Vectorized Normalization (broadcast over the freq dimension)
        amp_means_arr = self.amp_means.reshape(1, num_pairs)
        amp_stds_arr = self.amp_stds.reshape(1, num_pairs)
        normalized_amps = (amps - amp_means_arr) / amp_stds_arr

        # 4. Construct Output Tensor shape: (num_pairs, num_freqs, 2)
        x_tensor = torch.zeros(
            (num_pairs, self.num_attention_freqs, 2), dtype=torch.float32
        )

        # Transpose so shapes align to (num_pairs, num_freqs)
        x_tensor[:, :, 0] = torch.from_numpy(normalized_amps.T).float()
        x_tensor[:, :, 1] = torch.from_numpy(phases.T).float()

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
    # Defaulting batch_size to 16 instead of 32; LSTMs eat a massive amount of VRAM
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("data/processed/ogw_data.pkl", "rb") as f:
        data_map = pickle.load(f)
    train_ids, test_ids = get_train_test_ids(args.split, list(data_map.keys()))

    fixed_fft_bin_indices = np.arange(251)

    # Automatically generate correct lengths for means and stds based on the data
    num_sensor_pairs = list(data_map.values())[0].shape[1]
    amp_means = np.zeros(num_sensor_pairs)
    amp_stds = np.ones(num_sensor_pairs)

    train_dataset = LstmDataset(
        data_map, train_ids, fixed_fft_bin_indices, amp_means, amp_stds
    )
    test_dataset = LstmDataset(
        data_map, test_ids, fixed_fft_bin_indices, amp_means, amp_stds
    )

    # Added num_workers=2 to load batches in the background
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    model = LSTM_baseline(
        num_freqs=251,
        feature_dim_per_freq=2,
        num_sensor_pairs=num_sensor_pairs,
        lstm_hidden_dim=256,
        output_dim=2,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print(f"--- Training LSTM Baseline on Split {args.split} ---")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0

        # Wrapped the loader in tqdm so you can SEE it moving
        loader_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch:03d}/{args.epochs}", leave=False
        )

        for x, y in loader_pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            loader_pbar.set_postfix(loss=loss.item())

        train_loss /= len(train_loader.dataset)

        if epoch % 10 == 0 or epoch == 1:
            test_loss = evaluate(model, test_loader, criterion, device)
            print(
                f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}"
            )


if __name__ == "__main__":
    main()
