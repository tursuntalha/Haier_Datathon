import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class PatchTSTDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class PatchEmbedding(nn.Module):
    def __init__(self, patch_len, stride, in_channels, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.projection = nn.Linear(patch_len, d_model)

    def forward(self, x):
        batch_size, seq_len, n_vars = x.shape
        x = x.permute(0, 2, 1)

        patches = []
        for i in range(0, seq_len - self.patch_len + 1, self.stride):
            patch = x[:, :, i:i + self.patch_len]
            patches.append(patch)

        x = torch.stack(patches, dim=2)
        x = self.projection(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class PatchTST(nn.Module):
    def __init__(
        self,
        seq_len=36,
        pred_len=12,
        patch_len=6,
        stride=3,
        d_model=64,
        n_heads=4,
        d_ff=128,
        n_layers=3,
        dropout=0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride

        n_patches = ((seq_len - patch_len) // stride) + 1
        self.patch_embedding = PatchEmbedding(patch_len, stride, 1, d_model)

        self.pos_embedding = nn.Parameter(torch.randn(1, 1, n_patches, d_model))

        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_patches * d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, pred_len),
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.patch_embedding(x)
        x = x + self.pos_embedding
        x = x.squeeze(1)

        for layer in self.transformer_layers:
            x = layer(x)

        x = self.head(x)
        return x


def prepare_patchtst_data(train_df, product_df, seq_len=36, pred_len=12):
    train = train_df.copy()
    train['date'] = pd.to_datetime(train['date'])
    train = train.sort_values(['product_code', 'market', 'date'])

    scaler = StandardScaler()
    sales_scaled = scaler.fit_transform(train[['quantity']].fillna(0))

    sequences, targets = [], []
    for _, group in train.groupby(['product_code', 'market']):
        values = group['quantity'].values
        if len(values) >= seq_len + pred_len:
            for i in range(len(values) - seq_len - pred_len + 1):
                sequences.append(values[i:i + seq_len])
                targets.append(values[i + seq_len:i + seq_len + pred_len])

    if len(sequences) == 0:
        print("Warning: Not enough data to create sequences. Try reducing seq_len/pred_len.")
        return None, None, None, None, None

    sequences = np.array(sequences)
    targets = np.array(targets)

    split_idx = int(len(sequences) * 0.8)
    train_seq, val_seq = sequences[:split_idx], sequences[split_idx:]
    train_tgt, val_tgt = targets[:split_idx], targets[split_idx:]

    train_dataset = PatchTSTDataset(train_seq, train_tgt)
    val_dataset = PatchTSTDataset(val_seq, val_tgt)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, scaler, sequences.shape, targets.shape


def train_patchtst(train_loader, val_loader, seq_len=36, pred_len=12, epochs=50, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = PatchTST(seq_len=seq_len, pred_len=pred_len).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_train_loss += loss.item()

        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = model(batch_x)
                loss = criterion(output, batch_y)
                epoch_val_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    return model, train_losses, val_losses


def compare_with_baseline(train_df, product_df, seq_len=36, pred_len=12):
    print("\n" + "=" * 60)
    print("PatchTST Experiment")
    print("=" * 60)

    result = prepare_patchtst_data(train_df, product_df, seq_len, pred_len)
    train_loader, val_loader, scaler, train_shape, target_shape = result

    if train_loader is None:
        print("Insufficient data for PatchTST training.")
        return None

    print(f"Training samples: {train_shape[0]}, Sequence length: {seq_len}, Prediction length: {pred_len}")

    model, train_losses, val_losses = train_patchtst(train_loader, val_loader, seq_len, pred_len)

    final_train_loss = train_losses[-1] if train_losses else None
    final_val_loss = val_losses[-1] if val_losses else None

    print(f"\nFinal Train Loss (MSE): {final_train_loss:.6f}")
    print(f"Final Val Loss (MSE): {final_val_loss:.6f}")

    print("\nComparison summary (PatchTST vs LightGBM):")
    print("  PatchTST captures long-range dependencies via patching + transformer attention.")
    print("  LightGBM is faster to train and more robust on smaller datasets.")
    print("  PatchTST may outperform on longer sequence lengths (> 24 months).")
    print("=" * 60)

    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'n_sequences': train_shape[0],
        'seq_len': seq_len,
        'pred_len': pred_len,
    }
