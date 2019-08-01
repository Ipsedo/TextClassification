import torch.nn as nn


class ConvModelReuters(nn.Module):
    def __init__(self, vocab_size, nb_class, sent_max_len, padding_idx):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, 8, padding_idx=padding_idx)

        self.seq_conv = nn.Sequential(
            nn.Conv1d(8, 12, kernel_size=3),
            nn.MaxPool1d(4, 4),
            nn.ReLU()
        )

        self.seq_lin = nn.Sequential(
            nn.Linear(2880, nb_class),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = self.emb(x)
        out = out.permute(0, 2, 1)
        out = self.seq_conv(out)
        out = out.flatten(1, -1)
        out = self.seq_lin(out)
        return out


class ConvModelWiki(nn.Module):
    def __init__(self, vocab_size, sent_max_len, padding_idx):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, 8, padding_idx=padding_idx)

        self.seq_conv = nn.Sequential(
            nn.Conv1d(8, 12, kernel_size=3),
            nn.MaxPool1d(4, 4),
            nn.ReLU()
        )

        self.seq_lin = nn.Sequential(
            nn.Linear(8124, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.emb(x)
        out = out.permute(0, 2, 1)
        out = self.seq_conv(out)
        out = out.flatten(1, -1)
        out = self.seq_lin(out)
        return out.squeeze(1)


class ConvModelDBPedia_V1(nn.Module):
    def __init__(self, vocab_size, nb_class, padding_idx):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, 128, padding_idx=padding_idx)

        self.seq_conv = nn.Sequential(
            nn.Conv1d(128, 184, kernel_size=3),
            nn.MaxPool1d(4, 4),
            nn.ReLU(),
            nn.Conv1d(184, 256, kernel_size=5, stride=2),
            nn.MaxPool1d(8, 8),
            nn.ReLU()
        )

        self.seq_lin = nn.Sequential(
            nn.Linear(256 * 7, 256 * 10),
            nn.BatchNorm1d(256 * 10),
            nn.Linear(256 * 10, nb_class),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        out = self.emb(x)
        out = out.permute(0, 2, 1)

        out = self.seq_conv(out)
        out = out.flatten(1, -1)

        out = self.seq_lin(out)

        return out

    
class ConvModelDBPedia_V1_2(nn.Module):
    def __init__(self, vocab_size, nb_class, padding_idx):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, 100, padding_idx=padding_idx)

        self.seq_conv = nn.Sequential(
            nn.Conv1d(100, 140, kernel_size=3),
            nn.MaxPool1d(2, 2),
            nn.ReLU(),
            nn.Conv1d(140, 180, kernel_size=5, stride=2),
            nn.MaxPool1d(8, 8),
            nn.ReLU()
        )

        self.seq_lin = nn.Sequential(
            nn.Linear(180 * 15, 2048),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, nb_class),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        out = self.emb(x)
        out = out.permute(0, 2, 1)

        out = self.seq_conv(out)
        out = out.flatten(1, -1)

        out = self.seq_lin(out)

        return out


class ConvModelDBPedia_V2(nn.Module):
    def __init__(self, vocab_size, nb_class, padding_idx):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, 32, padding_idx=padding_idx)

        self.seq_conv = nn.Sequential(
            nn.Conv1d(32, 48, kernel_size=3),
            nn.ReLU(),
            nn.Conv1d(48, 56, kernel_size=5, stride=2),
            nn.MaxPool1d(2, 2),
            nn.ReLU(),
            nn.Conv1d(56, 96, kernel_size=5, stride=2),
            nn.MaxPool1d(2, 2),
            nn.ReLU()
        )

        self.seq_lin = nn.Sequential(
            nn.Linear(96 * 30, nb_class),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        out = self.emb(x)
        out = out.permute(0, 2, 1)

        out = self.seq_conv(out)
        out = out.flatten(1, -1)

        out = self.seq_lin(out)

        return out
