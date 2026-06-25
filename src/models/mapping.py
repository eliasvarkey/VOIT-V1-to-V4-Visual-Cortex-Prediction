import torch, torch.nn as nn, torch.nn.functional as F

class MLP_V1toV4(nn.Module):
    def __init__(self, in_dim, out_dim, h1=2048, h2=1024, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h1)
        self.ln1 = nn.LayerNorm(h1)
        self.fc2 = nn.Linear(h1, h2)
        self.ln2 = nn.LayerNorm(h2)
        self.fc3 = nn.Linear(h2, out_dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        x = self.drop(F.gelu(self.ln1(self.fc1(x))))
        x = self.drop(F.gelu(self.ln2(self.fc2(x))))
        return self.fc3(x)

class GRU_V1toV4(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=512):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 512), nn.GELU(), nn.Linear(512, out_dim))
    def forward(self, x_seq, onset_idx):
        h, _ = self.gru(x_seq)     
        outs = []
        for b in range(h.size(0)):
            outs.append(h[b, onset_idx[b]])  
        Ht = torch.stack(outs, 0)  
        return self.head(Ht)       

class FusionMixer(nn.Module):
    def __init__(self, v4_dim):
        super().__init__()
        self.mix = nn.Linear(2*v4_dim, v4_dim)
        nn.init.zeros_(self.mix.weight); nn.init.zeros_(self.mix.bias)
    def forward(self, y_mlp, y_gru_aligned):
        cat = torch.cat([y_mlp, y_gru_aligned], dim=-1)
        return self.mix(cat)
