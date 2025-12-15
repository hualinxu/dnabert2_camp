# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModel
from collections import Counter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= Dataset =================
class PromoterDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=81):  # Change max_length to 81
        df = pd.read_csv(csv_file)
        self.seqs = df.iloc[:, 0].tolist()
        self.labels = df.iloc[:, 1].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        label = int(self.labels[idx])
        seq_6mer = " ".join(seq[i:i+6] for i in range(len(seq)-5))
        enc = self.tokenizer(
            seq_6mer,
            padding=False,  # Remove padding
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": label,
            "raw_seq": seq
        }

# ================= Model =================
class CNNTransformerPromoter(nn.Module):
    def __init__(self, model_dir, num_labels=2, cnn_filters=512, kernel_size=3):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        self.bert = AutoModel.from_pretrained(model_dir, config=self.config, trust_remote_code=True)

        self.cnn = nn.Sequential(
            nn.Conv1d(self.config.hidden_size, cnn_filters, kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv1d(cnn_filters, cnn_filters // 2, kernel_size, padding="same"),
            nn.ReLU(),
        )

        self.attention = nn.MultiheadAttention(cnn_filters // 2, num_heads=4, batch_first=True)
        self.classifier = nn.Linear(cnn_filters // 2 + self.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        output_attentions=True, return_dict=True)
        seq_emb = out.last_hidden_state              # [B, L, H]
        attn_bert = out.attentions[-1]               # [B, heads, L, L]

        cnn_in = seq_emb.transpose(1, 2)
        cnn_feat = self.cnn(cnn_in).transpose(1, 2)  # [B, L, C]
        attn_out, _ = self.attention(cnn_feat, cnn_feat, cnn_feat)
        pooled = attn_out.mean(dim=1)

        fused = torch.cat([seq_emb[:, 0], pooled], dim=1)
        logits = self.classifier(fused)
        return logits, attn_bert, cnn_feat

# ================= Paths =================
MODEL_DIR = "./SaveModel/best_cnn_dnabert2_model"
CSV_PATH = "../Data/original_dataset/original_dataset(8720).csv"

# ================= Load =================
model = CNNTransformerPromoter(MODEL_DIR).to(DEVICE)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

dataset = PromoterDataset(CSV_PATH, tokenizer)
test_loader = DataLoader(dataset, batch_size=8, shuffle=False)

# ================= Containers =================
all_attn = []
all_attn_std = []
pos_attn, neg_attn = [], []
cnn_peaks = []
top_motif_counter = Counter()

# ================= Main Loop =================
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].numpy()
        raw_seqs = batch["raw_seq"]

        _, attn_bert, cnn_feat = model(input_ids, mask)

        # ---- Attention ----
        attn_mean_batch = attn_bert.mean(dim=(1, 2)).cpu().numpy()  # [B, L]
        attn_std_batch = attn_bert.std(dim=(1, 2)).cpu().numpy()

        for i, lab in enumerate(labels):
            all_attn.append(attn_mean_batch[i])
            all_attn_std.append(attn_std_batch[i])
            if lab == 1:
                pos_attn.append(attn_mean_batch[i])
            else:
                neg_attn.append(attn_mean_batch[i])

        # ---- CNN ----
        act = cnn_feat.abs().mean(dim=2)  # [B, L]
        threshold = torch.quantile(act, 0.9)
        for i in range(act.size(0)):
            cnn_peaks.append(act[i].max().item())
            high_pos = torch.where(act[i] >= threshold)[0]
            for p in high_pos:
                p = p.item()
                if p + 6 <= len(raw_seqs[i]):  # 6-mer length for motif
                    top_motif_counter[raw_seqs[i][p:p+6]] += 1

# ================= Attention: Mean + Stability =================

REAL_TOKEN_LEN = 81  # 81bp, thus 81 6-mer tokens

# -35 and -10 regions (6-mer aligned)
# NEG35_TOKEN = int(round(35 / 6))
# NEG10_TOKEN = int(round(10 / 6))
TSS_TOKEN = 60

NEG10_REGION = (48, 54)   # -12 ~ -7
NEG35_REGION = (23, 28)   # -37 ~ -32
mean_attn = np.mean(all_attn, axis=0)[:REAL_TOKEN_LEN]
std_attn = np.mean(all_attn_std, axis=0)[:REAL_TOKEN_LEN]

plt.figure(figsize=(8, 4))

x_real = np.arange(REAL_TOKEN_LEN)

plt.plot(x_real, mean_attn, label="Mean Attention", linewidth=2)
plt.fill_between(
    x_real,
    mean_attn - std_attn,
    mean_attn + std_attn,
    alpha=0.3,
    label="±1 std")

# --- Biological regions ---
# plt.axvline(NEG35_TOKEN, color="red", linestyle="--", linewidth=2, label="-35 region")
# plt.axvline(NEG10_TOKEN, color="green", linestyle="--", linewidth=2, label="-10 region")
#
plt.axvspan(NEG35_REGION[0], NEG35_REGION[1],color="red", alpha=0.15, label="-35 region")
plt.axvspan(NEG10_REGION[0],NEG10_REGION[1] , color="green", alpha=0.15, label="-10 region")
#

plt.xlabel("Token Position (6-mer aligned)")
plt.ylabel("Attention Weight")
plt.title("Average Attention Curve (Padding Excluded)")
plt.legend()
plt.tight_layout()
plt.savefig("attention_mean_stability_masked.png", dpi=300)
plt.close()

# ================= Positive vs Negative =================
pos_mean = np.mean(pos_attn, axis=0)
neg_mean = np.mean(neg_attn, axis=0)

plt.figure()
plt.plot(pos_mean, label="Positive", linewidth=2)
plt.plot(neg_mean, label="Negative", linewidth=2)
plt.xlabel("Token Position")
plt.ylabel("Attention")
plt.title("Attention: Positive vs Negative")
plt.legend()
plt.savefig("attention_pos_neg.png", dpi=300)
plt.close()

# ================= CNN Statistics =================
print("CNN activation peak statistics:")
print("Mean:", np.mean(cnn_peaks))
print("Std :", np.std(cnn_peaks))
print("Max :", np.max(cnn_peaks))

print("\nTop motifs from top 10% CNN activations:")
for m, c in top_motif_counter.most_common(15):
    print(m, c)
