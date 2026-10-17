import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        # (B, S) → (B, S, d_model)
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)



class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, dropout):
        super().__init__()
        assert d_model % heads == 0

        self.d_k = d_model // heads
        self.h = heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, S, _ = x.shape

        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        Q = Q.view(B, S, self.h, self.d_k).transpose(1, 2)
        K = K.view(B, S, self.h, self.d_k).transpose(1, 2)
        V = V.view(B, S, self.h, self.d_k).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = attn @ V
        out = out.transpose(1, 2).contiguous().view(B, S, -1)

        return self.w_o(out)


class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.norm = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))



class EncoderBlock(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, heads, dropout)
        self.ffn = FeedForwardBlock(d_model, d_ff, dropout)
        self.residuals = nn.ModuleList([
            ResidualConnection(d_model, dropout),
            ResidualConnection(d_model, dropout)
        ])

    def forward(self, x, mask=None):
        x = self.residuals[0](x, lambda x: self.attn(x, mask))
        x = self.residuals[1](x, self.ffn)
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = LayerNormalization(d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class ClassificationHead(nn.Module):
    def __init__(self, d_model, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        return self.fc(x[:, 0])



class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len,
        d_model=512,
        N=6,
        heads=8,
        d_ff=2048,
        dropout=0.1
    ):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.embed = InputEmbeddings(d_model, vocab_size)
        self.pos = PositionalEncoding(d_model, max_len + 1, dropout)

        self.encoder = Encoder(
            d_model,
            [EncoderBlock(d_model, heads, d_ff, dropout) for _ in range(N)]
        )

        self.classifier = ClassificationHead(d_model)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask=None):
        B = x.size(0)

        # Embed tokens
        x = self.embed(x)

        # Add CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Fix mask: (B, S) -> (B, S+1)
        if mask is not None:
            cls_mask = torch.ones(B, 1, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)

            # Expand for attention: (B, 1, 1, S+1)
            mask = mask[:, None, None, :].bool()

        x = self.pos(x)
        x = self.encoder(x, mask)

        return self.classifier(x)



#  tokenizer

from transformers import RobertaTokenizer
import pandas as pd
from torch.utils.data import Dataset, DataLoader


tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

class CodeXGLUEDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=512):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_map = {"SECURE": 0, "VULNERABLE": 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        encoding = self.tokenizer(
            row["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.label_map[row["label"]])
        }




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

train_dataset = CodeXGLUEDataset("real_world_vuln_data.csv", tokenizer)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

model = TransformerClassifier(vocab_size=50265, max_len=512).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

model.train()


from tqdm import tqdm

best_loss = float("inf")

for epoch in range(3):
    model.train()
    total_loss = 0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}",
        leave=True
    )

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)


        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        # 🔥 LIVE loss update
        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    avg_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch+1} completed | Avg Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "best_model.pt")
        print("✅ Best model saved!")


from sklearn.metrics import accuracy_score, f1_score

def evaluate(model, dataloader):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch["label"].to(device)

            logits = model(input_ids, attention_mask)

            pred = torch.argmax(logits, dim=1)

            preds.extend(pred.cpu().numpy())
            labels.extend(y.cpu().numpy())

    print("Accuracy:", accuracy_score(labels, preds))
    print("F1 Score:", f1_score(labels, preds))
