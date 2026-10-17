import torch
import torch.nn as nn
import math
from transformers import RobertaTokenizer

# =========================
# CONFIG (MUST MATCH TRAINING)
# =========================
MODEL_NAME = "microsoft/codebert-base"
VOCAB_SIZE = 50265
MAX_LEN = 512
D_MODEL = 512
N_LAYERS = 6
HEADS = 8
D_FF = 2048
DROPOUT = 0.1
N_CLASSES = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_MAP = {0: "SECURE", 1: "VULNERABLE"}

# =========================
# MODEL DEFINITIONS
# (EXACT COPY OF TRAINING)
# =========================
class InputEmbeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
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
            nn.Linear(d_ff, d_model),
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
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        return self.fc(x[:, 0])


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, max_len):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.embed = InputEmbeddings(D_MODEL, vocab_size)
        self.pos = PositionalEncoding(D_MODEL, max_len + 1, DROPOUT)

        self.encoder = Encoder(
            D_MODEL,
            [EncoderBlock(D_MODEL, HEADS, D_FF, DROPOUT) for _ in range(N_LAYERS)]
        )

        self.classifier = ClassificationHead(D_MODEL, N_CLASSES)

    def forward(self, x, mask=None):
        B = x.size(0)
        x = self.embed(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        if mask is not None:
            cls_mask = torch.ones(B, 1, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)
            mask = mask[:, None, None, :].bool()

        x = self.pos(x)
        x = self.encoder(x, mask)
        return self.classifier(x)

# =========================
# LOAD SYSTEM
# =========================
def load_system():
    print("🔄 Loading tokenizer & model...")

    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    model = TransformerClassifier(
        vocab_size=VOCAB_SIZE,
        max_len=MAX_LEN
    ).to(DEVICE)

    state = torch.load("best_model.pt", map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    print("✅ Model loaded successfully")
    return model, tokenizer


# =========================
# PREDICT
# =========================
def predict(code, model, tokenizer):
    encoding = tokenizer(
        code,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)

    return LABEL_MAP[pred.item()], conf.item()


# =========================
# CLI
# =========================
if __name__ == "__main__":
    model, tokenizer = load_system()

    print("\n🛡️  C-CODE VULNERABILITY DETECTOR")
    print("Type 'exit' to quit\n")

    while True:
        code = input("📝 Enter C Function:\n")
        if code.lower() == "exit":
            break

        label, conf = predict(code, model, tokenizer)
        print(f"\n🔍 Result: {label}")
        print(f"📊 Confidence: {conf * 100:.2f}%")
        print("-" * 50)
