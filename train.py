import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import re
from collections import Counter

# Import your model architecture
from trasformer_refrence import build_transformer, InputEmbeddings, PositionalEncoding, Encoder, EncoderBlock, MultiHeadAttentionBlock, FeedForwardBlock

# ==========================================
# 1. CONFIGURATION (UPDATED FOR C CODE)
# ==========================================
csv_file = "real_world_vuln_data.csv" 
SEQ_LEN = 256       # Increased from 64 to 256 to fit whole C functions
BATCH_SIZE = 8     # Decreased to save memory
D_MODEL = 128       
HIDDEN = 256
LAYERS = 4        
HEADS = 4
EPOCHS = 3          
LR = 0.001

# ==========================================
# 2. TOKENIZER & DATASET
# ==========================================
class SecurityDataset(Dataset):
    def __init__(self, csv_path, vocab=None, max_len=256):
        self.data = pd.read_csv(csv_path)
        self.max_len = max_len
        
        # Remove any rows where text is missing (NaN)
        self.data = self.data.dropna(subset=['text'])
        
        if vocab is None:
            self.vocab = self.build_vocab(self.data['text'])
        else:
            self.vocab = vocab
            
        self.label_map = {'SECURE': 0, 'VULNERABLE': 1}
        self.num_classes = 2
        print(f"Classes: {self.label_map}")

    def build_vocab(self, texts):
        counter = Counter()
        for text in texts:
            # Tokenize C code: capture words and symbols like *, {, }, ;
            tokens = re.findall(r'\w+|[^\w\s]', str(text).lower())
            counter.update(tokens)
        
        # Increase vocab size for real data
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for word, _ in counter.most_common(10000): # Top 10k words
            vocab[word] = len(vocab)
        return vocab

    def tokenize(self, text):
        tokens = re.findall(r'\w+|[^\w\s]', str(text).lower())
        ids = [self.vocab.get(t, 1) for t in tokens]
        
        # Pad or Truncate
        if len(ids) < self.max_len:
            ids = ids + [0] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
            
        return torch.tensor(ids)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text_tensor = self.tokenize(row['text'])
        label_id = self.label_map[row['label']]
        return text_tensor, torch.tensor(label_id)

# ==========================================
# 3. THE CLASSIFIER MODEL (Same as before)
# ==========================================
class VulnerabilityClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, n_classes, max_len):
        super().__init__()
        self.embedding = InputEmbeddings(d_model, vocab_size)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout=0.1)
        
        encoder_blocks = []
        for _ in range(LAYERS):
            attn = MultiHeadAttentionBlock(d_model, HEADS, dropout=0.1)
            ff = FeedForwardBlock(d_model, HIDDEN, dropout=0.1)
            encoder_blocks.append(EncoderBlock(attn, ff, dropout=0.1))
            
        self.encoder = Encoder(nn.ModuleList(encoder_blocks))
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, n_classes)
        )

    def forward(self, x):
        mask = (x != 0).unsqueeze(1).unsqueeze(2) 
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.encoder(x, mask)
        x = x.mean(dim=1) 
        logits = self.classifier(x)
        return logits

# ==========================================
# 4. TRAINING LOOP
# ==========================================
def train():
    print("Loading Dataset (This might take a moment)...")
    dataset = SecurityDataset(csv_file, max_len=SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device} | Vocab Size: {len(dataset.vocab)}")
    
    model = VulnerabilityClassifier(
        vocab_size=len(dataset.vocab),
        d_model=D_MODEL,
        n_classes=dataset.num_classes,
        max_len=SEQ_LEN
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print("\n--- Starting Training ---")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

        print(f"✅ Epoch {epoch+1} Complete | Avg Loss: {total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "vuln_classifier_c.pth")
    print("\nModel saved to vuln_classifier_c.pth")

if __name__ == "__main__":
    train()