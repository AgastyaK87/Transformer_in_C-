import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
import json
from transformers import BertTokenizer

#model/training settings
VOCAB_SIZE = 30522 #bert vocab
D_MODEL = 512 #Dimension of model
N_HEADS = 8 #attention heads
D_FF = 2048 #FFN inner dim
N_LAYERS = 6 #Encoder/Decoder layers
MAX_SEQ_LEN = 256 #max seq len process
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 20

#dataset class, load file

class NL2BashDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer):
        self.tokenizer = tokenizer
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item ['prompt']
        plan = json.dumps(item['plan']) #convert plan into a json string

        #tokenize prompt and plan
        #cls/sep = special transformer model tokens
        source = self.tokenizer.encode_plus(
            f"[CLS]{prompt}[SEP]",
            max_length=MAX_SEQ_LEN,
            padding = 'max_length',
            truncation = True,
            return_tensors = 'pt'
        )
        target = self.tokenizer.encode_plus(
            f"[CLS] {plan} [SEP]",
            max_length=MAX_SEQ_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'source_ids': source['input_ids'].squeeze(),
            'source_mask': source['attention_mask'].squeeze(),
            'target_ids': target['input_ids'].squeeze()
        }


#Transformer Model Architecture
#encoder-decoder transformer with pytorch

class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, D_MODEL)

        self.transformer = nn.Transformer(
            d_model=D_MODEL,
            nhead = N_HEADS,
            num_encoder_layers = N_LAYERS,
            num_decoder_layers = N_LAYERS,
            dim_feedforward = D_FF,
            batch_first = True,
        )
        #output = vocab size
        self.fc_out = nn.Linear(D_MODEL, VOCAB_SIZE)
    def forward(self,src,tgt, src_padding_mask = None):
        #embed source and target seq
        src_embed = self.embedding(src)
        tgt_embed = self.embedding(tgt)

        #create a target mask to prevent decoder looking ahead
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        output = self.transformer(
            src_embed,
            tgt_embed,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask
        )

        # Pass the output through the final linear layer
        return self.fc_out(output)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #init tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    #dataset and dataloader
    dataset = NL2BashDataset('dataset.jsonl', tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    #init model and move it to gpu
    model = TransformerModel().to(device)

    #loss funct and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("start training")
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for batch in dataloader:
            source_ids = batch['source_ids'].to(device)
            source_mask = batch['source_mask'].to(device)
            target_ids = batch['target_ids'].to(device)

            # Prepare the decoder input (shifted right) and the expected labels
            decoder_input = target_ids[:, :-1]
            labels = target_ids[:, 1:]

            #forward pass
            optimizer.zero_grad()
            outputs = model(source_ids, decoder_input, src_padding_mask=(source_mask == 0))

            #calc loss, reshape output and labels for loss
            loss = criterion(outputs.reshape(-1, VOCAB_SIZE), labels.reshape(-1))

            # --- Backward Pass ---
            loss.backward() # Backpropagation
            optimizer.step() # Update weights

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")

    # --- 5. Save the Trained Model ---
    torch.save(model.state_dict(), 'transformer_os_llm.pth')
    print("--- Training Complete. Model saved to transformer_os_llm.pth ---")

if __name__ == '__main__':
    train()