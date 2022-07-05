from g2p_fa.model import (Encoder, Decoder, Seq2Seq)
from g2p_fa.hparams import hparams as hp
from g2p_fa.utils import (IPA2tensor, tensor2IPA, word2tensor, WordDataset, MODEL_PATH)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class G2P_Fa:
    def __init__(self, checkpoint=MODEL_PATH, hparams={}):
        for key, value in hparams.items():
            hp[key] = value

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        enc = Encoder(hp['INPUT_DIM'], hp['ENC_EMB_DIM'], hp['HID_DIM'], hp['N_LAYERS'], hp['ENC_DROPOUT'])
        dec = Decoder(hp['OUTPUT_DIM'], hp['DEC_EMB_DIM'], hp['HID_DIM'], hp['N_LAYERS'], hp['DEC_DROPOUT'])

        self.model = Seq2Seq(enc, dec, self.device).to(self.device)
        if checkpoint:
            self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))

    def train(self, data, epoch = 20, CLIP = 1):
        ds = WordDataset(data)
        train_len = int(len(ds) * 0.8)
        valid_len = len(ds) - train_len
        train_ds, valid_ds = torch.utils.data.random_split(ds, [train_len, valid_len])
        train_dataloader = DataLoader(train_ds, batch_size=1024, shuffle=True)
        valid_dataloader = DataLoader(valid_ds, batch_size=1024, shuffle=True)
        print(f'train samples: {len(train_ds)}, validation samples: {len(valid_ds)}')
        optimizer = optim.Adam(self.model.parameters())
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        print(f'initial loss: {self.evaluate(valid_dataloader, criterion)}')
        for i in range(epoch):
            self.model.train()
            epoch_loss = 0
            for batch in train_dataloader:
                optimizer.zero_grad()

                src = batch[0].to(self.device)
                src = torch.transpose(src, 1, 0)
                trg = batch[1].to(self.device)
                trg = torch.transpose(trg, 1, 0)

                output = self.model(src, trg)
                output = output[1:].view(-1, hp['OUTPUT_DIM'])
                trg = trg[1:].reshape(-1)

                loss = criterion(output, trg)
        
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), CLIP)

                optimizer.step()

                epoch_loss += loss.item()

            eval_loss = self.evaluate(valid_dataloader, criterion)
            print(f'Epoch {i+1} / {epoch}\tTrain Loss: {epoch_loss/len(train_dataloader):.3f}, Valid loss: {eval_loss:.3f}')
        self.model.eval()

    def evaluate(self, iterator, criterion):
    
        self.model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(iterator):

                src = batch[0].to(self.device)
                src = torch.transpose(src, 1, 0)
                trg = batch[1].to(self.device)
                trg = torch.transpose(trg, 1, 0)

                output = self.model(src, trg, 0) #turn off teacher forcing

                output_dim = output.shape[-1]
            
                output = output[1:].view(-1, output_dim)
                trg = trg[1:].reshape(-1)

                loss = criterion(output, trg)
            
                epoch_loss += loss.item()
        return epoch_loss / len(iterator)

    def __call__(self, word):
        word_vector = word2tensor(word).to(self.device)
        trg = IPA2tensor('b').to(self.device)
        output_vector = self.model(word_vector.unsqueeze(1), trg.unsqueeze(1), 0)
        output_word = tensor2IPA(output_vector.argmax(2))

        return output_word

    def save(self, PATH):
        torch.save(self.model.state_dict(), PATH)


