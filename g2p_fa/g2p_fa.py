from g2p_fa.model import (Encoder, Decoder, Seq2Seq)
from g2p_fa.hparams import hparams as hp
from g2p_fa.utils import (tensor2IPA, word2tensor, WordDataset)
import torch
import torch.nn as nn
import torch.optim as optim

MODEL_PATH = ''

class G2P_Fa:
    def __init__(self, checkpoint=MODEL_PATH, hparams={}):
        for key, value in hparams.items():
            hp[key] = value

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        enc = Encoder(hp['INPUT_DIM'], hp['ENC_EMB_DIM'], hp['HID_DIM'], hp['N_LAYERS'], hp['ENC_DROPOUT'])
        dec = Decoder(hp['OUTPUT_DIM'], hp['DEC_EMB_DIM'], hp['HID_DIM'], hp['N_LAYERS'], hp['DEC_DROPOUT'])

        self.model = Seq2Seq(enc, dec, self.device).to(self.device)
        if checkpoint:
            self.model.load_state_dict(torch.load(checkpoint))

    def train(self, data, epoch = 20, CLIP = 1):
        optimizer = optim.Adam(self.model.parameters())
        criterion = nn.CrossEntropyLoss()
        self.model.train()

        for i in range(epoch):
            epoch_loss = 0
            for batch in data:
                optimizer.zero_grad()

                src = batch.features
                trg = batch.labels

                output = self.model(src, trg)
                output = output[1:].view(-1, hp['OUTPUT_DIM'])
                trg = trg[1:].view(-1)

                loss = criterion(output, trg)
        
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), CLIP)

                optimizer.step()

                epoch_loss += loss.item()
            print(f'Train Loss: {epoch_loss/len(data):.3f}')
        self.model.eval()

    def __call__(self, word):
        word_vector = word2tensor(word).to(torch.device)
        output_vector = self.model(word_vector)
        output_word = tensor2IPA(output_vector)

        return output_word

    def save(self, PATH=""):
        torch.save(self.model.state_dict(), PATH)


