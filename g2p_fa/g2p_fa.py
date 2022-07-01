from g2p_fa.model import (Encoder, Decoder, Seq2Seq)
from g2p_fa.hparams import hparams as hp
import torch

MODEL_PATH = ''

class G2P_Fa:
    def __init__(self, checkpoint=MODEL_PATH, hparams={}):
        for key, value in hparams.items():
            hp[key] = value

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        enc = Encoder(hp['INPUT_DIM'], hp['ENC_EMB_DIM'], hp['HID_DIM'], hp['N_LAYERS'], hp['ENC_DROPOUT'])
        dec = Decoder(hp['OUTPUT_DIM'], hp['DEC_EMB_DIM'], hp['HID_DIM'], hp['N_LAYERS'], hp['DEC_DROPOUT'])

        self.model = Seq2Seq(enc, dec, self.device).to(self.device)

        self.model.load_state_dict(torch.load(checkpoint))

    def train(self):
        pass

    def __call__(self, text):
        pass

    def save(self, PATH=""):
        torch.save(self.model.state_dict(), PATH)


