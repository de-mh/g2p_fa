import torch
import random
import math
from g2p_fa import G2P_Fa, model
import unittest

hp = {
    'ENC_DROPOUT' : 0.5,
    'DEC_DROPOUT' : 0.5
}

g2p = G2P_Fa(checkpoint=None, hparams=hp)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_example = torch.zeros(5, 10,dtype=torch.long).to(device)
for i in range(5):
    for j in range(10):
        input_example[i][j] = math.floor(random.random() * 6)

output_example = torch.zeros(7, 10,dtype=torch.long).to(device)
for i in range(5):
    for j in range(10):
        output_example[i][j] = math.floor(random.random() * 4)

class Data:
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

data = Data(input_example, output_example)


trg = output_example[1:].view(-1)


class TestModel(unittest.TestCase):
    def test_model(self):
        g2p.train([data for i in range(60)])
        output = g2p.model(input_example, torch.zeros(output_example.size(), dtype=torch.long).to(device))
        output = output[1:].view(-1, 4).argmax(1)
        self.assertGreater(sum(trg == output).item()/len(trg), 0.7)
        