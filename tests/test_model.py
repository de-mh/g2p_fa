import torch
import os
from g2p_fa import G2P_Fa
import unittest

THIS_DIR, _ = os.path.split(__file__)
DATA_PATH = os.path.join(THIS_DIR, "data_d.csv")

class TestModel(unittest.TestCase):
    def test_train(self):
        g2p = G2P_Fa(checkpoint=None)
        g2p.train(DATA_PATH)
        output = g2p('ب')
        self.assertEqual(output, 'b')

    def test_model(self):
        g2p = G2P_Fa()
        samples = [
            ('طلا', 'tʰælɒː'),
            ('تلاش', 'tʰælɒːʃ')
        ]
        for input, output in samples:
            self.assertEqual(g2p(input), output)
