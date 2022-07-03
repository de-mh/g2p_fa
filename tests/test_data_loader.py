import unittest
import torch
from g2p_fa.utils import (word2tensor, tensor2IPA)

class TestDataLoader(unittest.TestCase):
    def test_word2tensor(self):
        words_tuples = [
            ('آب', [1,4,7,2,0])
        ]
        for word, tensor in words_tuples:
            self.assertTrue(torch.equal(word2tensor(word, 5), torch.tensor(tensor, dtype=torch.long)))
    
    def test_tensor2word(self):
        t2w_tuples = [
            ('bed', [1,3,5,4,2,0,0])
        ]
        for word, tensor in t2w_tuples:
            
            self.assertEqual(tensor2IPA(torch.tensor(tensor, dtype=torch.long)), word)