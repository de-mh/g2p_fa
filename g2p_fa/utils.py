import torch
from torch.utils.data import Dataset

class WordDataset(Dataset):
    def __init__(self, data_file, img_dir):
        self.data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                word, IPA = line.strip().split(',')
                self.data.append((word, IPA))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

FA_LETTERS = 'ءآئابتثجحخدذرزسشصضطظعغفقلمنهوُِپچژکگی'
IPA_LETTERS = 'bdefhijklmnopqstuvxzæɒɡɾʃʒʔʰː'
PAD = 0
START = 1
END = 2

def fa_letter2tensor(letter):
    return FA_LETTERS.find(letter) + 3

def IPA_letter2tensor(letter):
    return IPA_LETTERS.find(letter) + 3

def word2tensor(word, max_len):
    word_tensor = torch.zeros(max_len, dtype=torch.long)
    word_tensor[0] = START
    for i, letter in enumerate(word):
        word_tensor[i+1] = fa_letter2tensor(letter)
    word_tensor[len(word)+1] = END
    return word_tensor

def IPA2tensor(IPA, max_len):
    IPA_tensor = torch.zeros(max_len, dtype=torch.long)
    IPA_tensor[0] = START
    for i, letter in enumerate(IPA):
        IPA_tensor[i+1] = IPA_letter2tensor(letter)
    IPA_tensor[len(IPA)+1] = END
    return IPA_tensor

def tensor2IPA(tensor):
    idx = 1
    word = ''
    while(tensor[idx].item() != END):
        word += IPA_LETTERS[tensor[idx].item()-3]
        idx += 1
    return word
    

