import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import text_helper
import spacy  # for tokenizer
import pandas as pd  # for lookup in annotation file

input_dir = 'D:/data/vqa/coco/simple_vqa'
input_vqa = 'train.npy'
qst_vocab = text_helper.VocabDict(input_dir+'/vocab_questions.txt')
ans_vocab = text_helper.VocabDict(input_dir+'/vocab_answers.txt')
max_qst_length = 30
max_num_ans=10

spacy_eng = spacy.load("en_core_web_sm")

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]



vqa = np.load(input_dir + '/' + input_vqa, allow_pickle=True)
caption = pd.read_csv(input_dir + '/captions.txt')


print(vqa[2]['caption'][1])

vocab = Vocabulary(5)
vocab.build_vocabulary(caption)

print(vocab)

numericalized_caption = [vocab.stoi["<SOS>"]]
numericalized_caption += vocab.numericalize(vqa[2]['caption'][1])
numericalized_caption.append(vocab.stoi["<EOS>"])
#
print(torch.tensor(numericalized_caption))


