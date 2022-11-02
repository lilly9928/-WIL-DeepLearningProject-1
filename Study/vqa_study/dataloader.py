import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import text_helper
import spacy  # for tokenizer
import string

input_dir = 'D:/data/vqa/coco/simple_vqa'
input_vqa = 'train.npy'
qst_vocab = text_helper.VocabDict(input_dir+'/vocab_questions.txt')
ans_vocab = text_helper.VocabDict(input_dir+'/vocab_answers.txt')
max_qst_length = 30
max_num_ans=10

spacy_eng = spacy.load("en_core_web_sm")

itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
freq_threshold = 5
frequencies = {}
idx = 4

vqa = np.load(input_dir + '/' + input_vqa, allow_pickle=True)
#{'image_name': 'COCO_train2014_000000458752',
# 'image_path': 'D:\\data\\vqa\\coco\\simple_vqa\\Resized_Images\\train2014\\COCO_train2014_000000458752.jpg',
# 'question_id': 458752001, 'question_str': 'What position is this man playing?',
# 'question_tokens': ['what', 'position', 'is', 'this', 'man', 'playing', '?'],
# 'all_answers': ['pitcher', 'catcher', 'pitcher', 'pitcher', 'pitcher', 'pitcher', 'pitcher', 'pitcher', 'pitcher', 'pitcher'],
# 'valid_answers': ['pitcher', 'catcher', 'pitcher', 'pitcher', 'pitcher', 'pitcher', 'pitcher', 'pitcher', 'pitcher', 'pitcher']}

# def tokenizer_eng(text):
#     return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]
#
# def numericalize(self, text):
#     tokenized_text = self.tokenizer_eng(text)
#     return [
#         self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
#         for token in tokenized_text
#     ]
#
# print(vqa[2])
# image = vqa[1]['image_path']
# image = Image.open(image).convert('RGB')
#
# caption =np.array(vqa[1]['caption']).tolist()
#
# print(caption)
#
# import pandas as pd
# df = pd.read_csv(input_dir + '/captions.txt')
# captionn = df["captions"]
#
#
# itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
# stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
# freq_threshold = 3
#
# max_length = 0
# for sentence in captionn:
#     if len(sentence)>max_length:
#         max_length=len(sentence)
#     for word in tokenizer_eng(sentence):
#         if word not in frequencies:
#             frequencies[word] = 1
#
#         else:
#             frequencies[word] += 1
#
#         if frequencies[word] == freq_threshold:
#             stoi[word] = idx
#             itos[idx] = word
#             idx += 1
#
#
#
# # for i in range(1,len(vqa[1]['question_tokens'][1])):
# for w in vqa[1]['caption_tokens'][1]:
#     print(w)
#
# # print(stoi)
# # print(itos)
# print(vqa[1]['question_tokens'])
# print(max_length)

# numericalized_caption = [stoi["<SOS>"]]
# numericalized_caption += numericalize(caption)
# numericalized_caption.append(stoi["<EOS>"])
#
# print(numericalized_caption)


# cap_vocab = text_helper.VocabDict(input_dir+'/vocab_caption.txt')

# cap2idc = np.array([cap_vocab.word2idx('<pad>')] * 100)
#cap2id=[cap_vocab.word2idx(w) for w in vqa[1]['caption_tokens']]

# cap2idc[:len(vqa[1]['caption_tokens'])] = [cap_vocab.word2idx(w) for w in vqa[1]['caption_tokens']]


# qst_vocab.word2idx('<pad>') == 0
# qst2idc = np.array([qst_vocab.word2idx('<pad>')] * max_qst_length)


# qst2idc[:len(vqa[1]['question_tokens'])] = [qst_vocab.word2idx(w) for w in vqa[1]['question_tokens']]
# sample = {'image': image, 'question': qst2idc}
# ans2idc = [ans_vocab.word2idx(w) for w in vqa[1]['valid_answers']]
# ans2idx = np.random.choice(ans2idc)
# sample['answer_label'] = ans2idx
# mul2idc = list([-1] * max_num_ans)
# mul2idc[:len(ans2idc)] = ans2idc
# sample['answer_multi_choice'] = mul2idc

# Googletrans API 부르기
from googletrans import Translator

translator = Translator()

def trans(text):#번역기
    try:
        if text == '<pad>' or text == '<unk>' :
            return text
        elif not text.encode().isalpha():#한글=>영어
            return text
        elif text.encode().isalpha():#영어=>한글
            return translator.translate(text, dest='ko').text

        else:
            return text

    except:
        print('error')
# 번역할 파일 열기
f = open('D:/data/vqa/coco/simple_vqa' + '/vocab_answers.txt', 'r')

contents = f.read()
#print(contents)         # contents 내용 출력
result = []

words = contents.split()
#print(words)

for w in words:
    result.append(trans(w))
print(result)      # result 내용 출력

# # 번역본 파일 저장하기
with open('D:/data/vqa/coco/simple_vqa' + '/kovocab_answers.txt', 'w') as f:
    for w in result:
        f.write(w+'\n')

