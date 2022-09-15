import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import text_helper

input_dir = 'D:/data/vqa/coco/simple_vqa'
input_vqa = 'train.npy'
qst_vocab = text_helper.VocabDict(input_dir+'/vocab_questions.txt')
ans_vocab = text_helper.VocabDict(input_dir+'/vocab_answers.txt')
max_qst_length = 30
max_num_ans=10

vqa = np.load(input_dir + '/' + input_vqa, allow_pickle=True)
#{'image_name': 'COCO_train2014_000000458752',
# 'image_path': 'D:\\data\\vqa\\coco\\simple_vqa\\Resized_Images\\train2014\\COCO_train2014_000000458752.jpg',
# 'question_id': 458752001, 'question_str': 'What position is this man playing?',
# 'question_tokens': ['what', 'position', 'is', 'this', 'man', 'playing', '?'],
# 'all_answers': ['pitcher', 'catcher', 'pitcher', 'pitcher', 'pitcher', 'pitcher', 'pitcher', 'pitcher', 'pitcher', 'pitcher'],
# 'valid_answers': ['pitcher', 'catcher', 'pitcher', 'pitcher', 'pitcher', 'pitcher', 'pitcher', 'pitcher', 'pitcher', 'pitcher']}

image = vqa[1]['image_path']
image = Image.open(image).convert('RGB')
# qst_vocab.word2idx('<pad>') == 0
qst2idc = np.array([qst_vocab.word2idx('<pad>')] * max_qst_length)


qst2idc[:len(vqa[1]['question_tokens'])] = [qst_vocab.word2idx(w) for w in vqa[1]['question_tokens']]
sample = {'image': image, 'question': qst2idc}
ans2idc = [ans_vocab.word2idx(w) for w in vqa[1]['valid_answers']]
ans2idx = np.random.choice(ans2idc)
sample['answer_label'] = ans2idx
mul2idc = list([-1] * max_num_ans)
mul2idc[:len(ans2idc)] = ans2idc
sample['answer_multi_choice'] = mul2idc


print(sample)
