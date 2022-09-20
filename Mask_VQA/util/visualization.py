import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import random
from .text_helper import VocabDict

def print_examples(model,data_path,dataset,max_qst_length = 30):
    input_dir = 'D:/data/vqa/coco/simple_vqa'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    qst_vocab = VocabDict(input_dir + '/vocab_questions.txt')

    transform = transforms.Compose(
        [
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406),
                              (0.229, 0.224, 0.225))
        ]
    )
    model.eval()

    for i in range(5):
        testdata = np.load(data_path, allow_pickle=True)
        num = random.randint(0, len(testdata))

        image_path = testdata[num]['image_path']
        image = testdata[num]['image_path']
        image = Image.open(image).convert('RGB')
        image = transform(image)
        question = testdata[num]['question_str']

        qst2idc = np.array([qst_vocab.word2idx('<pad>')] * max_qst_length)  # padded with '<pad>' in 'ans_vocab'
        qst2idc[:len(testdata[num]['question_tokens'])] = [qst_vocab.word2idx(w) for w in testdata[num]['question_tokens']]
        qst2idc=torch.Tensor(qst2idc)
        print(image_path)
        print(question)
        print("Example ", i ,"CORRECT:")
        print(
            "Example", i ,"OUTPUT: "
            + " ".join(model.visualization_vqa(image.to(device).float(),qst2idc.to(device).long(),dataset.ans_vocab))
        )



    # model.eval()
    # test_img1 = transform(Image.open("C:/Users/1315/Desktop/vqadata/flickr8k/test_examples/dog.jpg").convert("RGB")).unsqueeze(
    #     0
    # )
    # print("Example 1 CORRECT: Dog on a beach by the ocean")
    # print(
    #     "Example 1 OUTPUT: "
    #     + " ".join(model.caption_images(test_img1.to(device), dataset.vocab))
    # )
    # test_img2 = transform(
    #     Image.open("C:/Users/1315/Desktop/vqadata/flickr8k/test_examples/child.jpg").convert("RGB")
    # ).unsqueeze(0)
    # print("Example 2 CORRECT: Child holding red frisbee outdoors")
    # print(
    #     "Example 2 OUTPUT: "
    #     + " ".join(model.caption_images(test_img2.to(device), dataset.vocab))
    # )
    # test_img3 = transform(Image.open("C:/Users/1315/Desktop/vqadata/flickr8k/test_examples/bus.png").convert("RGB")).unsqueeze(
    #     0
    # )
    # print("Example 3 CORRECT: Bus driving by parked cars")
    # print(
    #     "Example 3 OUTPUT: "
    #     + " ".join(model.caption_images(test_img3.to(device), dataset.vocab))
    # )
    # test_img4 = transform(
    #     Image.open("C:/Users/1315/Desktop/vqadata/flickr8k/test_examples/boat.png").convert("RGB")
    # ).unsqueeze(0)
    # print("Example 4 CORRECT: A small boat in the ocean")
    # print(
    #     "Example 4 OUTPUT: "
    #     + " ".join(model.caption_images(test_img4.to(device), dataset.vocab))
    # )
    # test_img5 = transform(
    #     Image.open("C:/Users/1315/Desktop/vqadata/flickr8k/test_examples/horse.png").convert("RGB")
    # ).unsqueeze(0)
    # print("Example 5 CORRECT: A cowboy riding a horse in the desert")
    # print(
    #     "Example 5 OUTPUT: "
    #     + " ".join(model.caption_images(test_img5.to(device), dataset.vocab))
    # )
    # model.train()


#
# if __name__ == "__main__":
#     test_path = 'D:/data/vqa/coco/simple_vqa/test.npy'
#     input_dir = 'D:/data/vqa/coco/simple_vqa'
#     ans_vocab = text_helper.VocabDict(input_dir + '/vocab_answers.txt')
#
#     print_examples(VqaModel,test_path,ans_vocab)