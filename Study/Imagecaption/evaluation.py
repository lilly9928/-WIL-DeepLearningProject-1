import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from utils import *
from nltk.translate.bleu_score import corpus_bleu
from ImageCaption_get_loader import get_loader
import torch.nn.functional as F
from tqdm import tqdm
from ImageCaption_model import CNNtoRNN
from captionData import CaptionDataset
from PIL import Image


# Parameters
data_folder = 'D:/data/vqa/coco/simple_vqa/cococaption'  # folder with data files saved by create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
checkpoint = 'D:\GitHub\-WIL-Expression-Recognition-Study\Study\Imagecaption\my_checkpoint_coco_30.pth.tar'  # model checkpoint
checkpoint_pth = 'D:\GitHub\-WIL-Expression-Recognition-Study\Study\Imagecaption\my_checkpoint_coco_30.pth'
#word_map_file = '/media/ssd/caption data/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'  # word map, ensure it's the same the data was encoded with and the model was trained with
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead


transform = transforms.Compose(
    [
        transforms.Resize((356, 356)),
        transforms.RandomCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

train_loader, dataset = get_loader(
    root_folder="D:/data/vqa/coco/simple_vqa/Images/train2014/",
    annotation_file="D:/data/vqa/coco/simple_vqa/captions.txt",
    transform=transform,
    num_workers=2
)
# 하이퍼파라미터
embed_size = 256
hidden_size = 256
vocab_size = len(dataset.vocab)
num_layers = 2
learning_rate = 3e-4
num_epochs = 30

model = CNNtoRNN(embed_size,hidden_size,vocab_size,num_layers).to(device)
model.load_state_dict(torch.load(checkpoint_pth)['state_dict'])
model.eval()


def evaluate():
    """
    Evaluation
    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """

    loader, dataset = get_loader(
        root_folder="D:/data/vqa/coco/simple_vqa/Images/train2014/",
        annotation_file="D:/data/vqa/coco/simple_vqa/captions.txt",
        transform=transforms.Compose([transform]),
        num_workers=2,
        batch_size=1
    )

    wordmap = dataset.vocab

    allcaption =dict()
    idx = list()
    for i in range(1,len(dataset.imgs)):
        name = dataset.imgs[i-1]
        if name == dataset.imgs[i]:
            idx.append(i)
        else:
            allcaption[name] = idx
            idx = list()
            name = dataset.imgs[i]

    references = list()
    hypotheses = list()

    # For each image
    for _,img_name in enumerate(tqdm(allcaption)):

        image = transform(
            Image.open(f"D:/data/vqa/coco/simple_vqa/Images/train2014/{img_name}").convert(
                "RGB")).unsqueeze(0)
        captions=[]
        for idx in allcaption[img_name]:
            captions.append(loader.dataset.captions[idx])

        # Move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)
        #caps = captions.to(device)

        # Encode
        outputs = model.caption_images(image,wordmap)

        # References
        img_captions = list(map(lambda ref: ref.split(), captions)) # remove <start> and pads
        references.append(img_captions)

        # Hypotheses
        hypotheses.append([w for w in outputs if w not in {'<SOS>', '<EOS>', '<PAD>','<UNK>'}])

        total_bleu4=0
    #Calculate BLEU-4 scores
        total_bleu4 += corpus_bleu(references, list(hypotheses.split()),weights=(0, 0, 0, 1))

    bleu4=total_bleu4%len(allcaption)

   # bleu4 = corpus_bleu.sentence_bleu(list(map(lambda ref: ref.split(), references)), list(outputs.split()), weights=(0, 0, 0, 1))

    return bleu4


if __name__ == '__main__':
    print("\nBLEU-4 score @ is %.4f." % (evaluate()))
