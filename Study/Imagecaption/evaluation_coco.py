import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torchvision.datasets as dset
from ImageCaption_get_loader import get_loader
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import torch
from ImageCaption_model import CNNtoRNN
import json


dataDir='..'
dataType='train2014'
annFile='D:/data/vqa/coco/simple_vqa/Annotations/annotations/instances_train2014.json'.format(dataDir,dataType)
root = 'D:/data/vqa/coco/simple_vqa/Images/train2014'
checkpoint_pth = 'D:\GitHub\-WIL-Expression-Recognition-Study\Study\Imagecaption\my_checkpoint.pth'
jsonFile = 'D:\GitHub\-WIL-Expression-Recognition-Study\Study\Imagecaption\captions_train2014_cap_results.json'


cap = dset.CocoCaptions(root = root,
                        annFile = annFile,
                        transform=transforms.PILToTensor())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose(
    [
        transforms.Resize((356, 356)),
        transforms.RandomCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

loader, dataset = get_loader(
    root_folder="D:/data/vqa/coco/simple_vqa/Images/train2014/",
    annotation_file="D:/data/vqa/coco/simple_vqa/captions.txt",
    transform=transforms.Compose([transform]),
    num_workers=2,
    batch_size=1
)

wordmap = dataset.vocab

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

data = []
for _,i in enumerate(cap.ids):
    img_name=cap.coco.imgs[i]['file_name']

    image = transform(
        Image.open(f"D:/data/vqa/coco/simple_vqa/Images/train2014/{img_name}").convert(
            "RGB")).unsqueeze(0)

    image = image.to(device)

    outputs=model.caption_images(image, wordmap)

    output_string = " ".join(w for w in outputs if w not in {'<SOS>', '<EOS>', '<PAD>','<UNK>'})

    #print(output_string)

    data.append({
        "image_id": i,
        "caption": output_string,
    })

    #print(data)

with open(jsonFile, 'w+') as outfile:
    json.dump(data, outfile)




# for i in enumerate(tqdm(range(len(cap)))):
#     img,target = cap[i]
#     print(img)
