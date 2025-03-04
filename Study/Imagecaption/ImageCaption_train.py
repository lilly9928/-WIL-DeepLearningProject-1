import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
#from torch.utils.tensorboard import SummaryWriter
from ImageCaption_utils import save_checkpoint, load_checkpoint, print_examples
from ImageCaption_get_loader import get_loader
from ImageCaption_model import CNNtoRNN

def train():
    transform = transforms.Compose(
        [
            transforms.Resize((356,356)),
            transforms.RandomCrop((299,299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ]
    )

    train_loader, dataset = get_loader(
        root_folder = "D:/data/vqa/coco/simple_vqa/Images/train2014/",
        annotation_file="D:/data/vqa/coco/simple_vqa/captions.txt",
        transform=transform,
        num_workers = 2
    )

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = True

    #하이퍼파라미터
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 2
    learning_rate = 3e-4
    num_epochs = 30

    #for tensorboard

    #writer = SummaryWriter("../runs/coco")
    step = 0

    #initialize model , loss etc
    model = CNNtoRNN(embed_size,hidden_size,vocab_size,num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    if load_model:
        step = load_checkpoint(torch.load("D:\GitHub\-WIL-Expression-Recognition-Study\Study\Imagecaption\checkpoint_coco_30.pth.tar"),model,optimizer)

    model.train()

    for epoch in range(num_epochs):

        print_examples(model, device, dataset)
        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step":step,
            }
            save_checkpoint(checkpoint)

        for idx,(imgs,captions) in enumerate(train_loader):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs,captions[:-1])
            print(outputs.shape)
            print(captions.shape)
            print(outputs.reshape(-1,outputs.shape[2]).shape)
            print(captions.reshape(-1).shape)
            loss = criterion(outputs.reshape(-1,outputs.shape[2]),captions.reshape(-1))

            #print("Training loss", loss.item())
            step += 1

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()

        print("epochs",epoch,"Training loss", loss.item())

if __name__ == "__main__":
    train()
