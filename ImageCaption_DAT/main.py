import torch,gc
gc.collect()
torch.cuda.empty_cache()
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from dataloader import get_loader
from model import DAT
from utils import save_checkpoint, load_checkpoint,print_examples
def train():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224,224))

        ]
    )

    train_loader, dataset = get_loader(
        root_folder = "D:/data/vqa/flickr8k/images/",
        annotation_file="D:/data/vqa/flickr8k/captions.txt",
        transform=transform,
        num_workers = 2
    )

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = True

    #하이퍼파라미터
    vocab_size = len(dataset.vocab)
    num_heads = 4
    num_decoder_layers = 6
    patch_size = 4
    learning_rate = 3e-4
    num_epochs = 10

    img_size = 224
    expansion = 4
    dim_stem = 96
    dims = [96, 192, 384, 768]
    depths = [2, 2, 6, 2]
    heads = [3, 6, 12, 24]
    window_sizes = [7, 7, 7, 7]
    drop_rate = 0.0
    attn_drop_rate = 0.0
    drop_path_rate = 0.0
    strides = [1, 1, 1, 1]
    offset_range_factor = [1, 1, 2, 2]
    stage_spec = [['L', 'S'], ['L', 'S'], ['L', 'D', 'L', 'D', 'L', 'D'], ['L', 'D']]
    groups = [1, 1, 3, 6]
    use_pes = [False, False, False, False]
    dwc_pes = [False, False, False, False]
    sr_ratios = [8, 4, 2, 1]
    fixed_pes = [False, False, False, False]
    no_offs = [False, False, False, False]
    ns_per_pts = [4, 4, 4, 4]
    use_dwc_mlps = [False, False, False, False]
    use_conv_patches = False

    step = 0

    #initialize model , loss etc

    model = DAT(img_size,patch_size,expansion,dim_stem,dims,depths,heads,window_sizes,drop_rate,attn_drop_rate,drop_path_rate,strides,
                offset_range_factor,stage_spec,groups,use_pes,dwc_pes,sr_ratios,fixed_pes,no_offs,ns_per_pts,use_dwc_mlps,use_conv_patches,
                vocab_size,num_heads,num_decoder_layers).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    if load_model:
        step = load_checkpoint(torch.load("D:\GitHub\-WIL-Expression-Recognition-Study\Study\Imagecaption\checkpoint_coco_30.pth.tar"),model,optimizer)

    model.train()

    start_token = torch.tensor([1], dtype=torch.long)

    for epoch in range(num_epochs):

        print_examples(model,start_token,device,dataset)
        #print_ixray_examples(model,start_token,device,dataset)
        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step":step,
            }
            save_checkpoint(checkpoint)

        for idx,(imgs,captions) in enumerate(train_loader):
            imgs = imgs.to(device)
            captions = captions.permute(1,0).to(device)

            y_input = captions[:,:-1]
            y_expected = captions[:,1:]

            sequence_length = y_input.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(device)

            outputs,_,_ = model(imgs,y_input,tgt_mask)
            #outputs= model(imgs, y_input, tgt_mask)
            outputs = outputs.permute(1,2,0)
            loss = criterion(outputs,y_expected)

            step += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("epochs",epoch,"Training loss", loss.item())

        # for idx,(imgs,captions,_) in enumerate(train_loader):
        #     src = imgs.to(device)
        #     tgt = captions.permute(1,0).to(device)
        #
        #     tgt_input = tgt[:-1,:]
        #
        #     src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = model.create_mask(tgt, tgt_input)
        #
        #     logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        #
        #     optimizer.zero_grad()
        #
        #     tgt_out = tgt[1:, :]
        #     loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        #     loss.backward()
        #
        #     optimizer.step()
        #     losses += loss.item()
        #
        # print("epochs", epoch, "Training loss", loss.item())

if __name__ == "__main__":
    train()
