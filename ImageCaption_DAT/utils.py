import torch
import torchvision.transforms as transforms
from PIL import Image


def print_examples(model, start_token,device, dataset):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    model.eval()
    test_img1 = transform(Image.open("D:/data/vqa/coco/simple_vqa/test_image/1.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 1 CORRECT: a bunch of trays that have different food")
    print(
        "Example 1 OUTPUT: "
        + " ".join(model.caption_images(test_img1.to(device), start_token.to(device),dataset.vocab,device))
    )
    test_img2 = transform(
        Image.open("D:/data/vqa/coco/simple_vqa/test_image/2.jpg").convert("RGB")
    ).unsqueeze(0)
    print("Example 2 CORRECT: a giraffe eating food from the top of the tree")
    print(
        "Example 2 OUTPUT: "
        + " ".join(model.caption_images(test_img1.to(device), start_token.to(device),dataset.vocab,device))
    )
    test_img3 = transform(Image.open("D:/data/vqa/coco/simple_vqa/test_image/3.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 3 CORRECT: a flower vase is sitting on a porch stand")
    print(
        "Example 3 OUTPUT: "
        + " ".join(model.caption_images(test_img1.to(device), start_token.to(device),dataset.vocab,device))
    )
    test_img4 = transform(
        Image.open("D:/data/vqa/coco/simple_vqa/test_image/4.jpg").convert("RGB")
    ).unsqueeze(0)
    print("Example 4 CORRECT:a zebra grazing on lush green grass in a field")
    print(
        "Example 4 OUTPUT: "
        + " ".join(model.caption_images(test_img1.to(device), start_token.to(device),dataset.vocab,device))
    )
    test_img5 = transform(
        Image.open("D:/data/vqa/coco/simple_vqa/test_image/5.jpg").convert("RGB")
    ).unsqueeze(0)
    print("Example 5 CORRECT: woman in swim suit holding parasol on sunny day")
    print(
        "Example 5 OUTPUT: "
        + " ".join(model.caption_images(test_img1.to(device), start_token.to(device),dataset.vocab,device))
    )
    model.train()


def save_checkpoint(state, filename="my_checkpoint.pth"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step