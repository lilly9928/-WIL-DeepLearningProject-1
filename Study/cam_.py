
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
from torch import topk
import numpy as np
import skimage.transform


image = Image.open("C:/Users/1315/Desktop/test/dog.jpg")
plt.imshow(image)
plt.show()

# Imagenet mean/std

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

# Preprocessing - scale to 224x224 for model, convert to tensor,
# and normalize to -1..1 with mean/std for ImageNet

preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])

display_transform = transforms.Compose([
   transforms.Resize((224,224))])


tensor = preprocess(image)

prediction_var = Variable((tensor.unsqueeze(0)).cuda(), requires_grad=True)

model = models.resnet18(pretrained=True)

model.cuda()
model.eval()

class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
    def remove(self): self.hook.remove()



final_layer = model._modules.get('layer4')

activated_features = SaveFeatures(final_layer)

prediction = model(prediction_var)
pred_probabilities = F.softmax(prediction).data.squeeze()
activated_features.remove()

topk(pred_probabilities,1)

def getCAM(feature_conv, weight_fc, class_idx):
    _, nc, h, w = feature_conv.shape
    cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return [cam_img]

weight_softmax_params = list(model._modules.get('fc').parameters())
weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())


weight_softmax_params

class_idx = topk(pred_probabilities,1)[1].int()

overlay = getCAM(activated_features.features, weight_softmax, class_idx )

plt.imshow(overlay[0], alpha=0.5, cmap='jet')

plt.show()

plt.imshow(display_transform(image))
plt.imshow(skimage.transform.resize(overlay[0], tensor.shape[1:3]), alpha=0.5, cmap='jet');
plt.show()


class_idx = topk(pred_probabilities,2)[1].int()
print(class_idx)
overlay = getCAM(activated_features.features, weight_softmax, 332 )

plt.imshow(display_transform(image))
plt.imshow(skimage.transform.resize(overlay[0], tensor.shape[1:3]), alpha=0.5, cmap='jet');
plt.show()