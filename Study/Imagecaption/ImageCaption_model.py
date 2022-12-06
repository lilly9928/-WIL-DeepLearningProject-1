import torch
import torch.nn as nn
import torchvision.models as models
import torchvision

class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):
        super(DeformableConv2d, self).__init__()

        self.padding = padding

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size * kernel_size,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                        1 * kernel_size * kernel_size,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        h, w = x.shape[2:]
        max_offset = max(h, w) / 4.

        offset = self.offset_conv(x).clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))

        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=self.padding,
                                          mask=modulator
                                          )
        return x

class Modify_Resnet(nn.Module):
    def __init__(self,embed_size):
        super(Modify_Resnet,self).__init__()
        self.model = models.resnet34(pretrained=True)
       # self.model.layer4.deform1= DeformableConv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
      #  self.model.layer4.deform2= DeformableConv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        #self.model.layer4.deform3 = DeformableConv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.fc = nn.Linear(512, embed_size)


    def forward(self,x):
        x = self.model(x)

        return x

class EncoderCNN(nn.Module):
    def __init__(self,embed_size,train_CNN =False):
        super(EncoderCNN,self).__init__()
        # self.train_CNN = train_CNN
        # self.inception = models.inception_v3(pretrained=True,aux_logits=False)
        # self.inception.fc = nn.Linear(self.inception.fc.in_features,embed_size)
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.5)
        self.model = Modify_Resnet(embed_size)

    def forward(self,images):
        #features = self.inception(images)

        # for name,param in self.inception.named_parameters():
        #     if "fc.weight" in name or "fc.bias" in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = self.train_CNN

        with torch.no_grad():
            img_feature = self.model(images)  # [batch_size, vgg16(19)_fc=4096]
            #   img_feature = self.fc(img_feature)                   # [batch_size, embed_size]

        l2_norm = img_feature.norm(p=2, dim=1, keepdim=True).detach()
        img_feature = img_feature.div(l2_norm)  # l2-normalized feature vector

        return img_feature

class DecoderRNN(nn.Module):
    def __init__(self,embed_size,hidden_size,vocab_size,num_layers):
        super(DecoderRNN,self).__init__()
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size,num_layers)
        self.linear = nn.Linear(hidden_size,vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self,features,captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0),embeddings),dim = 0)
        hiddens,_ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

class CNNtoRNN(nn.Module):
    def __init__(self,embed_size,hidden_size,vocab_size,num_layers):
        super(CNNtoRNN,self).__init__()
        self.encoderCNN =EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size,hidden_size,vocab_size,num_layers)

    def forward(self,image,captions):
        features = self.encoderCNN(image)
        outputs = self.decoderRNN(features,captions)
        return outputs

    def caption_images(self,image,vocabulary,max_length= 50):
        result_caption = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens,states = self.decoderRNN.lstm(x,states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)

                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] =="<EOS>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption]

