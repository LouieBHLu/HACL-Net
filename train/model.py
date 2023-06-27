"""
Model definition of HACL-Net
"""

import torch.nn as nn
import torch
from torchvision.models import resnet18
import os
import segmentation_models_pytorch as smp
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


class HACL_Net(nn.Module):

    def __init__(self):
        super(HACL_Net, self).__init__()

        tmp_model = resnet18(pretrained=True, progress=True)
        self.feature_net = nn.Sequential(tmp_model.conv1, tmp_model.bn1, tmp_model.relu)
        self.second_feature = nn.Sequential(tmp_model.maxpool, tmp_model.layer1, tmp_model.layer2, tmp_model.layer3, tmp_model.layer4, tmp_model.avgpool)
        
        ct = 0
        for child in self.feature_net.children():
            ct += 1
            if ct < 5:
                for param in child.parameters():
                    param.requires_grad = False

        ct = 0
        for child in self.second_feature.children():
            ct += 1
            if ct < 5:
                for param in child.parameters():
                    param.requires_grad = False

        unet = smp.Unet(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
            activation="sigmoid",
            output_mask=False
        ).cuda()

        unet.load_state_dict(torch.load("/home/placenta/unet_exp/saved_models/unet_256.pth"))

        self.unet = unet
        self.unet.eval()

        self.fc6 = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(64, 2)
        )


    def forward(self, x):

        hh = torch.zeros([x.shape[0], 64, 64, 64], dtype=torch.float).to(device)
        for i in range(x.shape[0]):

            mask = self.unet(x[i].float().unsqueeze(0))

            image = nn.functional.interpolate(x[i].float().unsqueeze(0),
                                              size=(128, 128), 
                                              mode='bilinear',
                                              align_corners=False)

            image = torch.repeat_interleave(image, repeats=3, dim=1)

            image_f = self.feature_net(image)

            # Residual Learning
            hh[i] = (1.0 + mask[0]) * image_f[0]
     
        output = self.second_feature(hh)

        output = output.view(output.size()[0], -1)

        Y_pred = self.fc6(output)

        return Y_pred

