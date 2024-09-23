import torch
import torchvision

## reference: https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
# VGG perceptual loss
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(weights='VGG16_Weights.DEFAULT').features[:4].eval())
        blocks.append(torchvision.models.vgg16(weights='VGG16_Weights.DEFAULT').features[4:9].eval())
        blocks.append(torchvision.models.vgg16(weights='VGG16_Weights.DEFAULT').features[9:16].eval())
        blocks.append(torchvision.models.vgg16(weights='VGG16_Weights.DEFAULT').features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            height, width = input.size()[-2:]
            factor = 224 / max(height, width)
            height = int(height * factor)
            width = int(width * factor)
            input = self.transform(input, mode='bilinear', size=(height, width), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(height, width), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss
    
    
# image_path = 'finetune_ip2p/data/cat/0_00002.png'
# from PIL import Image
# import numpy as np
# image = Image.open(image_path).convert('RGB')
# image = np.array(image)
# image = torch.tensor(image).float() # [H, W, C]
# image = image.permute(2, 0, 1).unsqueeze(0) / 255.0 # [1, C, H, W]

# image_path = 'finetune_ip2p/data/cat/0_00003.png'
# target = Image.open(image_path).convert('RGB')
# target = np.array(target)
# target = torch.tensor(target).float() # [H, W, C]
# target = target.permute(2, 0, 1).unsqueeze(0) / 255.0 # [1, C, H, W]

# vgg_loss = VGGPerceptualLoss(resize=False)
# mse_loss = torch.nn.MSELoss(reduction='mean')
# loss1 = vgg_loss(image, target)
# loss2 = mse_loss(image, target)
# print(loss1)
# print(loss2)
 