import torch
import os
import cv2
import rawpy
import torchvision
from PIL import Image
from torchvision.ops import nms
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F

model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights(MaskRCNN_ResNet50_FPN_Weights.DEFAULT))

dir = '/home/vclab/Desktop/Photometric_Stereo/data/obj/mask/IMG_0972.JPG'
#img_path = sorted(os.listdir(dir))

#img = cv2.imread(dir)
#raw = rawpy.imread(dir)

for idx, img in enumerate(dir):
    #img_path = os.path.join(dir, img)
    image = cv2.imread(dir)
    image_tensor = F.to_tensor(image)
    input_tensor = torch.unsqueeze(image_tensor, 0)
    
    model.eval()
    with torch.no_grad():
        predictions = model(input_tensor)
        predicts = predictions[0]['masks']
        for i, mask in enumerate(predicts):
            mask = mask[0]
            mask = mask >= 0.5
            mask = mask.to(dtype=torch.uint8) * 255
            mask_img = Image.fromarray(mask.cpu().detach().numpy())
            
            mask_img.save("/home/vclab/Desktop/Photometric_Stereo/data/obj/mask/{:03d}_{}.png".format(idx, i))