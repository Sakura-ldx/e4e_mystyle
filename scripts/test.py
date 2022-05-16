import cv2
import dlib
import numpy as np
import inference
import cv2

'''image = inference.run_alignment('./vSYYZ139_mr1584119663256.jpg')
image.save('test.jpg')
print('end')'''

import torch
ckpt = torch.load('/home/ssd/priv/workspace/e4e/scripts/pretrained_models/e4e_ffhq_encode.pt')
print('end')