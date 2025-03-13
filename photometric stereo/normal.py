import cv2
import rawpy
import glob
import numpy as np
from PIL import Image
from sklearn.preprocessing import normalize

scale = 0.2

mask = cv2.imread('/home/vclab/Desktop/Photometric_Stereo/data/obj_2/mask/mask.png')
mask = cv2.resize(mask, (0,0), fx =scale, fy=scale,interpolation=cv2.INTER_NEAREST)

mask2 = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
height, width = mask2.shape
dst = np.zeros((height, width, 3), np.uint8)
for k in range(3):
    for i in range(height):
        for j in range(width):
            dst[i,j][k] = 255 - mask[i,j][k]
'''
L = np.array([[-0.64520098, 0.0773786, 0.76008437],
              [0.41889658, -0.23207567, 0.87787615],
              [0.01928419, -0.14583669, 0.98912071],
              [0.70799374, 0.09938713, 0.69919029]])
'''
L = np.array([[0.45050877, -0.31093049, 0.83687758],
              [-0.58518786, -0.04625991, 0.80957717],
              [0.72806716, -0.01563334, 0.68532752],
              [0.07169012, -0.21268068, 0.97448831]])


I = []
images = glob.glob('/home/vclab/Desktop/Photometric_Stereo/data/obj_2/*.CR3')
for i in range(len(images)):
    #raw = rawpy.imread('/home/vclab/Desktop/Photometric_Stereo/data/obj_2/' + str(i+1) + '.CR3')
    #img = raw.postprocess()
    img = np.array(Image.open('/home/vclab/Desktop/Photometric_Stereo/imgb_np' + str(i+1) + '.png'), 'f')
    img = np.array(img)
    img = cv2.resize(img, (0,0), fx =scale, fy=scale,interpolation=cv2.INTER_NEAREST)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = img.shape
    img = img.reshape((-1,1)).squeeze(1)
    I.append(img)
I = np.array(I)

normal = np.linalg.lstsq(L, I, rcond=-1)[0].T
normal = normalize(normal, axis=1)
 
N = np.reshape(normal, (h, w, 3))
N[:,:,0], N[:,:,2] = N[:,:,2], N[:,:,0].copy()
N = (N+1)/2
result = N
result [mask < 255 ]  = 1
result *= 255
result = result.astype(np.uint8)

cv2.imshow('normal map', result)
cv2.waitKey()
cv2.destroyAllWindows()

cv2.imwrite('/home/vclab/Desktop/Photometric_Stereo/result_3.png', result)