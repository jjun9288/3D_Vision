import glob
import cv2
import rawpy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#Circle detection
#Finding chrome ball in image. Once we detect the chrome ball, we can get the radius and center point of the chrome ball.

raw = rawpy.imread('/home/vclab/Desktop/Photometric_Stereo/data/circle_fit_2/IMG_0996.CR3')
img = raw.postprocess()
#img = cv2.imread('/home/vclab/Desktop/Photometric_Stereo/data/circle_fit_2/IMG_0996.JPG')
circle_detected = img.copy()
h, w = img.shape[0], img.shape[1]
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray = cv2.medianBlur(gray, 5)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=80, param2=41, minRadius=1610, maxRadius=1640)
circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    cv2.circle(circle_detected, (i[0], i[1]), i[2], (0,255,0), 2)
    cv2.circle(circle_detected, (i[0], i[1]), 2, (0,255,0), 100)
image = Image.fromarray(circle_detected)
image.show()
cv2.imwrite('/home/vclab/Desktop/circle.png', circle_detected)
c = circles[0,0,:2]
cam_rad = circles[0,0,2]
world_rad = 125

#Point light detection
#Detect the brightest point on the image and consider that point as point light source.

images = glob.glob('/home/vclab/Desktop/Photometric_Stereo/data/lights_2/*.JPG')
for i in range(len(images)):
    #raw = rawpy.imread(images[i])
    #img = raw.postprocess()
    img = cv2.imread(images[i])
    point_detected = img.copy()
    #max_points = np.argwhere(raw.raw_image == np.max(raw.raw_image))
    max_points = np.argwhere(img == np.max(img))
    #for i in max_points:
        #cv2.circle(point_detected, (i[1], i[0]), 1, (0,255,0), -1)
    p_y = max_points[(max_points.shape[0]-1)//2][0]
    p_x = max_points[(max_points.shape[0]-1)//2][1]
    
    #if i == 2:
        #p_x += 130
    
    cv2.circle(point_detected, (p_x, p_y), 5, (0,255,0), 70)
    cv2.imwrite('/home/vclab/Desktop/point' + str(i+1) + '.png', point_detected)
    image = Image.fromarray(point_detected)
    image.show()     
    
    #Light direction calibration
    #Light direction calibration using the coordinate of the center of the chrome ball and the coordinate of the point light source
    
    r = np.array([0,0,1])
    n = np.array([p_x-c[0], p_y-c[1], np.sqrt(cam_rad**2-((p_x-c[0])**2-(p_y-c[1])**2))])
    norm = np.linalg.norm(n)
    n /= norm
    l = np.sum(n*r, axis=-1)
    l = (2*l*n) - r
    print("Light direction for image {0} : ".format(i), l)

light_direction_1 = np.array([[-0.64520098, 0.0773786, 0.76008437],
                            [0.41889658, -0.23207567, 0.87787615],
                            [0.01928419, -0.14583669, 0.98912071],
                            [0.70799374, 0.09938713, 0.69919029]])

light_direction_2 = np.array([[0.45050877, -0.31093049, 0.83687758],
                              [-0.58518786, -0.04625991, 0.80957717],
                              [0.72806716, -0.01563334, 0.68532752],
                              [0.07169012, -0.21268068, 0.97448831]])

light_position = light_direction_2

ball_center = np.array([0, 0, 0])


plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(ball_center[0], ball_center[1], ball_center[2], c= 'r', label = 'chrome ball')
ax.scatter(light_position[:,0], light_position[:,1], light_position[:,2], c = 'b', label = 'Point light')
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(-1,1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()
