import numpy as np
import cv2

# cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
# ksize - size of gabor filter (n, n)
# sigma - standard deviation of the gaussian function
# theta - orientation of the normal to the parallel stripes
# lambda - wavelength of the sunusoidal factor
# gamma - spatial aspect ratio
# psi - phase offset
# ktype - type and range of values that each pixel in the gabor kernel can hold

ksize = 4
sigma = 10
theta = 135
lamda = 5
gamma = 4

kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

for num in range(1000):
    # 讀取資料
    img = cv2.imread("read path" + str(num+1) + '.tif')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 生成Gabor影像
    g_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
    filtered_img = cv2.filter2D(gray, cv2.CV_8UC3, g_kernel)
    save_path = "save path"
    cv2.imwrite(save_path + str(num+1) + '.tif', filtered_img)
    print("save Gabor Image  {}.tif ".format(str(num+1)))
    # 生成Prewitt影像
    img_gaussian = cv2.GaussianBlur(gray, (3, 3), 0)
    img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
    save_path_x = "save path"
    cv2.imwrite(save_path_x + str(num+1) + '.tif', img_prewittx)
    print("save Prewitt_X Image  {}.tif ".format(str(num+1)))

