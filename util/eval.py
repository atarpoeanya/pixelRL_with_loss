import numpy as np
import math
import cv2

from skimage.metrics import structural_similarity
from scipy.stats import entropy
from PIL import Image

def img_res(img):
  with Image.open("filename") as image:
    xres, yres = image.info['dpi']

def calculate_psnr(img1, img2, max_value=255):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((img1 - img2) ** 2)
    # Calculate the maximum possible pixel value (Max)
    # For 8-bit images
    if mse == 0:
      return 100
    # Calculate PSNR using the formula
    # return 20 * math.log10((max_value ** 2) / mse)
    return 20 * math.log10((max_value)) - 10 * math.log10(mse)
def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    
    greyA, greyB = cv2.cvtColor(imageA.copy(), cv2.COLOR_BGR2GRAY), cv2.cvtColor(imageB.copy(), cv2.COLOR_BGR2GRAY)

    err = np.sum((greyA.astype("float") - greyB.astype("float")) ** 2)
    err /= float(greyA.shape[0] * greyA.shape[1])
    return err
    # return the MSE, the lower the error, the more "similar"
    # the two images are

def cal_psnr(image1, image2):
   # Own implementation
    mse = np.mean((image1.astype(np.float32) / 255 - image2.astype(np.float32) / 255) ** 2)
    return 10 * np.log10(1. / mse)

def even_more_psnr(s, r):
  
  height, width, channel = s.shape
  size = height*width

  sb,sg,sr = cv2.split(s)
  rb,rg,rr = cv2.split(r)

  mseb = ((sb-rb)**2).sum()
  mseg = ((sg-rg)**2).sum()
  mser = ((sr-rr)**2).sum()

  MSE = (mseb+mseg+mser)/(3*size)
  # print("individual MSE:", MSE, "\n")
  psnr = 10*math.log10(255**2/MSE)
  return round(psnr,2)

# PSNR = 20 * log10(max_val) - 10 * log10(MSE)

def ssim(original_image: np.ndarray,distorted_image: np.ndarray):

    # Convert the images to grayscale (optional, but often done for SSIM)
    original_image = cv2.cvtColor(original_image.copy(), cv2.COLOR_BGR2GRAY)
    distorted_image =cv2.cvtColor(distorted_image.copy(), cv2.COLOR_BGR2GRAY)

    # original_image = original_image.astype(float)
    # distorted_image = distorted_image.astype(float)

    # Calculate SSIM
    return structural_similarity(original_image, distorted_image, data_range=255)

def cal_entropy(img: np.ndarray):
  gray_image = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
  _bins = 128

  hist, _ = np.histogram(gray_image.ravel(), bins=_bins, range=(0, _bins))
  prob_dist = hist / hist.sum()
  image_entropy = entropy(prob_dist, base=2)
  return image_entropy

def calculate_entropy(image_path):
    # Open the image
    img = Image.open(image_path)

    # Convert the image to grayscale
    img = img.convert('L')

    # Get the histogram of pixel intensities
    histogram = img.histogram()

    # Calculate the probability of each pixel intensity
    probabilities = [float(h) / sum(histogram) for h in histogram]

    # Calculate entropy
    entropy = -sum(p * math.log2(p) for p in probabilities if p != 0)

    return round(entropy, 2)

def c_entropy(im):
    # Compute normalized histogram -> p(g)
    p = np.array([(im==v).sum() for v in range(256)])
    p = p/p.sum()
    # Compute e = -sum(p(g)*log2(p(g)))
    e = -(p[p>0]*np.log2(p[p>0])).sum()
    
    return e

# img1 = cv2.imread('validation_data/000015.jpg')
# img2 = cv2.imread('low_contrast_data/000015.png')
# img3 = cv2.imread('enhanced_lc/result_1_1/0_output.png')

# img4 = cv2.imread("new_test/4_output.jpg")
# img5 = cv2.imread("res_2311/re_7/4_output.png")

# path_test = "new_test/5_output.jpg"
# path_result = "res_2311/re_7/5_output.png"

# print("Test Data")
# print( f"LC Entropy: {calculate_entropy(path_test)}")
# print( f"Enhanced Entropy: {calculate_entropy(path_result)}")

# print( f"PSNR: {even_more_psnr(img4, img5)}")
# print("SSIM :",ssim(img4,img5))


# print("with CLAHE")
# print( f"PSNR: {even_more_psnr(img1, img2)}")
# # print( f"Entropy: {cal_entropy(img3)}")
# # print( f"Entropy: {calculate_entropy('enhanced_lc/result_1_6/0_output.png')}")
# print( f"Original Entropy: {c_entropy(img1)}")
# print( f"LC Entropy: {c_entropy(img2)}")
# print( f"Enhanced Entropy: {c_entropy(img3)}")
# # print("PSNR cv2 :", cv2.PSNR(img1,img3))
# print("PSNR mse :", mse(img1, img2))
# print("SSIM :",ssim(img1,img2))