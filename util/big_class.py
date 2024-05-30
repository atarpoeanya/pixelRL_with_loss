import smallFunc
import cv2
import numpy as np
import math
from eval import even_more_psnr, cal_entropy, mse, ssim

# D:\Tugas_Kuliah\Tugas_Akhir\DRL\pixLOSS\
# np.random.seed(1)
# imag = np.random.randint(0, 256, size=(4, 4, 3), dtype=np.uint8)
imag = cv2.imread("test_image/new_image.png")
# imag = cv2.imread("test_image/4_output.jpg")
img = cv2.resize(imag, (0, 0), None, .5, .5)
# print(smallFunc.matsum(image=img) == smallFunc.matsum_copy(image=img))
# numpy_horizontal = np.hstack((img, smallFunc.matsum_see(image=img)))

eval_per_step = {}
# worse_con = smallFunc.lower_contrast(image=img)
# print(even_more_psnr(worse_con, img))
# # enhancement via dark stretching
# eval_per_step["ds"] = even_more_psnr(smallFunc.stretching(worse_con), img)

# # Enhacnement via clahe on HSV colorpsace
# eval_per_step["cl_hsv"] = even_more_psnr(smallFunc.clahe_hsv(worse_con), img)

# # Enhacnement via clahe on LAB colorpsace
# eval_per_step["cl_lab"] = even_more_psnr(smallFunc.clahe_lab(worse_con), img)

# # Enhancement via umf
# eval_per_step["umf"] = even_more_psnr(smallFunc.umf(worse_con), img)

# for i,(key,val) in enumerate(eval_per_step.items()):
#    if i == 0:
#       print("PSNR values comparison on individual step")
#    print(f"{key}: {val}")

# print("end\n")


# worse_con = smallFunc.lower_contrast(image=img, contrast_factor=0.5 , b=7)

# numpy_horizontal_concat = np.concatenate((img, worse_con,smallFunc.umf(smallFunc.clahe_hsv(smallFunc.stretching(worse_con)))), axis=1)

# cv2.imshow('image window', numpy_horizontal_concat)
# # add wait key. window waits until user presses a key
# cv2.waitKey(0)
# # and finally destroy/close all open windows
# cv2.destroyAllWindows()


GAMMA = 0
MULTI = 1
CLIP = 0.7
SIGMA = 0.4
eval_sequence = {}
res = np.ndarray(img.shape, img.dtype)
def seq_enhc(image, isWorsen=False):
   input = smallFunc.lower_contrast(image, 0.7, 0.3) if isWorsen else image
   # enhancement via dark stretching
   eval_sequence["ds"] = even_more_psnr(smallFunc.stretching(input, gamma=GAMMA, multi = MULTI), img)
   ds = smallFunc.stretching(input, gamma=GAMMA, multi = MULTI)

   # Enhacnement via clahe on HSV colorpsace
   eval_sequence["cl_hsv"] = even_more_psnr(smallFunc.clahe_hsv(ds,clip=CLIP, tileSize=(8,8)), img)
   cl_hsv = smallFunc.clahe_hsv(ds,clip=CLIP, tileSize=(8,8))

   # Enhacnement via clahe on LAB colorpsace
   eval_sequence["cl_lab"] = even_more_psnr(smallFunc.clahe_lab(ds,clip=CLIP, tileSize=(8,8)), img)
   cl_lab = smallFunc.clahe_lab(ds,clip=CLIP, tileSize=(8,8))

   # Enhancement via umf
   eval_sequence["umf_hsv"] = even_more_psnr(smallFunc.umf(cl_hsv, SIGMA= SIGMA), img)
   umf_hsv = smallFunc.umf(cl_hsv, SIGMA= SIGMA)
   eval_sequence["umf_lab"] = even_more_psnr(smallFunc.umf(cl_lab, SIGMA= SIGMA), input)
   umf_lab = smallFunc.umf(cl_lab, SIGMA= SIGMA)
   return umf_hsv
res = seq_enhc(img, True)

if len(eval_sequence) > 0:
   for i,(key,val) in enumerate(eval_sequence.items()):
      if i == 0:
         print("PSNR values comparison on sequenced step")
      print(f"{key}: {val}")

   print("end\n")

print(even_more_psnr(img,img))

print(
   "\nMSE",
   mse(res, img),
   "\nEntropy original",
   cal_entropy(img),
   "\nEntropy enahnced",
   cal_entropy(res),
   "\nSSIM",
   ssim(img, res),
)





numpy_horizontal_concat = np.concatenate((img, res), axis=1)

cv2.imshow('image window', numpy_horizontal_concat)
# add wait key. window waits until user presses a key
cv2.waitKey(0)
# and finally destroy/close all open windows
cv2.destroyAllWindows()


