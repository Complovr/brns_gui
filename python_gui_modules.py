#make pseudocolored img (img),choice,fused,L,H variables as object members
#class variables: gamma clut, eclut
import cv2
import numpy as np
def genVCplus(img):
	res1=cv2.cvtColor(np.uint8(img*255),cv2.COLOR_RGB2HSV);
	ress=res1[...,1];
	resh=res1[...,0];
	ress[(resh<95)]=0;
	return cv2.cvtColor(np.stack((resh, ress,res1[...,2]),axis=2),cv2.COLOR_HSV2RGB);

def genVCminus(img):
	res1=cv2.cvtColor(np.uint8(img*255),cv2.COLOR_RGB2HSV);
	ress=res1[...,1];
	resh=res1[...,0];
	ress[(resh>95)]=0;
	return cv2.cvtColor(np.stack((resh, ress,res1[...,2]),axis=2),cv2.COLOR_HSV2RGB);

def adjust_gamma(img, gamma):
   invGamma = 1.0 / gamma
   
   #make this static to the class
   gamma_table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")
   return cv2.LUT(np.uint8(img*255), gamma_table)

def genVDimg(img,val):
	if val>1:
		val=2-val;
		return adjust_gamma(img, val)
	else:
		res_g=np.stack((cv2.cvtColor(np.uint8(img*255),cv2.COLOR_RGB2GRAY),)*3,axis=2)
		res=(255 - cv2.threshold(res_g,(1-val)*255,255,cv2.THRESH_BINARY)[1])/255
		return img+res


def genIMImg(img,choice):
	res=np.array(img,copy=True)
	res[choice!=1]=1
	return res

def genOMImg(img,choice):
	res=np.array(img,copy=True)
	res[choice!=2]=1
	return res

def genVEimg(img,scale_factor):
	res_g=np.stack((cv2.cvtColor(np.uint8(img*255),cv2.COLOR_RGB2GRAY),)*3,axis=2)
	resve_ul=4**img*((255-res_g)/255) \
			 +2*img*(255 - cv2.threshold(res_g,0.52*255,255,cv2.THRESH_BINARY)[1])/255
	resve_ll=0.8*img \
			 +0.2*img*(255 - cv2.threshold(res_g,0.95*255,255,cv2.THRESH_BINARY)[1])/255
	resve_diff=resve_ul-resve_ll
	return resve_ll+scale_factor*resve_diff