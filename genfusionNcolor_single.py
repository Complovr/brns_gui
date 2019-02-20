import pywt
import numpy as np
from  scipy.signal import wiener
import os
from scipy.io import loadmat
import matplotlib.pyplot as plt
import cv2

clut=loadmat('eclut.mat')['CLUT']
#withobjle,withobjhe
def generateFusion(L,H):
    LE=L**3.2
    HE=H**0.2
    a1,(h1,v1,d1)=pywt.dwt2(LE,'haar')
    a2,(h2,v2,d2)=pywt.dwt2(HE,'haar')
    return (np.clip(pywt.idwt2((0.6*a1+a2/2,(h1+h2,v1+v2,d1+d2)),'haar'),0,1)*255).astype(np.uint8)

def generateChc(L,H):
    L1=wiener(L,[5,5])
    H1=wiener(H,[5,5])
    
    x_ax=(L1+H1)/2
    y_ax=H1-L1
    
    y_s=0.17213*(x_ax**3)-1.399*(x_ax**2)+1.2392*x_ax-0.0027535
    y_p=0.228*(x_ax**3)-0.51622*(x_ax**2)+0.30413*x_ax+0.0053274
    y_al=-0.409873*(x_ax**4)+0.90975*(x_ax**3)-1.298*(x_ax**2)+0.81073*x_ax+0.0018109
    y_a=(y_p+y_al)/2
    y_b=(y_al+y_s)/2
    
    choice_v1=np.zeros(y_ax.shape)
    choice_v1[y_ax>y_b]=1
    choice_v1[choice_v1==0]=2
    choice_v1[y_ax<y_a]=3

    a=L1;b=H1;c=np.log(a);d=np.log(b)
    q=c/d
    choice_v1[(q<1.17)&(H<0.19)&(L<0.16)]=4
    choice_v1[(q<1.24)&(H<0.42)&(L<0.3)]=4
    choice_v1[(x_ax<0.06) & (y_ax<0.06)]=1;
    return (choice_v1-1).astype(np.uint8)

def generateFColor(imfused,clut,choice_v1):
    r,c=imfused.shape
    pc_img=np.array([[clut[imfused[i,j],:,choice_v1[i,j]] for j in range(c)] for i in range(r)])
    return cv2.bilateralFilter(np.uint8(pc_img),6,157,157)/255

def loadimgfile(fpath):
    extension=fpath.split('.')[1]
    if extension=='txt':
    	return np.loadtxt(fpath)
    elif extension=='npy':
        return np.load(fpath)
def loadLH(noobjfpath,imgfpath):
    noobj=loadimgfile(noobjfpath)
    img=loadimgfile(imgfpath)
    M=noobj[:512,:640]
    N=noobj[:512,640:1280]
    A=img[:512,:640]
    B=img[:512,640:1280]
    L=A/M
    H=B/N
    return L,H
def getAllFilesInFolder(folderpath):
    return [os.path.join(root,file) for root,dir,files in os.walk(folderpath) for file in files]

"""
#Usage:
L,H=loadLH('NOOBJECT_20-08-2018.txt','STTEST1.txt')
fusedimg=generateFusion(L,H)
choice=generateChc(L,H)
pc_sttest1=generateFColor(fusedimg,clut,choice)
plt.imshow(pc_sttest1)
plt.show()
#np.save('pc_sttest1.npy',pc_sttest1)
#"""
