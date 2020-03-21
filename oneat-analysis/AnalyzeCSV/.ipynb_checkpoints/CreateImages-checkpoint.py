#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:57:47 2019

@author: aimachine
"""


import cv2
import numpy as np
from tifffile import imread 
import os
import glob
import math
from tifffile import imsave
import matplotlib.pyplot as plt
try:
    from pathlib import Path
    Path().expanduser()
except (ImportError,AttributeError):
    from pathlib2 import Path

try:
    import tempfile
    tempfile.TemporaryDirectory
except (ImportError,AttributeError):
    from backports import tempfile
    
"""
 
   Here we have added some of the useful functions taken from the csbdeep package which are a part of third party software called CARE
   https://github.com/CSBDeep/CSBDeep

"""    
  ##Save image data as a tiff file, function defination taken from CARE csbdeep python package  
    
def save_tiff_imagej_compatible(file, img, axes, **imsave_kwargs):
    """Save image in ImageJ-compatible TIFF format.

    Parameters
    ----------
    file : str
        File name
    img : numpy.ndarray
        Image
    axes: str
        Axes of ``img``
    imsave_kwargs : dict, optional
        Keyword arguments for :func:`tifffile.imsave`

    """
    t = np.uint8
    # convert to imagej-compatible data type
    t_new = t
    img = img.astype(t_new, copy=False)

    imsave_kwargs['imagej'] = True
    imsave(file, img, **imsave_kwargs)    
    
    
def CountTwoDevents(csv_file,Label):

      x, y, time =   np.loadtxt(csv_file, delimiter = ',', skiprows = 0, unpack=True)
      
      f = open(csv_file)
      numlines = len(f.readlines())
      
      return numlines, Label
  
def CountTimeLapseevents(csv_file,image, Label, save_dir):

         x, y, time =   np.loadtxt(csv_file, delimiter = ',', skiprows = 0, unpack=True)
         count = 0
         eventcounter = 0
         eventlist = []
         timelist = []   
         listtime = time.tolist()
         listtime = sorted(listtime)
         maxtime = image.shape[0]
        
         for t in range(0, maxtime):
             eventcounter = listtime.count(t)
             timelist.append(t)   
             eventlist.append(eventcounter)
         
         plt.plot(timelist, eventlist, '-r')
         plt.title(Label)
         plt.ylabel('Counts')
         plt.xlabel('Time')
         plt.savefig(save_dir  + Label   + '.png') 
         plt.show() 
         
    
def TimelapseImage(csv_file, image, Label, save_dir):

  x, y, time =   np.loadtxt(csv_file, delimiter = ',', skiprows = 0, unpack=True)   
  axes = 'TYX'
  ReturnImage = image
  for t in range(0, len(time)):
      
      Currentimage = image[int(time[t])-1, :, :]
      if math.isnan(x[t]):
         continue 
      else:
         location = (int(x[t]), int(y[t]))
         cv2.circle(Currentimage, location, 2,(255,0,0), thickness = -1 )
         cv2.circle(Currentimage, location, 20, (255,0,0), thickness = 2)
      ReturnImage[int(time[t])-1, :, :] = Currentimage
         
  save_tiff_imagej_compatible((save_dir  + Label + '.tif'  ) , ReturnImage, axes)
      
def TwoDImage(csv_file, image, Label, save_dir):

  x, y, time =   np.loadtxt(csv_file, delimiter = ',', skiprows = 0, unpack=True)   
  axes = 'YX'
  ReturnImage = image
      
  Currentimage = image

  f = open(csv_file)
  numlines = len(f.readlines())
  if numlines > 1:
  
     for t in range(0, len(x)):
        location = (int(x[t]), int(y[t]))
        cv2.circle(Currentimage, location, 2,(255,0,0), thickness = -1 )
        cv2.circle(Currentimage, location, 20, (255,0,0), thickness = 2)
        ReturnImage= Currentimage
        
  else:
        location = (int(x), int(y))
        cv2.circle(Currentimage, location, 2,(255,0,0), thickness = -1 )
        cv2.circle(Currentimage, location, 20, (255,0,0), thickness = 2)
        ReturnImage= Currentimage
             
         
  save_tiff_imagej_compatible((save_dir  + Label + '.tif'  ) , ReturnImage, axes)    
    
         
def main(csv_file, image_file, Label, save_dir): 
    TwoDImage(csv_file, image_file, Label, save_dir)
    

if __name__ == "__main__":
        
        save_dir = '/home/sancere/Desktop/Aurelien/ImageResults/'
        
        csv_dir = '/home/sancere/Desktop/Aurelien/Results/'
        image_dir =  '/home/sancere/Desktop/Aurelien/'
        

        Raw_path = os.path.join(image_dir,'*.tif')
        X = glob.glob(Raw_path)

        


        Path(save_dir).mkdir(exist_ok = True)
        

        for image_file in X:
          
          image= imread(image_file)
          image = np.asarray(image)
          Name = os.path.basename((os.path.splitext(image_file)[0])) 
          csv_Name = 'CNNMatureLocationEventCounts' + Name
          csv_file = csv_dir + csv_Name + '.csv'
        
        

          if os.path.exists(csv_file):
            
              emptyimage = np.zeros([image.shape[0], image.shape[1]], dtype='uint16')
              TwoDImage(csv_file, emptyimage, 'Mature' + Name, save_dir)   
          
         
