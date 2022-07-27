import torch
import pyautogui
import cv2
import numpy as np
import time
import mss
import numpy as np
import os
import shutil
from tqdm import tqdm
import argparse, os

model_path="models/text.pt"
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--inputpath')
    parser.add_argument('-o','--outputpath')
    parser.add_argument('-t','--textthreshold')
    return parser.parse_args()

            
parsed_args = parse_arguments()
print(parsed_args.textthreshold)
# mypath="C:\\Users\\MATHE\\Desktop\\meme-detection\\yolov5\\_test"
mypath=parsed_args.inputpath
new_path=parsed_args.outputpath
# new_path="C:\\Users\\MATHE\\Desktop\\meme-detection\\yolov5\\_test\\new"
text_area_thres=float(parsed_args.textthreshold)

from os import listdir
from os.path import isfile, join

onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]


model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
model.eval()

for file_path in tqdm(onlyfiles):
    try:
        frame = cv2.imread(file_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print (e)
    image_area=frame.shape[0]*frame.shape[1]
    results = model(frame)
    

    # print(results.pandas().xyxy[0].shape)
    # print(type(results.pandas().xyxy[0]))
    text_area=0.00001
    try:
        for i in range( results.xyxy[0].shape[0]):
            x0,y0,x1,y1,confi,cla = results.xyxy[0][i].cpu().numpy()
            w=x1-x0
            h=y1-y0
            area=w*h
            text_area=text_area+area
            if text_area/image_area>float(text_area_thres):
                # print(f"meme {text_area/image_area}")
                shutil.move(file_path, new_path)
                break
    except Exception as e:
        print (e)
    
        


