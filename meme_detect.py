'''
Meme detector , works on yolo v5
takes input, output and threshold as arguments 
'''

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
from os import listdir
from os.path import isfile, join

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
    parser.add_argument('-m',"--model", default=model_path)
    return parser.parse_args()

def get_parsed_params(parsed_args):
    inputpath=parsed_args.inputpath
    outputpath=parsed_args.outputpath
    text_area_thres=float(parsed_args.textthreshold)
    model_path=parsed_args.model
    return inputpath,outputpath,text_area_thres,model_path

def get_files(mypath):
    onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
    return onlyfiles

def find_and_move_meme(results,image_area,text_area_thres,file_path,new_path):
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
    return True

def meme_detect(inputpath,outputpath,text_area_thres,model_path):
    all_images_path=get_files(inputpath)

    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    model.eval()

    image_area=0
    for file_path in tqdm(all_images_path):
        try:
            frame = cv2.imread(file_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_area=frame.shape[0]*frame.shape[1]
        except:
            continue
        # except Exception as e:
        #     print (e)

        results = model(frame)
        completed_flag=find_and_move_meme(results,image_area,text_area_thres,file_path,outputpath)
    print(f"Run Completed with Threshold:  {text_area_thres}")

def main(parsed_args):
    parsed_args = parse_arguments()
    inputpath,outputpath,text_area_thres,model_path=get_parsed_params(parsed_args)
    print(f"Threshold {float(parsed_args.textthreshold)*100}%")
    meme_detect(inputpath,outputpath,text_area_thres,model_path)

if __name__ == "__main__":
    main(parse_arguments())
    
        


