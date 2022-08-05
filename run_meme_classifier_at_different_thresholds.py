'''
Code to run the meme detector at different thresholds. 
'''
import imp
from subprocess import Popen, PIPE,CREATE_NEW_CONSOLE
import time
import os
import meme_detect


model_path="models/text.pt"
string_ends='\"'
import argparse
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--inputpath')
    parser.add_argument('-o','--outputpath')
    parser.add_argument('-m',"--model", default=model_path)

    return parser.parse_args()
parsed_arguments=parse_arguments()

input_folder_path=parsed_arguments.inputpath
output_folder_general_path=parsed_arguments.outputpath
model_path=parsed_arguments.model
thresholds=[.5,.4,.3,.2,.1,.05]

for text_threshold in thresholds:
    output_folder_name=f"meme{int(text_threshold*100)}"
    output_folder_path=output_folder_general_path+"/"+output_folder_name
    os.system(f"mkdir {string_ends+output_folder_path+string_ends}")

    # process1 = Popen(['python', 'meme-detect.py', '-i',input_folder,"-o",output_folder_name,"-t",str(threshold)], stdout=PIPE, stderr=PIPE,shell=True)
    # process1.wait()
    # time.sleep(15)
    print(f"started run or threshold {text_threshold}")
    meme_detect.meme_detect(input_folder_path,output_folder_path,text_threshold,model_path)