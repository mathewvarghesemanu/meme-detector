# Meme-detector
WhatsApp meme detector for removing unnecessary memes.

# running meme detector locally
python meme.py -i {path} -o {path} -t {threshold} between 0 to 1

# Python Training code
python train.py --img 640 --batch 16 --epochs 3 --data {path to the dataset yaml file} --weights yolov5m.pt

# References 

Tutorial to get started: https://github.com/deepakcrk/yolov5-crowdhuman

Dataset: https://universe.roboflow.com/skie-u1yzr/csgo-tr4j7/9

Training code (Colab): https://colab.research.google.com/drive/1s9-S36hy57NLTG_ZpDqP0Wh1_R_sRAjS?usp=sharing

Yolo v5: https://github.com/ultralytics/yolov5

Annotation: https://roboflow.com/


# Future work
Create a UI in Sreamlit
Create a folder structure and move memes accordingly
Create an android app that can suggest you memes to remove periodically

