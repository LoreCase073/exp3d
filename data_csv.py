import numpy as np
import os
import csv
import random
import pandas as pd
import rarfile

path = '/mnt/diskone-first/lcaselli/dataset'

expressions = {
            'bareteeth': 0,
            'cheeks_in': 1,
            'eyebrow': 2,
            'high_smile': 3,
            'lips_back': 4,
            'lips_up': 5,
            'mouth_down': 6,
            'mouth_extreme': 7,
            'mouth_middle': 8,
            'mouth_open': 9,
            'mouth_side': 10,
            'mouth_up': 11,
        }


reader = pd.read_csv('coma_dataset/dataset_names.csv', header=None)
dataset = []
for i in range(len(reader)):
    for expression in expressions:
        dataset.append({'Subject': reader.iloc[i,0], 'Expression': expression})

df = pd.DataFrame(dataset)

df.to_csv('coma_dataset/dataset.csv', index=False)



""" csv_file = pd.read_csv('training.csv')
for i in range(len(csv_file)):
    folder = os.path.join(path,csv_file.iloc[i, 0])
    print(folder)
    folder = os.path.join(folder, csv_file.iloc[i, 1])
    emotion_name = csv_file.iloc[i,1].replace('.rar','')
    emotion_dir = folder.replace('.rar','')
    if not os.path.isdir(emotion_dir):
            print("Creating Dir")
            for i in range(61):
                print("Extracting file " + str(i))
                with rarfile.RarFile(folder) as rf:
                    num = str(str(0) + str(i)) if i < 10 else str(i)
                    rf.extract(emotion_name + '/' + emotion_name + '_' + num +'.obj', emotion_dir) """
