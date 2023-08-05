import numpy as np
import os
import csv
import random
import pandas as pd
import rarfile

path = '/mnt/diskone-first/lcaselli/dataset'

""" with open('dataset.csv','w') as file:
    with open('dataset_names.csv','w') as fname:
        wname = csv.writer(fname,delimiter=',')
        writer = csv.writer(file, delimiter=',')
        for filename in os.listdir(path):  
            wname.writerow([filename])
            subj = os.path.join(path,filename)
            for filerar in os.listdir(subj):
                writer.writerow([filename,filerar])
 """

csv_file = pd.read_csv('validation.csv')
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
                    rf.extract(emotion_name + '/' + emotion_name + '_' + num +'.obj', emotion_dir)
