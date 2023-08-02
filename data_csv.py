import numpy as np
import os
import csv
import random

path = '/mnt/diskone-first/lcaselli/dataset'

with open('dataset.csv','w') as file:
    with open('dataset_names.csv','w') as fname:
        wname = csv.writer(fname,delimiter=',')
        writer = csv.writer(file, delimiter=',')
        for filename in os.listdir(path):  
            wname.writerow([filename])
            subj = os.path.join(path,filename)
            for filerar in os.listdir(subj):
                writer.writerow([filename,filerar])
