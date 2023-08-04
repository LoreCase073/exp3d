from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import trimesh
import rarfile
import pandas as pd

'''
class Exp3dDataset(Dataset):

    def __init__(self, filepath, datatype, length):
        self.filepath = filepath
        self.length = length
        #TODO: implement how to get the dataset

    
    def __getitem__(self, index):

        file = ...
        
        #this is for 1 frame, i have to replicate for n = 60 frames
        for i in range():
            if i == 0:
                vertices = []
                with open(file) as f:
                    lines = f.readlines()

                    for l in lines:
                        if l.startswith('v '):
                            x = l.split(' ')
                            v = [float(x[1]), float(x[2]), float(x[3].replace('\n', ''))]
                            vertices.append(v)
                vertices = torch.FloatTensor(vertices)
            else:
                tmp = []
                with open(file) as f:
                    lines = f.readlines()

                    for l in lines:
                        if l.startswith('v '):
                            x = l.split(' ')
                            v = [float(x[1]), float(x[2]), float(x[3].replace('\n', ''))]
                            tmp.append(v)
                vertices = torch.cat([vertices, torch.FloatTensor(tmp)])

            #TODO: implement how to load the emotion vector
            #for single expression, 0 neutral, from 1 to 9 morph, from 10 50 expression climax, 60 neutral again
            emotion = ...


        return vertices, emotion

    
    def __len__(self):
        return self.len

'''

path = '/mnt/diskone-first/lcaselli/dataset'

class Exp3dDataset(Dataset):

    def __init__(self, filepath, csv_file, length):
        self.filepath = filepath
        #TODO: magari aggiungere che si può modificare lunghezza delle sequenze
        self.length = length
        self.csv_file = pd.read_csv(csv_file)
        self.extract_vertices = Extract_Vertices()
        self.emotion_encoding = Emotion_Encoding()
        

    #TODO: raffinare per prendere non solo lunghezza 61 stabilita
    def __getitem__(self, index):

        #folder to the file
        folder = os.path.join(self.filepath, self.csv_file.iloc[index, 0])

        #emotion to be extracted from the folder
        em = os.path.join(folder, self.csv_file.iloc[index, 1])

        #emotion name, removed the .rar
        emotion_name = self.csv_file.iloc[index,1].replace('.rar','')


        #extract vertices per ogni frame da 0 to 60
        
        vertices = self.extract_vertices(em, emotion_name)
        
        
        #for single expression, OHE of emotion
        emotion = self.emotion_encoding(emotion_name)


        return vertices, emotion

    def __len__(self):
        return self.length


class Extract_Vertices(object):
    def __init__(self):
        pass

    def __call__(self, folder, emotion_name):

        with rarfile.RarFile(folder) as rf:
            vertices = []
            for i in range(61):
                num = str(str(0) + str(i)) if i < 10 else str(i)
                rf.extract(emotion_name + '/' + emotion_name + '_' + num +'.obj','tmp/')
                #with rf.open(emotion_name + '/' + emotion_name + '_' + num +'.obj', mode='r') as f:
                #this is for 1 frame, i have to replicate for n = 61 frames
                myobj = trimesh.load_mesh('tmp/' + emotion_name + '/' + emotion_name + '_' + num +'.obj', file_type='obj')
                v = myobj.vertices
                vertices.append([v])
                os.remove('tmp/' + emotion_name + '/' + emotion_name + '_' + num +'.obj')
            
            vertices = torch.FloatTensor(np.array(vertices))
            vertices = vertices.squeeze()
            
            vertices = vertices.view(61,vertices.shape[1]*vertices.shape[2])
            
            
        
        return vertices

class Emotion_Encoding(object):
    def __init__(self):
        self.emotions = {
            'Disgust': 1,
            'Desire': 2,
            'Sad1': 3,
            'Angry1': 4,
            'Fear': 5,
            'Surprised': 6,
            'Concentrate': 7,
            'Happy': 8,
        }

    def __call__(self, emotion_name):

        emotion = []

        for i in range(61):
            emotion.append(self.emotions[emotion_name])

        emotion = torch.IntTensor(emotion)
        
        return emotion