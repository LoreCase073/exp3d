from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import trimesh
import rarfile

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
        self.length = length
        self.csv_file = csv_file
        self.extract_vertices = Extract_Vertices()
        self.emotion_encoding = Emotion_Encoding()
        #TODO: implement how to get the dataset

    
    def __getitem__(self, index):

        #TODO: estrarre rar
        folder = os.path.join(self.filepath, self.csv_file.iloc[index, 0])

        em = os.path.join(folder, self.csv_file.iloc[index, 1])

        emotion_name = self.csv_file.iloc[index,1].replace('.rar','')


        #TODO:extract vertices per ogni frame da 0 to 60
        
        vertices = self.extract_vertices(em, emotion_name)
        
        
        #TODO: implement how to load the emotion vector
        #for single expression, OHE of emotion
        emotion = self.emotion_encoding(emotion_name)


        return vertices, emotion

    
    def __len__(self):
        return self.len


class Extract_Vertices(object):
    def __init__(self):
        pass

    def __call__(self, folder, emotion_name):

        with rarfile.Rarfile(folder) as rf:
            vertices = []
            for i in range(61):
                num = str(0 + i) if i < 10 else str(i)
                with rf.open(emotion_name + '_' + num +'.obj') as f:
                    #this is for 1 frame, i have to replicate for n = 61 frames
                    myobj = trimesh.load_mesh(f)
                    v = myobj.vertices
                    vertices.append(v)
            
            vertices = torch.FloatTensor(vertices)
        
        return vertices

class Emotion_Encoding(object):
    def __init__(self):
        self.emotions = {
            'Disgust': [1,0,0,0,0,0,0,0],
            'Desire': [0,1,0,0,0,0,0,0],
            'Sad1': [0,0,1,0,0,0,0,0],
            'Angry1': [0,0,0,1,0,0,0,0],
            'Fear': [0,0,0,0,1,0,0,0],
            'Surprised': [0,0,0,0,0,1,0,0],
            'Concentrate': [0,0,0,0,0,0,1,0],
            'Happy': [0,0,0,0,0,0,0,1],
        }

    def __call__(self, emotion_name):

        emotion = []

        for i in range(61):
            emotion.append(self.emotions[emotion_name])

        emotion = torch.FloatTensor(emotion)
        
        return emotion