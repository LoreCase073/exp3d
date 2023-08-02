import numpy as np
import torch
from torch.utils.data import Dataset
import os
import trimesh

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

    def __init__(self, filepath, datatype, length):
        self.filepath = filepath
        self.length = length
        #TODO: implement how to get the dataset

    
    def __getitem__(self, index):

        file = ...
        
        #this is for 1 frame, i have to replicate for n = 61 frames
        myobj = trimesh.load_mesh(self.filepath + '.stl')
        vertices = myobj.vertices
        #TODO: implement how to load the emotion vector
        #for single expression, 0 neutral, from 1 to 9 morph, from 10 50 expression climax, 60 neutral again
        emotion = ...


        return vertices, emotion

    
    def __len__(self):
        return self.len