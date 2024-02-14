from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import trimesh
import pandas as pd




class Coma3dDataset(Dataset):

    def __init__(self, filepath, csv_file):
        self.filepath = filepath
        self.csv_file = pd.read_csv(csv_file)
        self.extract_vertices = Extract_Vertices_Coma()
        self.expression_enc = Expression_Encoding()
        

    def __getitem__(self, index):

        template = os.path.join(self.filepath, 'COMA_data_actors')

        path = os.path.join(self.filepath, 'COMA_data')

        

        #folder to the file
        folder = os.path.join(path, self.csv_file.iloc[index, 0])

        template = template + '/' + self.csv_file.iloc[index,0] + '/' + self.csv_file.iloc[index,0] + '.ply'
        myobj = trimesh.load_mesh(template, process=False)
        vert = []
        v = myobj.vertices
        vert.append([v])
        
        vert = torch.FloatTensor(np.array(vert))
        vert = vert.squeeze()
        
        template = vert.view(vert.shape[0]*vert.shape[1])


        

        #emotion to be extracted from the folder
        expression = os.path.join(folder, self.csv_file.iloc[index, 1])

        #emotion name, removed the .rar
        expression_name = self.csv_file.iloc[index,1]

        name = self.csv_file.iloc[index,0]+ '/' + expression_name

        
        vertices, num_frames = self.extract_vertices(expression, expression_name)
        
        
        #for single expression, OHE 
        expression_encoding = self.expression_enc(expression_name, num_frames)

        expression_path = expression + '/' 


        return vertices, expression_encoding, name, expression_path, template, num_frames

    def __len__(self):
        return self.csv_file.shape[0]


class Exp3dDataset(Dataset):

    def __init__(self, filepath, csv_file):
        self.filepath = filepath
        self.csv_file = pd.read_csv(csv_file)
        self.extract_vertices = Extract_Vertices()
        self.emotion_encoding = Emotion_Encoding()
        

    def __getitem__(self, index):

        template = os.path.join(self.filepath, 'COMA_Florence_Actors_Aligned')

        path = os.path.join(self.filepath, 'COMA_FLAME_Aligned')

        

        #folder to the file
        folder = os.path.join(path, self.csv_file.iloc[index, 0])

        template = template + '/' + self.csv_file.iloc[index,0] + '.ply'
        myobj = trimesh.load_mesh(template, process=False)
        vert = []
        v = myobj.vertices
        vert.append([v])
        
        vert = torch.FloatTensor(np.array(vert))
        vert = vert.squeeze()
        
        template = vert.view(vert.shape[0]*vert.shape[1])


        

        #emotion to be extracted from the folder
        em = os.path.join(folder, self.csv_file.iloc[index, 1])

        #emotion name, removed the .rar
        emotion_name = self.csv_file.iloc[index,1]

        name = self.csv_file.iloc[index,0]+ '_' + emotion_name

        #extract vertices per ogni frame da 0 to 60
        
        vertices = self.extract_vertices(em, emotion_name)
        
        
        #for single expression, OHE of emotion
        emotion = self.emotion_encoding(emotion_name)

        em = em + '/' + emotion_name  + '_' 


        return vertices, emotion, name, em, template

    def __len__(self):
        return self.csv_file.shape[0]



class Extract_Vertices(object):
    def __init__(self):
        pass

    def __call__(self, folder, emotion_name):

        
        vertices = []
        for i in range(61):
            num = str(str(0) + str(i)) if i < 10 else str(i)
            myobj = trimesh.load_mesh(folder + '/' + emotion_name + '_' + num +'.ply', process=False)
            v = myobj.vertices
            vertices.append([v])
        
        vertices = torch.FloatTensor(np.array(vertices))
        vertices = vertices.squeeze()
        
        vertices = vertices.view(61,vertices.shape[1]*vertices.shape[2])
            
            
        
        return vertices

class Extract_Vertices_Coma(object):
    def __init__(self):
        pass

    def __call__(self, folder, expression_name):

        
        vertices = []
        num_frames = 0
        folder_sorted = sorted(os.listdir(folder))
        for name in folder_sorted:
            myobj = trimesh.load_mesh(folder+'/'+name, process=False)
            v = myobj.vertices
            vertices.append([v])
            num_frames += 1
        
        vertices = torch.FloatTensor(np.array(vertices))
        vertices = vertices.squeeze()
        
        vertices = vertices.view(num_frames,vertices.shape[1]*vertices.shape[2])
   
        return vertices, num_frames

class Emotion_Encoding(object):
    def __init__(self):
        self.emotions = {
            'Disgust': 0,
            'Desire': 1,
            'Sad1': 2,
            'Angry1': 3,
            'Fear': 4,
            'Surprised': 5,
            'Concentrate': 6,
            'Happy': 7,
        }

    def __call__(self, emotion_name):

        emotion = []

        for i in range(61):
            emotion.append(self.emotions[emotion_name])

        emotion = torch.IntTensor(emotion)
        
        return emotion



class Expression_Encoding(object):
    def __init__(self):
        self.expressions = {
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

    def __call__(self, expression_name, length):

        expression = []

        for i in range(length):
            expression.append(self.expressions[expression_name])

        expression = torch.IntTensor(expression)
        
        return expression