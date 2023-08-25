from comet_ml import Experiment
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import os
from tqdm import tqdm
import time
import argparse
import json
import numpy as np
import torch.nn as nn
from data_loader import Exp3dDataset
from expmodel import ExpModel
import matplotlib.pyplot as plt
import trimesh


if __name__== '__main__':
    parser = argparse.ArgumentParser(description='Train Exp3D')

    parser.add_argument("--filepath", dest="filepath", 
                        help="Path to the dataset")
    parser.add_argument("--training_csv", dest="training_csv", 
                        help="Path to the csv training set")
    parser.add_argument("--validation_csv", dest="validation_csv", 
                        help="Path to the csv validation set")
    parser.add_argument("--project_name", dest="project_name",
                        help="Project name")
    parser.add_argument("--name_experiment", dest="name_experiment",
                        help="Experiment name")
    parser.add_argument("--weights_path", dest="weights_path",
                        help="Weights path")
    parser.add_argument("--obj_path", dest="obj_path",
                        help="Vertices save path")
    
    #model parameters
    parser.add_argument("--vertices_dim", dest="vertices_dim", default=5023,
                        help="Number of vertices")
    parser.add_argument("--feat_dim", dest="feat_dim", default=1024,
                        help="Dimension of features in the model")
    parser.add_argument("--emotion_dim", dest="emotion_dim", default=8,
                        help="Dimension of emotion vector")
    parser.add_argument("--dropout", dest="dropout", default=0.1,
                        help="Dropout applied to the model")
    parser.add_argument("--nhead_enc", dest="nhead_enc", default=8,
                        help="Number of head in the transformer encoder")
    parser.add_argument("--nhead_dec", dest="nhead_dec", default=8,
                        help="Number of head in the transformer decoder")
    parser.add_argument("--nlayer_dec", dest="nlayer_dec", default=4,
                        help="Number of layers in the transformer decoder")
    parser.add_argument("--nlayer_enc", dest="nlayer_enc", default=4,
                        help="Number of layers in the transformer encoder")
    parser.add_argument("--seq_dim", dest="seq_dim", default=61,
                        help="Dimension of sequences in the dataset")
    
    

    args = parser.parse_args()


    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {device}")

    hyper_parameters = {
    }

    experiment = Experiment(project_name=args.project_name, disabled=False)
    experiment.set_name(args.name_experiment)

    model = ExpModel(args=args, device=device)
    #model = ExpModelAutoregressive(args=args, device=device)

    
    experiment.log_parameters(hyper_parameters)
    experiment.set_model_graph(model)

    save_path = args.weights_path

    obj_save = os.path.join(args.obj_path, args.name_experiment)

    if not os.path.exists(obj_save):
        os.makedirs(obj_save)

    


    valid_set = Exp3dDataset(filepath=args.filepath,
    csv_file=args.validation_csv
    )


    val_loader = DataLoader(dataset=valid_set, 
    batch_size=1, 
    shuffle=True, 
    num_workers=1)

    model.load_state_dict(torch.load(save_path))
    model = model.to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Number of parameters in the model: {params}")

    experiment.log_other('n_param', params)

    print("Starting the training")
    

    

    lossFunc = nn.MSELoss()


    #START of the validation!
    
    print(f'Start Validation:')
    model.eval() 
    val_loss = 0.0
    
    with torch.no_grad():
        with tqdm(val_loader, unit='batch') as vepoch:
            for vertices, emotion, name, obj in vepoch:
                #TODO:fare evaluation in maniera autoregressiva

                vertices = vertices.to(device)
                emotion = emotion.to(device)

                output = model.predict(emotion,vertices[:,0,:],60).to(device)
                output = output.view(vertices.shape[0],61,int(args.vertices_dim),3)
                vertices = vertices.view(vertices.shape[0],61,int(args.vertices_dim),3)
                loss = lossFunc(output[:,1:,:], vertices[:,1:,:])
                val_loss += loss.item()

                torch.save(output.cpu(), obj_save + '/'  + str(name[0]) + '.pt')

                

        print('Validation Loss: ' + str(val_loss/len(val_loader)))
        experiment.log_metric('VAL_LOSS: ', val_loss/len(val_loader), step = 1)
        print(f'END EVALUATION ')
    
    
    experiment.end()