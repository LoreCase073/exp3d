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
from utils import compute_distance, composite_loss


if __name__== '__main__':
    parser = argparse.ArgumentParser(description='Train Exp3D')

    parser.add_argument("--epochs", dest="epochs", default=5, help="number of epochs")
    parser.add_argument("--batch_size", dest="batch_size", default=64, help="Batch size")
    parser.add_argument("--lr", dest="lr", default=1e-4, help="learning rate train", type=float)
    parser.add_argument("--filepath", dest="filepath", 
                        help="Path to the dataset")
    parser.add_argument("--template_path", dest="template_path", 
                        help="Path to the dataset templates")
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
    parser.add_argument("--feat_dim", dest="feat_dim", default=768,
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
    parser.add_argument("--c1", dest="c1", default=1,
                        help="Coefficient for loss")
    parser.add_argument("--c2", dest="c2", default=0.5,
                        help="Coefficient for loss")
    
    
    

    args = parser.parse_args()

    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    lr = float(args.lr)
    scheduling = int(args.scheduling)

    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {device}")

    hyper_parameters = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "scheduling": scheduling,
    }

    experiment = Experiment(project_name=args.project_name, disabled=False)
    experiment.set_name(args.name_experiment)

    model = ExpModel(args=args, device=device)

    
    experiment.log_parameters(hyper_parameters)
    experiment.set_model_graph(model)

    save_path = os.path.join(args.weights_path, args.name_experiment)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    
    print(f"Save weights in: {save_path}.")



    # save hyperparams dictionary in save_weights_path
    with open(save_path + '/hyperparams.json', "w") as outfile:
        json.dump(hyper_parameters, outfile, indent=4)
    
    #Dataset loading
    training_set = Exp3dDataset(filepath=args.filepath,
                             template_path=args.template_path,
    csv_file=args.training_csv
    )


    valid_set = Exp3dDataset(filepath=args.filepath,
                             template_path=args.template_path,
    csv_file=args.validation_csv
    )

    training_loader = DataLoader(dataset=training_set, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=8)

    val_loader = DataLoader(dataset=valid_set, 
    batch_size=1, 
    shuffle=True, 
    num_workers=4)

    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # define the model
    model = model.to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Number of parameters in the model: {params}")

    experiment.log_other('n_param', params)

    print("Starting the training")
    

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        train_distance = 0.0
        
        
        with tqdm(training_loader, unit="batch") as tepoch:

            for vertices, emotion, _, _, template in tepoch:
                tepoch.set_description(f"Epoch{epoch}")
                vertices = vertices.to(device)
                emotion = emotion.to(device)
                template = template.to(device)

                optimizer.zero_grad()

                output = model(emotion, vertices, template).to(device)
                output = output.view(vertices.shape[0],61,int(args.vertices_dim),3)
                vertices = vertices.view(vertices.shape[0],61,int(args.vertices_dim),3)

                loss = composite_loss(output[:,1:,:,:], vertices[:,1:,:,:], float(args.c1), float(args.c2))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                train_distance += compute_distance(output,vertices)


        
        
        print(f"Training loss: {running_loss/len(training_loader)} Epoch: {epoch}")
        experiment.log_metric('train_loss', running_loss/len(training_loader), step=epoch+1)
        experiment.log_metric('training_distance', train_distance/len(training_loader), step=epoch+1)
        if epoch % 10 ==0:
            torch.save(model.state_dict(), save_path + '/weights_' + (str(epoch+1)) + '.pth')

        #START of the validation!
        if epoch % 2 == 0:
            print(f'Start Validation:')
            model.eval() 
            val_loss = 0.0
            distance = 0.0
            
            with torch.no_grad():
                with tqdm(val_loader, unit='batch') as vepoch:
                    for vertices, emotion, name, obj, template in vepoch:

                        vepoch.set_description(f"Epoch{epoch}")
                        vertices = vertices.to(device)
                        emotion = emotion.to(device)
                        template = template.to(device)

                        output = model.predict(emotion,vertices[:,0,:],template,60).to(device)
                        output = output.view(vertices.shape[0],61,int(args.vertices_dim),3)
                        vertices = vertices.view(vertices.shape[0],61,int(args.vertices_dim),3)
                        loss = composite_loss(output[:,1:,:,:], vertices[:,1:,:,:], float(args.c1), float(args.c2))

                        distance += compute_distance(output,vertices)

                        
                        
                        val_loss += loss.item()

                        

                print('Validation Loss: ' + str(val_loss/len(val_loader)))
                experiment.log_metric('VAL_LOSS: ', val_loss/len(val_loader), step = epoch + 1)
                experiment.log_metric('Distance: ', distance/len(val_loader), step = epoch + 1)
                print(f'END EVALUATION {epoch}:')
        
    
    torch.save(model.state_dict(), save_path + '/weights_' + 'final.pth')

    experiment.end()
    print('End of the training')