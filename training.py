import comet_ml as Experiment
import torch
from torch.utils.data import Dataloader
import torch.optim as optim
import torch.nn.functional as F
import os
from tqdm import tqdm
import time
import argparse
import json
import numpy as np
import torch.nn as nn


if __name__== '__main__':
    parser = argparse.ArgumentParser(description='Train MSAT - Contacts')

    parser.add_argument("--epochs", dest="epochs", default=5, help="number of epochs")
    parser.add_argument("--batch_size", dest="batch_size", default=64, help="Batch size")
    parser.add_argument("--lr", dest="lr", default=1e-4, help="learning rate train")
    parser.add_argument("--scheduling", dest="scheduling", default=0,
                        help="1 if scheduling lr policy applied, 0 otherwise")
    
    #model parameters
    parser.add_argument("--vertices_dim", dest="vertices_dim", default=5023,
                        help="Number of vertices")
    parser.add_argument("--feat_dim", dest="feat_dim", default=1024,
                        help="Dimension of features in the model")
    parser.add_argument("--emotion_dim", dest="emotion_dim", default=8,
                        help="Dimension of emotion vector")
    parser.add_argument("--dropout", dest="dropout", default=0.1,
                        help="Dropout applied to the model")
    parser.add_argument("--nhead", dest="nhead", default=8,
                        help="Number of head in the transformer")
    parser.add_argument("--nlayer", dest="nlayer", default=4,
                        help="Number of layers in the transformer")
    



    parser = ....args(parser)

    args = parser.parse_args()

    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    lr = float(args.lr)
    scheduling = int(args.scheduling)
    alphabet_size = int(args.alphabet_size)
    padding_idx = int(args.padding_idx)
    max_sequences = int(args.max_sequences)
    max_positions = int(args.max_positions)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {device}")

    hyper_parameters = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "scheduling": scheduling,
    }

    experiment = Experiment(project_name=args.project_name)
    experiment.set_name(args.name_experiment)

    msat = ...(...)

    
    experiment.log_parameters(hyper_parameters)
    experiment.set_model_graph(msat)

    save_path = os.path.join(args.weights_path, args.name_experiment)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    print(f"Save weights in: {save_path}.")

    # save hyperparams dictionary in save_weights_path
    with open(save_path + '/hyperparams.json', "w") as outfile:
        json.dump(hyper_parameters, outfile, indent=4)
    
    #Dataset loading
    msa_dataset = ExpDataset(file_csv=args.csv_dataset,
    npz=args.dataset_path,
    max_seq_len=max_sequences, 
    max_pos=max_positions,
    padding_idx=padding_idx,
    )


    valid_set = ExpDataset(file_csv=args.csv_val,
    npz=args.dataset_path,
    max_seq_len=max_sequences, 
    max_pos=max_positions,
    padding_idx=padding_idx,
    )

    training_loader = DataLoader(dataset=msa_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=2, 
    collate_fn=collate_tensors)

    val_loader = DataLoader(dataset=valid_set, 
    batch_size=1, 
    shuffle=True, 
    num_workers=2, 
    collate_fn=collate_tensors)

    optimizer = ...

    model = model.to(device)

    params = sum(p.numel() for p in msat.parameters() if p.requires_grad)

    print(f"Number of parameters in the model: {params}")

    experiment.log_other('n_param', params)

    print("Starting the training")
    
    lossFunc = ...

    for epoch in range(epochs):
        msat.train()

        running_loss = 0.0
        train_precision = 0.0
        with tqdm(training_loader, unit="batch") as tepoch:

            for msa, distances,_ in tepoch:
                tepoch.set_description(f"Epoch{epoch}")

                optimizer.zero_grad()

                output = model()
                
                loss = lossFunc(contacts, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()


        
        
        print(f"Training loss: {running_loss/len(training_loader)} Epoch: {epoch}")
        experiment.log_metric('train_loss', running_loss/len(training_loader), step=epoch+1)
        if epoch % 1 ==0:
            torch.save(msat.state_dict(), save_path + '/weights_' + (str(epoch+1)) + '.pth')

        #START of the validation!
        if epoch % 5 == 0:
            print(f'Start Validation:')
            model.eval() 
            
            with torch.no_grad():
                with tqdm(val_loader, unit='batch') as vepoch:
                    for msa, distances,_ in vepoch:

                        vepoch.set_description(f"Epoch{epoch}")

                        output = model()
                        
                        loss = lossFunc()
                        
                        val_loss += loss.item()

                print('Validation Loss: ' + str(val_loss/len(val_loader)))
                experiment.log_metric('VAL_LOSS: ', val_loss/len(val_loader), step = epoch + 1)
                print(f'END EVALUATION {epoch}:')
        
    
    torch.save(model.state_dict(), save_path + '/weights_' + 'final.pth')

    experiment.end()
    print('End of the training')