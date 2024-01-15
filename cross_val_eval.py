import numpy as np
import os
import trimesh
import torch

folder = ['test_come_cross_1','test_come_cross_2','test_come_cross_3','test_come_cross_4']
path = '/mnt/diskone-first/lcaselli/vertices/'
path_gt = '/mnt/diskone-first/lcaselli/COMA_data/'

predictions = []
ground_truth = []

for i in folder:
    tmp_path = os.path.join(path,i)
    for el in os.listdir(tmp_path):
        x = el.split('_')
        subj = x[0]+ '_' + x[1]
        gt = os.path.join(path_gt,subj)
        gt = os.path.join(gt,x[2])
        for k in range(1,61):
            num = str(str(0) + str(k)) if k < 10 else str(k)
            #predictions
            pred_path = tmp_path + '/' + el + '/' + el + '_' + num + '.ply'
            pred = trimesh.load_mesh(pred_path,process=False)
            v = pred.vertices
            predictions.append([v])
            #gt
            gtp = gt + '/' + x[2] + '_' + num + '.ply'
            gt_obj = trimesh.load_mesh(gtp,process=False)
            w = gt_obj.vertices
            ground_truth.append([w])




predictions = torch.FloatTensor(np.array(predictions))
predictions = predictions.squeeze()

ground_truth = torch.FloatTensor(np.array(ground_truth))
ground_truth = ground_truth.squeeze()

mean_err = torch.mean(torch.sqrt(torch.sum((predictions-ground_truth)**2, axis = 2)))
std_err = torch.std(torch.sum((predictions-ground_truth)**2, axis = 2))

print(f'Error {mean_err:0,.6f}')
print('Std', std_err)