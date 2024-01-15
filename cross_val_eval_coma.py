import numpy as np
import os
import trimesh
import torch

folder = ['test_coma_cross_1','test_coma_cross_2','test_coma_cross_3','test_coma_cross_4']
path = '/mnt/diskone-first/lcaselli/vertices/'
path_gt = '/mnt/diskone-first/lcaselli/COMA_data/'

predictions = []
ground_truth = []

for i in folder:
    tmp_path = os.path.join(path,i)
    for el in os.listdir(tmp_path):
        subj = el
        gt_subh_path = os.path.join(path_gt,subj)
        pred_subj_path = os.path.join(tmp_path,subj)

        for expr in os.listdir(pred_subj_path):
            gt_expr_path = os.path.join(gt_subh_path,expr)
            pred_expr_path = os.path.join(pred_subj_path,expr)

            num_frames = 0
            folder_sorted = sorted(os.listdir(gt_expr_path))
            for name in folder_sorted:
                #predictions
                #pred_path = tmp_path + '/' + el + '/' + el + '_' + num + '.ply'
                pred = trimesh.load_mesh(pred_expr_path+'/'+name.replace('.','_').replace('_ply','.ply'),process=False)
                v = pred.vertices
                predictions.append([v])
                #gt
                gt_obj = myobj = trimesh.load_mesh(gt_expr_path+'/'+name, process=False)
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