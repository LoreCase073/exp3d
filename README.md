# exp3d
Project for Computer Graphics and 3D project.
The objective of the project was to generate a sequences of frames representing a 3D mesh representing a face expressing a given emotion, starting from a neutral expression 3D mesh.

This is done with implementing a Transformer with Linear biases, following the approach in [paper] (https://arxiv.org/pdf/2112.05329.pdf) .

The training is done on the Florence and COMA dataset.


## Dependencies

- Check the required python packages in `requirements.txt`.

## Data splits

- The train, val splits are specified for the Florence dataset from `training.csv` and `validation.csv`. If required to work with a cross validation set, in the `dataset_cross_val` and `coma_cross_val` are specified the subsets for k = 4 subsets.

- The data for the Florence dataset should be organized as follows.
- `Florence_directory/subject_name/emotion/emotion_framenumber.ply` for the sequences of meshes;
- `Florence_template_directory/subject_name.ply` for the templates to be subtracted in the model;
- The data for the COMA dataset should be organized as described for the Florence dataset.

## Training and Testing

To start the training it can be done as following:

 
	python training.py --epochs 500 --batch_size 1 --lr 5e-4 --scheduling 0 --vertices_dim 5023 --feat_dim 1536 \
    --emotion_dim 8 --dropout 0.1 --nhead_enc 4 --nhead_dec 6 --nlayer_enc 4 --nlayer_dec 6 --c1 1 --c2 0.5 \
    --seq_dim 61 --filepath 'path_to_dataset' --template_path 'path_to_templates' --training_csv './exp3d/dataset_cross_val/training_cross_4.csv' \
    --validation_csv './exp3d/dataset_cross_val/validation_cross_4.csv' --project_name 'name_for_COMET_project' \
    --name_experiment 'experiment_name' --weights_path 'path_to_weights' --obj_path 'path_to_vertices' 


Or you can use the corresponding bash file.

To start the test it can be done as following:


	python test.py --vertices_dim 5023 --feat_dim 1536 --emotion_dim 8 --dropout 0.1 --nhead_enc 4 --nhead_dec 6 --nlayer_enc 4 \
    --nlayer_dec 6 --c1 1 --c2 0.5 --filepath 'path_to_dataset' --template_path 'path_to_templates' \
    --validation_csv './exp3d/dataset_cross_val/validation_cross_4.csv' \
    --seq_dim 61 --project_name 'name_for_COMET_project' --name_experiment 'experiment_name' --weights_path 'path_to_weights' \
    --obj_path 'path_to_vertices'


Or you can use the corresponding bash file.

For the COMA dataset, the inputs are similar, just need to use different training_coma.py and test_coma.py.