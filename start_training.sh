python training.py --epochs 500 --batch_size 1 --lr 5e-4 --scheduling 0 --vertices_dim 5023 --feat_dim 1536 \
    --emotion_dim 8 --dropout 0.1 --nhead_enc 4 --nhead_dec 6 --nlayer_enc 4 --nlayer_dec 6 --c1 1 --c2 0.5 \
    --seq_dim 61 --filepath 'path_to_dataset' --template_path 'path_to_templates' --training_csv './exp3d/dataset_cross_val/training_cross_4.csv' \
    --validation_csv './exp3d/dataset_cross_val/validation_cross_4.csv' --project_name 'name_for_COMET_project' \
    --name_experiment 'experiment_name' --weights_path 'path_to_weights' --obj_path 'path_to_vertices' 