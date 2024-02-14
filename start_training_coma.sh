python training_coma.py --epochs 500 --batch_size 1 --lr 5e-4 --scheduling 0 --vertices_dim 5023 \
    --feat_dim 768 --emotion_dim 12 --dropout 0.1 --nhead_enc 4 --nhead_dec 6 --nlayer_enc 4 --nlayer_dec 6 \
    --c1 1 --c2 0.5 --filepath '/mnt/diskone-first/lcaselli/' --training_csv './exp3d/coma_cross_val/training_cross_4.csv' \
    --validation_csv './exp3d/coma_cross_val/validation_cross_4.csv' --project_name 'exp3d' --name_experiment 'coma_cross_4' \
    --weights_path '/mnt/diskone-first/lcaselli/weights/' --obj_path '/mnt/diskone-first/lcaselli/vertices/' 