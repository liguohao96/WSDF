DATA_DIR=$1

python preprocessing/generate_dataset_label.py --dataset coma --data_dir ${DATA_DIR} \
    --out_dir Data/dataset_split/CoMA

python preprocessing/generate_dataset_cache.py --dataset coma --data_dir ${DATA_DIR} \
    --label_file Data/dataset_split/CoMA/coma-coma_interpolation_train.json \
    --drop_ratio 5 \
    --out_dir Data/

python preprocessing/generate_dataset_cache.py --dataset coma --data_dir ${DATA_DIR} \
    --label_file Data/dataset_split/CoMA/coma-coma_interpolation_test.json \
    --drop_ratio 0 --dataset_type test \
    --out_dir Data/