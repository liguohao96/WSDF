DATA_DIR=$1

mkdir -p Data/dataset_split/FaceScape

CODE=$(cat << EOF
import json
import random

# most from https://github.com/rmraaron/FaceExpDisentanglement/blob/main/utils/facescape_dataset.py#L74C20-L74C20

random.seed(50)
pid_int = list(range(1, 832)) + list(range(833, 848))  # we don't have '832'
test_pid_i_list  = random.sample(pid_int, int(0.3*len(pid_int)))
publicable_list = [122, 212, 340, 344, 393, 395, 421, 527, 594, 610]

k = 0
for publicable_id in publicable_list:
    if publicable_id not in test_pid_i_list:
        test_pid_i_list[k] = int(publicable_id)
        k += 1

test_pids  = [str(i) for i in sorted(test_pid_i_list)]
train_pids = [str(i) for i in sorted(list(set(pid_int) - set(test_pid_i_list)))]

config = {
    "test":  test_pids,
    "train": train_pids,
    "valid": []
}

with open("Data/dataset_split/FaceScape/facescape_pid_split.json", "w") as f:
    json.dump(config, f, indent=2)
EOF
)
python -c "$CODE"

python preprocessing/generate_dataset_label.py --dataset facescape --data_dir ${DATA_DIR} \
    --out_dir Data/dataset_split/FaceScape --split_method split_by_pid --split_config Data/dataset_split/FaceScape/facescape_pid_split.json

python preprocessing/generate_dataset_cache.py --dataset facescape --data_dir ${DATA_DIR} \
    --label_file Data/dataset_split/FaceScape/facescape-split_by_pid-train.json \
    --drop_ratio 5 \
    --out_dir Data/

python preprocessing/generate_dataset_cache.py --dataset facescape --data_dir ${DATA_DIR} \
    --label_file Data/dataset_split/FaceScape/facescape-split_by_pid-test.json \
    --drop_ratio 0 --dataset_type test \
    --out_dir Data/