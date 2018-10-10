dataset=$1
exp_name=$2
num_classes=$3

WORK_DIR=$(pwd)
cd ..
cd ..
ROOT_DIR=$(pwd)
cd $WORK_DIR
SEGNET_TUTORIAL_DIR="${ROOT_DIR}"/SegNet-Tutorial
CAFFE_DIR="${ROOT_DIR}"/caffe-segnet-cudnn5/build/tools
PYCAFFE_DIR="${ROOT_DIR}"/caffe-segnet-cudnn5/python
DATA_DIR="${SEGNET_TUTORIAL_DIR}"/data/"${dataset}"
RESULTS_DIR="${SEGNET_TUTORIAL_DIR}"/results
MODELS_DIR="${SEGNET_TUTORIAL_DIR}"/Models
PROTOTXT_DIR="${MODELS_DIR}"/"${exp_name}"
TRAINING_DIR="${MODELS_DIR}"/Training/"${exp_name}"
INFERENCE_DIR="${MODELS_DIR}"/Inference/"${exp_name}"

mkdir -p "${RESULTS_DIR}"/"${exp_name}"/train
mkdir -p "${RESULTS_DIR}"/"${exp_name}"/val
mkdir -p "${RESULTS_DIR}"/"${exp_name}"/trainval


export PYTHONPATH=$PYTHONPATH:$PYCAFFE_DIR:$WORK_DIR

python eval_trainval.py \
	--inference_model "${PROTOTXT_DIR}"/inference_train.prototxt \
	--inference_dir "${INFERENCE_DIR}" \
	--training_dir "${TRAINING_DIR}" \
	--save_dir "${RESULTS_DIR}"/"${exp_name}"/train \
	--test_imgs "${DATA_DIR}"/trainannot \
	--num_classes $num_classes
	
	
# python eval_trainval.py \
# 	--inference_model "${PROTOTXT_DIR}"/inference_val.prototxt \
# 	--inference_dir "${INFERENCE_DIR}" \
# 	--training_dir "${TRAINING_DIR}" \
# 	--save_dir "${RESULTS_DIR}"/"${exp_name}"/val \
# 	--test_imgs "${DATA_DIR}"/val
# 
# python plot_trainval.py \
# 	--train_results "${RESULTS_DIR}"/"${exp_name}"/train \
# 	--val_results "${RESULTS_DIR}"/"${exp_name}"/val \
# 	--save_dir "${RESULTS_DIR}"/"${exp_name}"/trainval
	
