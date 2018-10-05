dataset=$1
exp_name=$2
iter=$3
num_classes=$4

WORK_DIR=$(pwd)
cd ..
cd ..
ROOT_DIR=$(pwd)
cd $WORK_DIR
SEGNET_TUTORIAL_DIR="${ROOT_DIR}"/SegNet-Tutorial
CAFFE_DIR="${ROOT_DIR}"/caffe-segnet-cudnn5/build/tools
PYCAFFE_DIR="${ROOT_DIR}"/caffe-segnet-cudnn5/python
DATA_DIR="${SEGNET_TUTORIAL_DIR}"/"${dataset}"
RESULTS_DIR="${SEGNET_TUTORIAL_DIR}"/results
MODELS_DIR="${SEGNET_TUTORIAL_DIR}"/Models
PROTOTXT_DIR="${MODELS_DIR}"/"${exp_name}"
TRAINING_DIR="${MODELS_DIR}"/Training/"${exp_name}"
INFERENCE_DIR="${MODELS_DIR}"/Inference/"${exp_name}"

mkdir -p "${RESULTS_DIR}"/"${exp_name}"/test
mkdir -p "${RESULTS_DIR}"/"${exp_name}"/test/blended_"${iter}"


export PYTHONPATH=$PYTHONPATH:$PYCAFFE_DIR:$WORK_DIR

python eval_test.py \
	--inference_model "${PROTOTXT_DIR}"/inference_train.prototxt \
	--iteration $iter \
	--inference_dir "${INFERENCE_DIR}" \
	--training_dir "${TRAINING_DIR}" \
	--save_dir "${RESULTS_DIR}"/"${exp_name}"/test \
	--test_imgs "${DATA_DIR}"/train
	

# python visualize_segmentation.py \
#     --model "${PROTOTXT_DIR}"/inference_train.prototxt \
#     --weights "${INFERENCE_DIR}"/snapshot_iter_"${iter}"/test_weights.caffemodel \
#     --imgs_txt "${DATA_DIR}"/train.txt \
# 	--save_dir "${RESULTS_DIR}"/"${exp_name}"/test/blended_"${iter}" \
# 	--num_classes $num_classes
	
	
