dataset=$1
exp_name=$2
num_classes=$4

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

mkdir -p "${RESULTS_DIR}"/"${exp_name}"/test
mkdir -p "${RESULTS_DIR}"/"${exp_name}"/test/blended_"${iter}"


export PYTHONPATH=$PYTHONPATH:$PYCAFFE_DIR:$WORK_DIR
	

python visualize_segmentation.py \
    --model "${PROTOTXT_DIR}"/inference_test.prototxt \
    --weights "${INFERENCE_DIR}"/test_weights.caffemodel \
    --imgs_txt "${DATA_DIR}"/test.txt \
	--save_dir "${RESULTS_DIR}"/"${exp_name}"
	--num_classes $num_classes
	
	
