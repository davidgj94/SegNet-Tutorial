dataset=$1
exp_name=$2
iter=$3
num_classes=$4
part=$5

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
SAVE_DIR_MASK="${RESULTS_DIR}"/"${exp_name}"/"${part}"_"${iter}"_nolabeL_mask
SAVE_DIR_BLENDED="${RESULTS_DIR}"/"${exp_name}"/"${part}"_"${iter}"_nolabeL_blended

mkdir -p $SAVE_DIR_MASK
mkdir -p $SAVE_DIR_BLENDED

export PYTHONPATH=$PYTHONPATH:$PYCAFFE_DIR:$WORK_DIR

python test_no_label.py \
	--model $PROTOTXT_DIR/inference_"${part}".prototxt \
	--weights $INFERENCE_DIR/snapshot_iter_$iter/test_weights.caffemodel \
	--imgs_txt $DATA_DIR/"${part}"_cropped.txt \
	--imgs_dir $DATA_DIR/"${part}" \
	--num_classes $num_classes \
	--save_dir_mask $SAVE_DIR_MASK \
	--save_dir_blende $SAVE_DIR_BLENDED
