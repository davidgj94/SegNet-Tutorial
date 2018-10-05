exp_name="segnet_canard_v5"

WORK_DIR=$(pwd)
cd ..
cd ..
ROOT_DIR=$(pwd)
cd $WORK_DIR
SEGNET_TUTORIAL_DIR="${ROOT_DIR}"/SegNet-Tutorial
CAFFE_DIR="${ROOT_DIR}"/caffe-segnet-cudnn5/build/tools
PYCAFFE_DIR="${ROOT_DIR}"/caffe-segnet-cudnn5/python
RESULTS_DIR="${SEGNET_TUTORIAL_DIR}"/results
MODELS_DIR="${SEGNET_TUTORIAL_DIR}"/Models
PROTOTXT_DIR="${MODELS_DIR}"/"${exp_name}"
TRAINING_DIR="${MODELS_DIR}"/Training/"${exp_name}"
INFERENCE_DIR="${MODELS_DIR}"/Inference/"${exp_name}"

export PYTHONPATH=$PYTHONPATH:$PYCAFFE_DIR:$WORK_DIR

python test_canard_v5.py  \
	--solver "${PROTOTXT_DIR}"/solver.prototxt \
	--weights "${SEGNET_TUTORIAL_DIR}"/segnet_pascal.caffemodel





