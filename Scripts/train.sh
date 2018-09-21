exp_name=$1
nepoch=$2
niter=$3

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

mkdir -p "${TRAINING_DIR}"
mkdir -p "${INFERENCE_DIR}"

export PYTHONPATH=$PYTHONPATH:$PYCAFFE_DIR:$WORK_DIR

python solve.py  \
	--solver "${PROTOTXT_DIR}"/solver.prototxt \
	--train_model "${PROTOTXT_DIR}"/train.prototxt \
	--weights "${SEGNET_TUTORIAL_DIR}"/segnet_pascal.caffemodel \
	--niter "${niter}" \
	--nepoch "${nepoch}" \
	--training_dir "${TRAINING_DIR}" \
	--inference_dir "${INFERENCE_DIR}"



