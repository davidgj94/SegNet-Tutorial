WORK_DIR=$(pwd)
cd ..
cd ..
ROOT_DIR=$(pwd)
SEGNET_TUTORIAL_DIR="${ROOT_DIR}"/SegNet-Tutorial
CAFFE_DIR="${ROOT_DIR}"/caffe-segnet-cudnn5/build/tools
PYCAFFE_DIR="${ROOT_DIR}"/caffe-segnet-cudnn5/python
ROADS_DIR="${SEGNET_TUTORIAL_DIR}"/$4/ROADS
RESULTS_DIR="${SEGNET_TUTORIAL_DIR}"/results
MODELS_DIR="${SEGNET_TUTORIAL_DIR}"/Models
PROTOTXT_DIR="${MODELS_DIR}"/$1

exp_name=$2
num_epochs=$3

mkdir -p "${MODELS_DIR}"/Training/"${exp_name}"
mkdir -p "${MODELS_DIR}"/Inference/"${exp_name}"
mkdir -p "${RESULTS_DIR}"/"${exp_name}"/train

cd $WORK_DIR

./change_snapshot_prefix.sh "${PROTOTXT_DIR}" "${exp_name}"

export PYTHONPATH=$PYTHONPATH:$PYCAFFE_DIR:$WORK_DIR

python solve.py --solver "${PROTOTXT_DIR}"/solver.prototxt --weights "${SEGNET_TUTORIAL_DIR}"/segnet_pascal.caffemodel --snapshot_dir "${MODELS_DIR}"/Training/"${exp_name}" --nepoch $num_epochs

python compute_bn_statistics.py --train_model "${PROTOTXT_DIR}"/train.prototxt --weights_dir "${SEGNET_TUTORIAL_DIR}"/Models/Training/"${exp_name}" --out_dir "${SEGNET_TUTORIAL_DIR}"/Models/Inference/"${exp_name}"

python test_segnet.py --model "${PROTOTXT_DIR}"/inference_train.prototxt --weights_dir "${SEGNET_TUTORIAL_DIR}"/Models/Inference/"${exp_name}" --models_dir "${SEGNET_TUTORIAL_DIR}"/Models/Training/"${exp_name}" --save_dir $RESULTS_DIR/"${exp_name}"/train --test_imgs "${ROADS_DIR}"/train

python plot_results.py --save_dir $RESULTS_DIR/"${exp_name}"/train --results_dir $RESULTS_DIR/"${exp_name}"/train

