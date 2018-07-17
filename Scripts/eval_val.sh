WORK_DIR=$(pwd)
cd ..
cd ..
ROOT_DIR=$(pwd)
SEGNET_TUTORIAL_DIR="${ROOT_DIR}"/SegNet-Tutorial
CAFFE_DIR="${ROOT_DIR}"/caffe-segnet-cudnn5/build/tools
PYCAFFE_DIR="${ROOT_DIR}"/caffe-segnet-cudnn5/python
ROADS_DIR="${SEGNET_TUTORIAL_DIR}"/roads/ROADS
RESULTS_DIR="${SEGNET_TUTORIAL_DIR}"/results
MODELS_DIR="${SEGNET_TUTORIAL_DIR}"/Models
PROTOTXT_DIR="${MODELS_DIR}"/$1

exp_name=$2

mkdir -p "${RESULTS_DIR}"/"${exp_name}"/val

cd $WORK_DIR

export PYTHONPATH=$PYTHONPATH:$PYCAFFE_DIR

python test_segnet.py --model "${PROTOTXT_DIR}"/inference_val.prototxt --weights_dir "${SEGNET_TUTORIAL_DIR}"/Models/Inference/"${exp_name}" --models_dir "${SEGNET_TUTORIAL_DIR}"/Models/Training/"${exp_name}" --save_dir $RESULTS_DIR/"${exp_name}"/val --test_imgs "${ROADS_DIR}"/val

python plot_results.py -vs --train_dir $RESULTS_DIR/"${exp_name}"/train --val_dir $RESULTS_DIR/"${exp_name}"/val --save_dir $RESULTS_DIR/"${exp_name}"/val
