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
iter=$3

cd $WORK_DIR

export PYTHONPATH=$PYTHONPATH:$PYCAFFE_DIR

python test_segnet.py --model "${PROTOTXT_DIR}"/inference_test.prototxt --weights_dir "${SEGNET_TUTORIAL_DIR}"/Models/Inference/"${exp_name}" --models_dir "${SEGNET_TUTORIAL_DIR}"/Models/Training/"${exp_name}" --save_dir $RESULTS_DIR/"${exp_name}"/test --test_imgs "${ROADS_DIR}"/test --iteration "${iter}"

mkdir -p $RESULTS_DIR/"${exp_name}"/test/blended_"${iter}"/

python test_segmentation_roads.py --model "${PROTOTXT_DIR}"/inference_test.prototxt --weights "${SEGNET_TUTORIAL_DIR}"/Models/Inference/"${exp_name}"/snapshot_iter_"${iter}"/test_weights.caffemodel --save_dir $RESULTS_DIR/"${exp_name}"/test/blended_"${iter}"/ --imgs_txt "${ROADS_DIR}"/test.txt

python print_test_results.py --exp_name $exp_name --iteration $iter
