WORK_DIR=$(pwd)
cd ..
cd ..
ROOT_DIR=$(pwd)
SEGNET_TUTORIAL_DIR="${ROOT_DIR}"/SegNet-Tutorial
ROADS_DIR="${SEGNET_TUTORIAL_DIR}"/roads/ROADS/test
CAFFE_DIR="${ROOT_DIR}"/caffe-segnet-cudnn5/build/tools
RESULTS_DIR="${SEGNET_TUTORIAL_DIR}"/results/$1
ROADS_DIR="${SEGNET_TUTORIAL_DIR}"/roads/ROADS/val

cd $WORK_DIR
python test_segnet.py --model "${SEGNET_TUTORIAL_DIR}"/Models/segnet_inference.prototxt --weights "${SEGNET_TUTORIAL_DIR}"/Models/Inference/test_weights.caffemodel --save_dir $RESULTS_DIR --test_imgs $ROADS_DIR
