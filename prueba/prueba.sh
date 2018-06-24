WORK_DIR=$(pwd)
cd ..
cd ..
ROOT_DIR=$(pwd)
SEGNET_TUTORIAL_DIR="${ROOT_DIR}"/SegNet-Tutorial
CAFFE_DIR="${ROOT_DIR}"/caffe-segnet-cudnn5/build/tools
PYCAFFE_DIR="${ROOT_DIR}"/caffe-segnet-cudnn5/python
ROADS_DIR="${SEGNET_TUTORIAL_DIR}"/roads/ROADS
RESULTS_DIR="${SEGNET_TUTORIAL_DIR}"/results
export PYTHONPATH=$PYTHONPATH:$PYCAFFE_DIR
cd $WORK_DIR

python3 road_masks_script.py $1 $2
python prueba.py --model "${SEGNET_TUTORIAL_DIR}"/Models/segnet_inference_prueba.prototxt --weights "${SEGNET_TUTORIAL_DIR}"/Models/Inference/prueba_all/snapshot_iter_1810/test_weights.caffemodel
