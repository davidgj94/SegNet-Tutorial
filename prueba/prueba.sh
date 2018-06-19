ROOT_DIR=/home/david/projects
SEGNET_TUTORIAL_DIR="${ROOT_DIR}"/SegNet-Tutorial
CAFFE_DIR="${ROOT_DIR}"/caffe-segnet-cudnn5/build/tools

python3 road_masks_script.py $1 $2
python prueba.py --model "${SEGNET_TUTORIAL_DIR}"/Models/segnet_inference_prueba.prototxt --weights "${SEGNET_TUTORIAL_DIR}"/Models/Inference/test_weights.caffemodel
