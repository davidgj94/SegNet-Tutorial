
ROOT_DIR=/home/david/projects
SEGNET_TUTORIAL_DIR="${ROOT_DIR}"/SegNet-Tutorial
CAFFE_DIR="${ROOT_DIR}"/caffe-segnet-cudnn5/build/tools"

./"${CAFFE_DIR}"/caffe train -gpu 0 -solver "${SEGNET_TUTORIAL_DIR}"/Models/segnet_solver.prototxt -weights ${SEGNET_TUTORIAL_DIR}/segnet_pascal.caffemodel

# python "${SEGNET_TUTORIAL_DIR}"/Scripts/compute_bn_statistics.py "${SEGNET_TUTORIAL_DIR}"/Models/segnet_train.prototxt "${SEGNET_TUTORIAL_DIR}"/Models/Training/segnet_iter_10000.caffemodel "${SEGNET_TUTORIAL_DIR}"/Models/Inference/
# 
# python "${SEGNET_TUTORIAL_DIR}"/Scripts/test_segmentation_roads.py --model "${SEGNET_TUTORIAL_DIR}"/Models/segnet_inference.prototxt --weights "${SEGNET_TUTORIAL_DIR}"/Models/Inference/test_weights.caffemodel 