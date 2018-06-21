WORK_DIR=$(pwd)
cd ..
cd ..
ROOT_DIR=$(pwd)
SEGNET_TUTORIAL_DIR="${ROOT_DIR}"/SegNet-Tutorial
CAFFE_DIR="${ROOT_DIR}"/caffe-segnet-cudnn5/build/tools
PYCAFFE_DIR="${ROOT_DIR}"/caffe-segnet-cudnn5/python
export PYTHONPATH=$PYTHONPATH:$PYCAFFE_DIR
cd $WORK_DIR

"${CAFFE_DIR}"/caffe train -gpu 0 -solver "${SEGNET_TUTORIAL_DIR}"/Models/segnet_solver.prototxt -weights "${SEGNET_TUTORIAL_DIR}"/segnet_pascal.caffemodel -iterations $1

"${CAFFE_DIR}"/caffe train -gpu 0 -solver "${SEGNET_TUTORIAL_DIR}"/Models/segnet_solver.prototxt -snapshot "${SEGNET_TUTORIAL_DIR}"/Training/$1/$1_iter_$2.solverstate -iterations $3

# python "${SEGNET_TUTORIAL_DIR}"/Scripts/compute_bn_statistics.py "${SEGNET_TUTORIAL_DIR}"/Models/segnet_train.prototxt "${SEGNET_TUTORIAL_DIR}"/Models/Training/$1/$1_iter_$2.solverstate.caffemodel "${SEGNET_TUTORIAL_DIR}"/Models/Inference/$1

# python "${SEGNET_TUTORIAL_DIR}"/Scripts/test_segmentation_roads.py --model "${SEGNET_TUTORIAL_DIR}"/Models/segnet_inference_train.prototxt --weights "${SEGNET_TUTORIAL_DIR}"/Models/Inference/test_weights.caffemodel --save_dir "${SEGNET_TUTORIAL_DIR}"/results3 --iter 377
# 
# python "${SEGNET_TUTORIAL_DIR}"/Scripts/test_segmentation_roads.py --model "${SEGNET_TUTORIAL_DIR}"/Models/segnet_inference_val.prototxt --weights "${SEGNET_TUTORIAL_DIR}"/Models/Inference/test_weights.caffemodel --save_dir "${SEGNET_TUTORIAL_DIR}"/results3 --iter 377
# 
# python "${SEGNET_TUTORIAL_DIR}"/Scripts/test_segmentation_roads.py --model "${SEGNET_TUTORIAL_DIR}"/Models/segnet_inference_test.prototxt --weights "${SEGNET_TUTORIAL_DIR}"/Models/Inference/test_weights.caffemodel --save_dir "${SEGNET_TUTORIAL_DIR}"/results3 --iter 377

# python "${SEGNET_TUTORIAL_DIR}"/Scripts/test_segmentation_camvid.py --model "${SEGNET_TUTORIAL_DIR}"/Models/segnet_inference.prototxt --weights "${SEGNET_TUTORIAL_DIR}"/Models/Inference/test_weights.caffemodel --iter 377
