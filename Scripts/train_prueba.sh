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

# "${CAFFE_DIR}"/caffe train -gpu 0 -solver "${SEGNET_TUTORIAL_DIR}"/Models/segnet_solver.prototxt -weights "${SEGNET_TUTORIAL_DIR}"/segnet_pascal.caffemodel -iterations $1

# "${CAFFE_DIR}"/caffe train -gpu 0 -solver "${SEGNET_TUTORIAL_DIR}"/Models/segnet_solver.prototxt -snapshot "${SEGNET_TUTORIAL_DIR}"/Training/$1/$1_iter_$2.solverstate -iterations $3

# python compute_bn_statistics.py --train_model "${SEGNET_TUTORIAL_DIR}"/Models/segnet_train.prototxt --weights_dir "${SEGNET_TUTORIAL_DIR}"/Models/Training/$1 --out_dir "${SEGNET_TUTORIAL_DIR}"/Models/Inference/$1 --last_iter $2

# python test_segnet.py --model "${SEGNET_TUTORIAL_DIR}"/Models/segnet_inference_train.prototxt --weights_dir "${SEGNET_TUTORIAL_DIR}"/Models/Inference/$1 --models_dir "${SEGNET_TUTORIAL_DIR}"/Models/Training/$1 --save_dir $RESULTS_DIR/$1/train --test_imgs "${ROADS_DIR}"/train

# python plot_results.py --save_dir $RESULTS_DIR/$1/train

# python test_segnet.py --model "${SEGNET_TUTORIAL_DIR}"/Models/segnet_inference_val.prototxt --weights_dir "${SEGNET_TUTORIAL_DIR}"/Models/Inference/$1 --models_dir "${SEGNET_TUTORIAL_DIR}"/Models/Training/$1 --save_dir $RESULTS_DIR/$1/val --test_imgs "${ROADS_DIR}"/val

# python plot_results.py --save_dir $RESULTS_DIR/$1/val

python plot_results.py -vs --train_dir $RESULTS_DIR/$1/train --val_dir $RESULTS_DIR/$1/val

# python test_segnet.py --model "${SEGNET_TUTORIAL_DIR}"/Models/segnet_inference_test.prototxt --weights_dir "${SEGNET_TUTORIAL_DIR}"/Models/Inference/$1 --models_dir "${SEGNET_TUTORIAL_DIR}"/Models/Training/$1 --save_dir $RESULTS_DIR/$1/test --test_imgs "${ROADS_DIR}"/test

# python plot_results.py --save_dir $RESULTS_DIR/$1/test


# python "${SEGNET_TUTORIAL_DIR}"/Scripts/test_segmentation_roads.py --model "${SEGNET_TUTORIAL_DIR}"/Models/segnet_inference_train.prototxt --weights "${SEGNET_TUTORIAL_DIR}"/Models/Inference/test_weights.caffemodel --save_dir "${SEGNET_TUTORIAL_DIR}"/results3 --iter 377
# 
# python "${SEGNET_TUTORIAL_DIR}"/Scripts/test_segmentation_roads.py --model "${SEGNET_TUTORIAL_DIR}"/Models/segnet_inference_val.prototxt --weights "${SEGNET_TUTORIAL_DIR}"/Models/Inference/test_weights.caffemodel --save_dir "${SEGNET_TUTORIAL_DIR}"/results3 --iter 377
# 
# python "${SEGNET_TUTORIAL_DIR}"/Scripts/test_segmentation_roads.py --model "${SEGNET_TUTORIAL_DIR}"/Models/segnet_inference_test.prototxt --weights "${SEGNET_TUTORIAL_DIR}"/Models/Inference/test_weights.caffemodel --save_dir "${SEGNET_TUTORIAL_DIR}"/results3 --iter 377

# python "${SEGNET_TUTORIAL_DIR}"/Scripts/test_segmentation_camvid.py --model "${SEGNET_TUTORIAL_DIR}"/Models/segnet_inference.prototxt --weights "${SEGNET_TUTORIAL_DIR}"/Models/Inference/test_weights.caffemodel --iter 377
