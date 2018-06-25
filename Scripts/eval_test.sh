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


############################################################################################################################################################################################################

# FASE 4: Probamos en el test set los mejores parametros en validación y guardamos las imagenes blended

exp_name=$1
iter=$2

python test_segnet.py --model "${SEGNET_TUTORIAL_DIR}"/Models/segnet_inference_test.prototxt --weights_dir "${SEGNET_TUTORIAL_DIR}"/Models/Inference/"${exp_name}" --models_dir "${SEGNET_TUTORIAL_DIR}"/Models/Training/"${exp_name}" --save_dir $RESULTS_DIR/"${exp_name}"/test --test_imgs "${ROADS_DIR}"/test --iteration "${iter}"

python test_segmentation_roads.py --model "${SEGNET_TUTORIAL_DIR}"/Models/segnet_inference_test.prototxt --weights "${SEGNET_TUTORIAL_DIR}"/Models/Inference/"${exp_name}"/snapshot_iter_"${iter}"/test_weights.caffemodel --save_dir $RESULTS_DIR/"${exp_name}"/test/blended/ --imgs_txt "${ROADS_DIR}"/test.txt

############################################################################################################################################################################################################
