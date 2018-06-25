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

# FASE 5: Mostramos resultados

exp_name=$1

python print_test_results.py --exp_name $exp_name

############################################################################################################################################################################################################
