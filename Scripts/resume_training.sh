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

# FASE 2 (opcional): Seguimos entrenando por el último snapshot, calculamos bn statistics, probamos en el train set y visualizamos

exp_name=$1
iter=$2

"${CAFFE_DIR}"/caffe train -gpu 0 -solver "${SEGNET_TUTORIAL_DIR}"/Models/segnet_solver.prototxt -snapshot "${SEGNET_TUTORIAL_DIR}"/Models/Training/"${exp_name}"/snapshot_iter_"${iter}".solverstate 

python compute_bn_statistics.py --train_model "${SEGNET_TUTORIAL_DIR}"/Models/segnet_train.prototxt --weights_dir "${SEGNET_TUTORIAL_DIR}"/Models/Training/"${exp_name}" --out_dir "${SEGNET_TUTORIAL_DIR}"/Models/Inference/"${exp_name}"

python test_segnet.py --model "${SEGNET_TUTORIAL_DIR}"/Models/segnet_inference_train.prototxt --weights_dir "${SEGNET_TUTORIAL_DIR}"/Models/Inference/"${exp_name}" --models_dir "${SEGNET_TUTORIAL_DIR}"/Models/Training/"${exp_name}" --save_dir $RESULTS_DIR/"${exp_name}"/train --test_imgs "${ROADS_DIR}"/train

python plot_results.py --save_dir $RESULTS_DIR/"${exp_name}"/train --results_dir $RESULTS_DIR/"${exp_name}"/train

############################################################################################################################################################################################################
