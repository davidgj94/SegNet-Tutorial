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

# FASE 3: Probamos en el val set y visualizamos los resultados y los comparamos con el train set

exp_name=$1

python test_segnet.py --model "${SEGNET_TUTORIAL_DIR}"/Models/segnet_inference_val.prototxt --weights_dir "${SEGNET_TUTORIAL_DIR}"/Models/Inference/"${exp_name}" --models_dir "${SEGNET_TUTORIAL_DIR}"/Models/Training/"${exp_name}" --save_dir $RESULTS_DIR/"${exp_name}"/val --test_imgs "${ROADS_DIR}"/val

python plot_results.py -vs --train_dir $RESULTS_DIR/"${exp_name}"/train --val_dir $RESULTS_DIR/"${exp_name}"/val

############################################################################################################################################################################################################
