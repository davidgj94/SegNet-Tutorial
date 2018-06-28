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

# FASE 1: Empezamos a entrenar, calculamos bn statistics, probamos en el train set y visualizamos

option=$1
case ${option} in 
   -by) 
		PROTOTXT_DIR="${SEGNET_TUTORIAL_DIR}"/Models/bayesian
		;; 
   -s) 
		PROTOTXT_DIR="${SEGNET_TUTORIAL_DIR}"/Models/segnet
		;; 
   -ba) 
		PROTOTXT_DIR="${SEGNET_TUTORIAL_DIR}"/Models/segnet_balanced
		;; 
   *)  
      	echo "`basename ${0}`:usage: [-by bayesian segnet] | [-s segnet] [-ba balanced segnet]" 
      	exit 1 # Command to come out of the program with status 1
      	;; 
esac

exp_name=$2

#python create_exp_dirs.py --exp_name "${exp_name}"

./change_snapshot_prefix.sh "${PROTOTXT_DIR}" "${exp_name}"

exit 1

"${CAFFE_DIR}"/caffe train -gpu 0 -solver "${PROTOTXT_DIR}"/solver.prototxt -weights "${SEGNET_TUTORIAL_DIR}"/segnet_pascal.caffemodel

python compute_bn_statistics.py --train_model "${PROTOTXT_DIR}"/train.prototxt --weights_dir "${SEGNET_TUTORIAL_DIR}"/Models/Training/"${exp_name}" --out_dir "${SEGNET_TUTORIAL_DIR}"/Models/Inference/"${exp_name}"

python test_segnet.py --model "${PROTOTXT_DIR}"/inference_train.prototxt --weights_dir "${SEGNET_TUTORIAL_DIR}"/Models/Inference/"${exp_name}" --models_dir "${SEGNET_TUTORIAL_DIR}"/Models/Training/"${exp_name}" --save_dir $RESULTS_DIR/"${exp_name}"/train --test_imgs "${ROADS_DIR}"/train

python plot_results.py --save_dir $RESULTS_DIR/"${exp_name}"/train --results_dir $RESULTS_DIR/"${exp_name}"/train

############################################################################################################################################################################################################
