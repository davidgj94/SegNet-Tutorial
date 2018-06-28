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
iter=$3

python test_segnet.py --model "${PROTOTXT_DIR}"/inference_test.prototxt --weights_dir "${SEGNET_TUTORIAL_DIR}"/Models/Inference/"${exp_name}" --models_dir "${SEGNET_TUTORIAL_DIR}"/Models/Training/"${exp_name}" --save_dir $RESULTS_DIR/"${exp_name}"/test --test_imgs "${ROADS_DIR}"/test --iteration "${iter}"

python test_segmentation_roads.py --model "${PROTOTXT_DIR}"/inference_test.prototxt --weights "${SEGNET_TUTORIAL_DIR}"/Models/Inference/"${exp_name}"/snapshot_iter_"${iter}"/test_weights.caffemodel --save_dir $RESULTS_DIR/"${exp_name}"/test/blended/ --imgs_txt "${ROADS_DIR}"/test.txt

python print_test_results.py --exp_name $exp_name

#######################################################################################################################################################################
