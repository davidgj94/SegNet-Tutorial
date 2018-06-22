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

# "${CAFFE_DIR}"/caffe train -gpu 0 -solver "${SEGNET_TUTORIAL_DIR}"/Models/segnet_solver.prototxt -weights "${SEGNET_TUTORIAL_DIR}"/segnet_pascal.caffemodel -iterations $1

# python compute_bn_statistics.py --train_model "${SEGNET_TUTORIAL_DIR}"/Models/segnet_train.prototxt --weights_dir "${SEGNET_TUTORIAL_DIR}"/Models/Training/$1 --out_dir "${SEGNET_TUTORIAL_DIR}"/Models/Inference/$1

# python test_segnet.py --model "${SEGNET_TUTORIAL_DIR}"/Models/segnet_inference_train.prototxt --weights_dir "${SEGNET_TUTORIAL_DIR}"/Models/Inference/$1 --models_dir "${SEGNET_TUTORIAL_DIR}"/Models/Training/$1 --save_dir $RESULTS_DIR/$1/train --test_imgs "${ROADS_DIR}"/train

# python plot_results.py --save_dir $RESULTS_DIR/$1/train

############################################################################################################################################################################################################

# FASE 2 (opcional): Seguimos entrenando por el último snapshot, calculamos bn statistics, probamos en el train set y visualizamos

# "${CAFFE_DIR}"/caffe train -gpu 0 -solver "${SEGNET_TUTORIAL_DIR}"/Models/segnet_solver.prototxt -snapshot "${SEGNET_TUTORIAL_DIR}"/Training/$1/$1_iter_$2.solverstate -iterations $3

# python compute_bn_statistics.py --train_model "${SEGNET_TUTORIAL_DIR}"/Models/segnet_train.prototxt --weights_dir "${SEGNET_TUTORIAL_DIR}"/Models/Training/$1 --out_dir "${SEGNET_TUTORIAL_DIR}"/Models/Inference/$1 --last_iter $2

# python test_segnet.py --model "${SEGNET_TUTORIAL_DIR}"/Models/segnet_inference_train.prototxt --weights_dir "${SEGNET_TUTORIAL_DIR}"/Models/Inference/$1 --models_dir "${SEGNET_TUTORIAL_DIR}"/Models/Training/$1 --save_dir $RESULTS_DIR/$1/train --test_imgs "${ROADS_DIR}"/train

# python plot_results.py --save_dir $RESULTS_DIR/$1/train

############################################################################################################################################################################################################

# FASE 3: Probamos en el val set y visualizamos los resultados y los comparamos con el train set

# python test_segnet.py --model "${SEGNET_TUTORIAL_DIR}"/Models/segnet_inference_val.prototxt --weights_dir "${SEGNET_TUTORIAL_DIR}"/Models/Inference/$1 --models_dir "${SEGNET_TUTORIAL_DIR}"/Models/Training/$1 --save_dir $RESULTS_DIR/$1/val --test_imgs "${ROADS_DIR}"/val

# python plot_results.py --save_dir $RESULTS_DIR/$1/val

# python plot_results.py -vs --train_dir $RESULTS_DIR/$1/train --val_dir $RESULTS_DIR/$1/val

############################################################################################################################################################################################################

# FASE 4: Probamos en el test set los mejores parametros en validaciń y guardamos las imagenes blended

python test_segnet.py --model "${SEGNET_TUTORIAL_DIR}"/Models/segnet_inference_test.prototxt --weights_dir "${SEGNET_TUTORIAL_DIR}"/Models/Inference/$1 --models_dir "${SEGNET_TUTORIAL_DIR}"/Models/Training/$1 --save_dir $RESULTS_DIR/$1/test --test_imgs "${ROADS_DIR}"/test --iteration $2

# python test_segmentation_roads.py --model "${SEGNET_TUTORIAL_DIR}"/Models/segnet_inference_test.prototxt --weights "${SEGNET_TUTORIAL_DIR}"/Models/Inference/$1/snapshot_iter_$2/test_weights.caffemodel --save_dir $RESULTS_DIR/$1/test/blended/ --imgs_txt "${ROADS_DIR}"/test.txt
