import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import shutil
import argparse
import numpy as np
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



def print_test_results_(hist, acc, per_class_acc, per_class_iu, class_names, save_dir, iteration):
    
    print '>>>','overall accuracy', acc

    for idx, class_acc in enumerate(per_class_acc):
        print '>>>', 'Class {} accuracy:'.format(idx), class_acc
        
    for idx, class_iu in enumerate(per_class_iu):
        print '>>>', 'Class {} iu:'.format(idx), class_iu
    
    print '>>>','confusion matrix'
    print hist / hist.sum(1)[:, np.newaxis]
    
    plt.figure()
    plot_confusion_matrix(hist, class_names)
    plt.savefig(os.path.join(save_dir,'conf_matrix_{}.png'.format(iteration)))


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, required=True)
parser.add_argument('--iteration', type=int, required=True)
args = parser.parse_args()

results_path_test = '../results/{}/test/results_{}.p'.format(args.exp_name,args.iteration)

if os.path.exists(results_path_test):
    with open(results_path_test, 'rb') as f:
        hist, acc, per_class_acc, per_class_iu = pickle.load(f)
        class_names = ['background', 'disconn', 'other', 'disconn-short']
        print_test_results_(hist[0], acc[0], per_class_acc[0], per_class_iu[0], class_names, '../results/{}/test/'.format(args.exp_name), args.iteration)
else:
    print 'No hay resultados!!!'
    
