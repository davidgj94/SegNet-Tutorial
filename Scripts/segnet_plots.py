import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import shutil
import argparse
import pdb
import itertools
import numpy as np

def get_class_score(scores, idx):
    return [el[idx] for el in scores]

def plot_results(acc, per_class_acc, per_class_iu, classes, save_dir):
    
    num_epochs = range(len(acc))
    
    # Global accuracy
    plt.figure()
    plt.plot(num_epochs, acc)
    plt.title('Global Accuracy')
    plt.grid()
    plt.savefig(os.path.join(save_dir,'global_acc.png'))

    # Per-Class accuracy

    plt.figure()
    for index,_ in enumerate(classes):
        plt.plot(num_epochs, get_class_score(per_class_acc, index))

    plt.title('Per-Class accuracy')
    plt.legend(classes)
    plt.grid()
    plt.savefig(os.path.join(save_dir,'per_class_acc.png'))
    
    # Per-Class Iu

    plt.figure()
    for index,_ in enumerate(classes):
        plt.plot(num_epochs, get_class_score(per_class_iu, index))

    plt.title('Per-Class IU')
    plt.legend(classes)
    plt.grid()
    plt.savefig(os.path.join(save_dir,'per_class_iu.png'))

    # Mean Iu
    plt.figure()
    plt.plot(num_epochs, np.mean(per_class_iu, axis=1))
    plt.title('Mean IU')
    plt.grid()
    plt.savefig(os.path.join(save_dir,'mean_iu.png'))

def plot_results_vs(acc_train, acc_val, per_class_acc_train, per_class_acc_val, per_class_iu_train, per_class_iu_val, classes, save_dir):
    
    num_epochs = range(len(acc_train))
    
    # Global accuracy

    plt.figure()
    plt.plot(num_epochs, acc_train)
    plt.plot(num_epochs, acc_val)
    plt.legend(['train', 'val'])
    plt.title('Global Accuracy')
    plt.grid()
    plt.savefig(os.path.join(save_dir,'global_acc.png'))

    # Per-Class accuracy

    for index, _class in enumerate(classes):
        plt.figure()
        plt.plot(num_epochs, get_class_score(per_class_acc_train, index))
        plt.plot(num_epochs, get_class_score(per_class_acc_val, index))
        plt.title('Class {} accuracy'.format(_class))
        plt.legend(['train', 'val'])
        plt.grid()
        plt.savefig(os.path.join(save_dir,'per_class_acc_{}.png'.format(_class)))
    
        
    # Per-Class iu

    for index, _class in enumerate(classes):
        plt.figure()
        plt.plot(num_epochs, get_class_score(per_class_iu_train, index))
        plt.plot(num_epochs, get_class_score(per_class_iu_val, index))
        plt.title('Class {} IU'.format(_class))
        plt.legend(['train', 'val'])
        plt.grid()
        plt.savefig(os.path.join(save_dir,'per_class_iu_{}.png'.format(_class)))
    

    # Mean iu

    plt.figure()
    plt.plot(num_epochs, np.mean(per_class_iu_train, axis=1))
    plt.plot(num_epochs, np.mean(per_class_iu_val, axis=1))
    plt.title('Mean IU')
    plt.legend(['train', 'val'])
    plt.grid()
    plt.savefig(os.path.join(save_dir, 'mean_iu.png'))


def plot_confusion_matrix(cm, classes, save_path,
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
    plt.savefig(save_path)
    
