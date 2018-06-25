import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import shutil
import argparse

def get_class_score(scores, idx):
    return [el[idx] for el in scores]

def plot_results_(acc, per_class_acc, per_class_iu):
    
    num_epochs = range(len(acc))
    
    # Global accuracy
    plt.figure()
    plt.plot(num_epochs, acc)
    plt.title('Global Accuracy')
    #plt.show()
    plt.grid()
    plt.savefig('../tmp/global_acc.png')

    # Per-Class accuracy
    plt.figure()
    plt.plot(num_epochs, get_class_score(per_class_acc, 0))
    plt.plot(num_epochs, get_class_score(per_class_acc, 1))
    plt.plot(num_epochs, get_class_score(per_class_acc, 2))
    plt.plot(num_epochs, get_class_score(per_class_acc, 3))
    plt.title('Per-Class accuracy')
    plt.legend(['0', '1', '2', '3'])
    plt.grid()
    plt.savefig('../tmp/per_class_acc.png')
    #plt.show()
    
    # Per-Class Iu
    plt.figure()
    plt.plot(num_epochs, get_class_score(per_class_iu, 0))
    plt.plot(num_epochs, get_class_score(per_class_iu, 1))
    plt.plot(num_epochs, get_class_score(per_class_iu, 2))
    plt.plot(num_epochs, get_class_score(per_class_iu, 3))
    plt.title('Per-Class Iu')
    plt.legend(['0', '1', '2', '3'])
    plt.grid()
    plt.savefig('../tmp/per_class_iu.png')
    #plt.show()

def plot_results_vs(acc_train, acc_val, per_class_acc_train, per_class_acc_val, per_class_iu_train, per_class_iu_val):
    
    num_epochs = range(len(acc_train))
    
    # Global accuracy
    plt.figure()
    plt.plot(num_epochs, acc_train)
    plt.plot(num_epochs, acc_val)
    plt.legend(['train', 'val'])
    plt.title('Global Accuracy')
    plt.grid()
    plt.savefig('../tmp/global_acc.png')
    #plt.show()

    # Per-Class accuracy
    plt.figure()
    plt.plot(num_epochs, get_class_score(per_class_acc_train, 0))
    plt.plot(num_epochs, get_class_score(per_class_acc_val, 0))
    plt.title('Class 0 accuracy')
    plt.legend(['train', 'val'])
    plt.grid()
    plt.savefig('../tmp/per_class_acc_0.png')
    #plt.show()
    
    plt.figure()
    plt.plot(num_epochs, get_class_score(per_class_acc_train, 1))
    plt.plot(num_epochs, get_class_score(per_class_acc_val, 1))
    plt.title('Class 1 accuracy')
    plt.legend(['train', 'val'])
    plt.grid()
    plt.savefig('../tmp/per_class_acc_1.png')
    #plt.show()
    
    plt.figure()
    plt.plot(num_epochs, get_class_score(per_class_acc_train, 2))
    plt.plot(num_epochs, get_class_score(per_class_acc_val, 2))
    plt.title('Class 2 accuracy')
    plt.legend(['train', 'val'])
    plt.savefig('../tmp/per_class_acc_2.png')
    plt.grid()
    #plt.show()
    
    plt.figure()
    plt.plot(num_epochs, get_class_score(per_class_acc_train, 3))
    plt.plot(num_epochs, get_class_score(per_class_acc_val, 3))
    plt.title('Class 3 accuracy')
    plt.legend(['train', 'val'])
    plt.grid()
    plt.savefig('../tmp/per_class_acc_3.png')
    #plt.show()

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default='')
parser.add_argument('-vs', action='store_true')
parser.add_argument('--train_dir', type=str, default='')
parser.add_argument('--val_dir', type=str, default='')
args = parser.parse_args()

if os.path.exists('../tmp'):
    shutil.rmtree('../tmp', ignore_errors=True)
os.makedirs('../tmp')

if args.vs:
    if args.train_dir and args.val_dir:
        results_path_train = os.path.join(args.train_dir,'results.p')
        results_path_val = os.path.join(args.val_dir,'results.p')
        if os.path.exists(results_path_train) and os.path.exists(results_path_val):
            with open(results_path_train, 'rb') as f:
                _, acc_train, per_class_acc_train, per_class_iu_train = pickle.load(f)
            with open(results_path_val, 'rb') as f:
                _, acc_val, per_class_acc_val, per_class_iu_val = pickle.load(f)    
            plot_results_vs(acc_train, acc_val, per_class_acc_train, per_class_acc_val, per_class_iu_train, per_class_iu_val)
        else:
            print 'No hay resultados!!!'
else:
    results_path = os.path.join(args.save_dir,'results.p')
    if os.path.exists(results_path):
        with open(results_path, 'rb') as f:
            _, acc, per_class_acc, per_class_iu = pickle.load(f)
            plot_results_(acc, per_class_acc, per_class_iu)
    else:
        print 'No hay resultados!!!'
    
