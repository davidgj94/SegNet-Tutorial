import os
import matplotlib.pyplot as plt
import pickle

def get_class_score(scores, idx):
    return [el[idx] for el in scores]

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, required=True)
args = parser.parse_args()


results_path = os.path.join(args.save_dir,'results.p')
if os.path.exists(args.results_path):
    with open(results_path, 'rb') as f:
        hist, acc, per_class_acc, per_class_iu = pickle.load(f)
        plot_results_(hist, acc, per_class_acc, per_class_iu)
else:
    print 'No hay resultados!!!'
    
def plot_results_(hist, acc, per_class_acc, per_class_iu):
    
    num_epochs = range(len(acc))
    # Import arguments
    
    # Global accuracy
    plt.plot(num_epochs, acc)
    plt.title('Global Accuracy')
    plt.show()

    # Per-Class accuracy
    plt.plot(num_epochs, get_class_score(per_class_acc, 0))
    plt.plot(num_epochs, get_class_score(per_class_acc, 1))
    plt.plot(num_epochs, get_class_score(per_class_acc, 2))
    plt.plot(num_epochs, get_class_score(per_class_acc, 3))
    plt.title('Per-Class accuracy')
    plt.legend(['0', '1', '2', '3'])
    plt.grid()
    plt.show()
    
    # Per-Class Iu
    plt.plot(num_epochs, get_class_score(per_class_iu, 0))
    plt.plot(num_epochs, get_class_score(per_class_iu, 1))
    plt.plot(num_epochs, get_class_score(per_class_iu, 2))
    plt.plot(num_epochs, get_class_score(per_class_iu, 3))
    plt.title('Per-Class Iu')
    plt.legend(['0', '1', '2', '3'])
    plt.grid()
    plt.show()
