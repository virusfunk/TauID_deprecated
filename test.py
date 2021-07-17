import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from dataset import TauDatasetTest
import tensorflow as tf
import matplotlib.pyplot as plt
from model import get_model
import sklearn.metrics
import itertools
from matplotlib.lines import Line2D

def plot_confusion_matrix(cm, target_names=None, cmap=None, normalize=True, labels=True, title='Confusion matrix'):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    plt.figure(figsize=(12, 10),dpi=120)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)
    
    if labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}'.format(accuracy, misclass))
    plt.savefig('cf_dropout.png')


def plot_multilabel_auc(model, x_test, y_test, labels, n_classes=6):
    predict_baseline    = model.predict(x_test)
    test_score_baseline = model.evaluate(x_test, y_test)

    n_classes = 6
    print('Plotting ROC for labels {}'.format(labels))

    df = pd.DataFrame()
    fpr  = {}
    tpr  = {}
    auc1 = {}
    %matplotlib inline
    colors  = ['#67001f','#b2182b','#d6604d','#f4a582','#fddbc7','#d1e5f0','#92c5de','#4393c3','#2166ac','#053061']
    fig, ax = plt.subplots(figsize=(10, 10))
    for i, label in enumerate(labels):

        df[label] = y_test[:,i]
        df[label + '_pred'] = predict_baseline[:,i]
        fpr[label], tpr[label], threshold = metrics.roc_curve(df[label],df[label+'_pred'])
        auc1[label] = metrics.auc(fpr[label], tpr[label])    

        plt.plot(1-fpr[label],tpr[label],label=r'{}, AUC = {:.1f}%'.format(label,auc1[label]*100), linewidth=1.5,c=colors[i],linestyle='solid')

    plt.ylabel("True Positive Rate")
    plt.xlabel("1-False Positive Rate")
    plt.legend(loc='lower left')
    lines = [Line2D([0], [0], ls='-'),
             Line2D([0], [0], ls='--')]
    plt.savefig('auc.png')


def main():
    model = get_model()
    model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['acc'])
    model.load_weights('best_weights.hdf5')
    
    testset = TauDatasetTest(6*20)
    tf.keras.backend.set_learning_phase(0)

    cf = np.zeros((6,6))
    for i in range(len(testset)):
        if i % 10 == 0:
            print(f'{i+1}/{len(testset)}')
        x_test, y_test = testset[i]
        y_pred = model.predict(x_test)
        matrix = sklearn.metrics.confusion_matrix(y_test.argmax(axis=1), 
                                                  y_pred.argmax(axis=1))
        cf += matrix

    labels = [r'Z->$\tau^{\mp}$->$\pi^{\mp}\nu$',
                  r'Z->$\tau$->$\pi^{\mp}$$\pi^{\pm}$$\pi^{\mp}\nu$',
                  r'Z->$\tau$->$\pi^{\mp}$$\pi^{\pm}$$\pi^{\mp}$$\pi^{0}\nu$',
                  r'Z->$\tau$->$\pi^{\mp}$$\pi^{0}\nu$',
                  r'Z->$\tau$->$\pi^{\mp}$$\pi^{0}$$\pi^{0}\nu$',
                  r'Z->q$\bar{q}$']
        
    plot_confusion_matrix(cf, target_names=labels)
    
    testset = TauDatasetTest(6*3000)
    x_test, y_test = testset[0]
    plot_multilabel_auc(model, x_test=x_test, y_test=y_test, labels=labels)
    
    