

#################
#### IMPORTS ####
#################

# General Utilities
import math
import random
import numpy as np
import uproot

# Keras and TensorFlow
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
import tensorflow as tf

# Sklearn Utilities
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Plotting Utilities
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import matplotlib.pyplot as plt





######################
#### INITIALIZING ####
######################

### Initializing Input File Information

INPUT_FILES_INFO = {}

INPUT_FILES_INFO["bkg1"] = {"fname" : "../rootfiles/bkg1_Events.root",
                            "encoding" : [1,0,0],
                            "label" : "BKG 1"}
INPUT_FILES_INFO["bkg2"] = {"fname" : "../rootfiles/bkg2_Events.root",
                            "encoding" : [0,1,0],
                            "label" : "BKG 2"}
INPUT_FILES_INFO["sig"] =  {"fname" : "../rootfiles/sig_Events.root",
                            "encoding" : [0,0,1],
                            "label" : "SIGNAL"}

### Initializing Input Variable Information

INPUT_VAR_NAMES = ["pt", "mass", "eta"]

### Initializing empty X, Y, W

X = [] 
Y = []
W = []





#######################
#### READING FILES ####
#######################

### Reading Root Files and Filling X, Y, W

for sample_name, sample in INPUT_FILES_INFO.items():
    f = uproot.open(sample["fname"])
    input_vars = [f["tree/" + var].array() for var in INPUT_VAR_NAMES]

    if(len(X) > 0):
        X = np.concatenate((X, np.transpose(input_vars)), axis=0)
        Y = np.concatenate((Y, [sample["encoding"]]*len(input_vars[0])), axis=0)
        W = np.concatenate((W, [1]*len(input_vars[0])),axis=0)
    else:
        X = np.transpose(input_vars)
        Y = [sample["encoding"]]*len(input_vars[0])
        W = [1]*len(input_vars[0])





############################
#### CREATING DNN MODEL ####
############################

### Defining Baseline Model

def baseline_model():
    ## BUILD MODEL HERE with 3 -> 6 -> 6 -> 3 STRUCTURE




    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], weighted_metrics=[tf.keras.losses.categorical_crossentropy])
    return model

### Splitting into Training and Testing Sets
X_train, X_test, Y_train, Y_test, W_train, W_test = train_test_split(X, Y, W, test_size=.5, random_state=123)

### Defining our Keras Model
estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=32, validation_split=0.35, verbose=1, shuffle=True)
history = estimator.fit(np.array(X_train),np.array(Y_train), sample_weight=np.array(W_train), callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=30,verbose=1)])





##############################
#### PLOTTING PERFORMANCE ####
##############################

pdf_pages = PdfPages("./dnn_history_tutorial1.pdf")
fig, ax = plt.subplots(1)
fig.suptitle("Model Accuracy")
ax.plot(history.history['accuracy'])
ax.plot(history.history['val_accuracy'])
ax.set_ylabel('Accuracy')
ax.set_xlabel('Epoch')
ax.set_ylim([0,1.1])
ax.legend(['train', 'test'], loc='upper left')
fig.set_size_inches(6,6)

fig2, ax2 = plt.subplots(1)
fig2.suptitle("Model Loss")
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.legend(['train', 'test'], loc='upper left')
fig2.set_size_inches(6,6)

pdf_pages.savefig(fig)
pdf_pages.savefig(fig2)





############################
#### PLOTTING CONFUSION ####
############################

def convert_onehot(Y_to_convert):
    Y_cat = []
    for i in range(0, len(Y_to_convert)):
        for j in range(0, 3):
            if(Y_to_convert[i][j] == 1):
                Y_cat.append(j)
    return Y_cat


predictions = estimator.predict(np.array(X_test))

softmax_outputs = estimator.model.predict(np.array(X_test))

Y_pred = predictions
Y_true_cat = np.array(convert_onehot(Y_test))

cm = sklearn.metrics.confusion_matrix(Y_true_cat, Y_pred)

for i in range(0,len(cm)):
    row_sum = float(np.sum(cm[i]))
    for j in range(0, len(cm[i])):
        cm[i][j] = float(cm[i][j]) / row_sum * 100.0

fig3, ax3 = plt.subplots(1)
ylabels = [sample["label"] for sample_name, sample in INPUT_FILES_INFO.items()]
xlabels = ylabels

ax3 = sns.heatmap(cm, cmap=sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True), xticklabels = xlabels, yticklabels=ylabels, linewidth=.25, annot=True)
ax3.set_xticklabels(ax3.get_xticklabels(),rotation=90)
ax3.set_title("Confusion Matrix")
ax3.set_xlabel("Predicted")
ax3.set_ylabel("True")
plt.tight_layout()
fig3.set_size_inches(6,6)
pdf_pages.savefig(fig3)





#############################
#### PLOTTING DNN SCORES ####
#############################

softmax_outputs = estimator.model.predict(np.array(X_test))
Y_true_cat = np.array(convert_onehot(Y_test))

BKG_1_scores = [[],[],[]]
BKG_2_scores = [[],[],[]]
SIG_scores   = [[],[],[]]

for i in range(0, len(Y_true_cat)):
    BKG_1_scores[Y_true_cat[i]].append(softmax_outputs[i][0])
    BKG_2_scores[Y_true_cat[i]].append(softmax_outputs[i][1])
    SIG_scores[Y_true_cat[i]].append(softmax_outputs[i][2])

bins = np.histogram(BKG_1_scores[0], 20, (0,1))[1]

BKG_1_scores_binned = [np.histogram(BKG_1_scores[i], 20, (0,1), density=True)[0] for i in range(0, len(BKG_1_scores))]
BKG_2_scores_binned = [np.histogram(BKG_2_scores[i], 20, (0,1), density=True)[0] for i in range(0, len(BKG_2_scores))]
SIG_scores_binned = [np.histogram(SIG_scores[i], 20, (0,1), density=True)[0] for i in range(0, len(SIG_scores))]

fig4, ax4 = plt.subplots(1)
ax4.set_title("BKG 1 Scores")
ax4.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], BKG_1_scores_binned[0], label="BKG 1", color="tab:blue")
ax4.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], BKG_1_scores_binned[1], label="BKG 2", color="tab:green")
ax4.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], BKG_1_scores_binned[2], label="SIG", color="tab:orange")
ax4.set_xlim([0,1])
ax4.set_xlabel("BKG 1 Scores")
ax4.set_ylabel("A.U.")
ax4.legend()
ax4.set_yscale('log')
plt.tight_layout()
fig4.set_size_inches(6,6)

pdf_pages.savefig(fig4)


fig5, ax5 = plt.subplots(1)
ax5.set_title("BKG 2 Scores")
ax5.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], BKG_2_scores_binned[0], label="BKG 1", color="tab:blue")
ax5.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], BKG_2_scores_binned[1], label="BKG 2", color="tab:green")
ax5.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], BKG_2_scores_binned[2], label="SIG", color="tab:orange")
ax5.set_xlim([0,1])
ax5.set_xlabel("BKG 2 Scores")
ax5.set_ylabel("A.U.")
ax5.legend()
ax5.set_yscale('log')
plt.tight_layout()
fig5.set_size_inches(6,6)

pdf_pages.savefig(fig5)


fig6, ax6 = plt.subplots(1)
ax6.set_title("Signal Scores")
ax6.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], SIG_scores_binned[0], label="BKG 1", color="tab:blue")
ax6.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], SIG_scores_binned[1], label="BKG 2", color="tab:green")
ax6.errorbar([bins[i]+(bins[i+1]-bins[i])/2 for i in range(0, len(bins)-1)], SIG_scores_binned[2], label="SIG", color="tab:orange")
ax6.set_xlim([0,1])
ax6.set_xlabel("Signal Scores")
ax6.set_ylabel("A.U.")
ax6.legend()
ax6.set_yscale('log')
plt.tight_layout()
fig6.set_size_inches(6,6)

pdf_pages.savefig(fig6)


pdf_pages.close()



