# Tutorial 1 - Introduction to Classifier NN
This tutorial is developed for the Offshell Workshop - Tuesday 14:05-15:05.

[Introduction Slides](https://indico.cern.ch/event/1375252/timetable/#16-machine-learning)

## Getting Setup

```
cd Tutorial_1
```

## Imports
The first step is to import the packages we will need to create the NN.

```
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
```

## Initializing Variables
We also need to initialize some variables we want to use in our script.
```
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

## Initializing Input Variable Information

INPUT_VAR_NAMES = ["pt", "mass", "eta"]

### Initializing empty X, Y, W

X = []
Y = []
W = []
```

## Reading Root Files using Uproot
We will be using Uproot to read our input root files. Uproot is very useful because it makes the interface with ML tools much easier since branches are converted to numpy arrays. This makes the process of reading in our data very straight forward - we can simply loop through each file and pulling the input variables directly. One thing to be mindful of is that we need our `X` array to be in the proper format, so it should be that each element in the 2D array should correspond to the input layer but when we get these variables our of the root file using `f["tree/" + var].array()` each element will be for a given variable. Thus, we will need to take the transpose of the `input_var` array. 

Example:
```
input_var_directly = [ [pt1, pt2, pt3, pt4 ...],
                       [mass1, mass2, mass3, mass4 ...]
                       [eta1, eta2, eta3, eta4 ...] ]

input_var_transposed  = [ [pt1, mass1, eta1],
                          [pt2, mass2, eta2],
                          [pt3, mass3, eta3],
                          [pt4, mass4, eta4], ...]
```

```
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

```

## Defining the Keras Model


We can create our NN model using Keras. In a function, here called `baseline_model`, we can set how many layers we want, how many nodes in each layer, the type of layer, each layers activiation function, and the model's loss function, optimizer algorithm, and metrics.

To add layers to the model we simply call `model.add()`. For the first layer, we need to add an argument `input_dim=len(INPUT_VAR_NAMES)`.

**Try to use the example below to modify dnn_classifier.py**

```
### Defining Baseline Model

def baseline_model():
    model = Sequential() # 3, 6, 6, 3
    model.add(Dense(6, input_dim=len(INPUT_VAR_NAMES), activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], weighted_metrics=[tf.keras.losses.categorical_crossentropy])
    return model
```

Then we can split our data into two orthogonal sets of Training and Testing sets.

Finally, we can define the `estimator` and perform the fit to the training set. This is also where we can change many of the hyper-parameters of the training such as the number of epochs, batch size, and early stopping algorithms. 
```
### Splitting into Training and Testing Sets
X_train, X_test, Y_train, Y_test, W_train, W_test = train_test_split(X, Y, W, test_size=.5, random_state=123)


### Defining our Keras Model
estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=32, validation_split=0.35, verbose=1, shuffle=True)
history = estimator.fit(np.array(X_train),np.array(Y_train), sample_weight=np.array(W_train), callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=30,verbose=1)])
```

## Plotting Loss and Accuracy
It is always important to plot the loss and accuracy of the training and validation sets to look for indications of training errors or overtraining. 
```
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
```

## Confusion Matrix
The confusion matrix is an easy way to view the performance of a classifier NN and is often a good way to look for ways to improve the NN. By looking at where the NN gets confused most often, it can indicate that additional variables are needed or that some samples are weighted incorrectly.
```
# Useless function added here to simplify some code
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
```

## Plotting Score Distributions
The final step in this tutorial is to plot the distributions of scores. For each event we will get an output layer that has as many values as we put in our final layer (in our case that is 3 since we have 3 categories). Since we know the true category each event should be in we can plot the distribution of scores for a given category for each of the true category. This essentially displays the same information that the confusion matrix shows but in more detail. If the NN scores are to be used with a threshold instead of using the maximum value, it is also important to make these plots. 

Finally, if the same plots are created for the training set, one could perform the KS test between the test and training sets to ensure that the NN was not overtrained.
```
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
```

## Overtraining

We can attempt to simulate overtraining the NN by changing our model's hyperparameters. 

**Try to add too many nodes to each layer in dnn_classifier.py**

**Try to add too many layers in dnn_classifier.py**

## Event Weighting

We can also see what happens  when we weight each class different.

**Modify the event weights from for loop that reads the root files using the example below.**

```

for sample_name, sample in INPUT_FILES_INFO.items():
    f = uproot.open(sample["fname"])
    input_vars = [f["tree/" + var].array() for var in INPUT_VAR_NAMES]

    w = 1
    if(sample["label"] is "BKG 1"):
        w = 10

    if(len(X) > 0):
        X = np.concatenate((X, np.transpose(input_vars)), axis=0)
        Y = np.concatenate((Y, [sample["encoding"]]*len(input_vars[0])), axis=0)
        W = np.concatenate((W, [w]*len(input_vars[0])),axis=0)
    else:
        X = np.transpose(input_vars)
        Y = [sample["encoding"]]*len(input_vars[0])
        W = [w]*len(input_vars[0])

```

