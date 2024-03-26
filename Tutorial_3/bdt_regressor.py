


#################
#### IMPORTS ####
#################

# General Utilities
import math
import random
import numpy as np
import uproot

# Sklearn Utilities
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Plotting Utilities
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import matplotlib.pyplot as plt


######################
#### INITIALIZING ####
######################

### Initializing Input File Information

INPUT_FILE_NAME = "../../EMTF_ntuple_slimmed_v2.root"

### Initializing Input Variable Information

INPUT_VAR_NAMES = ["theta", "st1_ring2", "dPhi_12", "dPhi_23", "dPhi_34", "dPhi_13", "dPhi_14", "dPhi_24", "FR_1", "bend_1", "dPhiSum4", "dPhiSum4A", "dPhiSum3", "dPhiSum3A", "outStPhi", "dTh_14", "RPC_1", "RPC_2", "RPC_3", "RPC_4"]

### Initializing empty X, Y, W

X = []
Y = []
W = []


#######################
#### READING FILES ####
#######################

### Reading Root Files and Filling X, Y, W

f = uproot.open(INPUT_FILE_NAME)
input_vars = [f["tree/" + var].array() for var in INPUT_VAR_NAMES]

X = np.transpose(input_vars)
Y = f["tree/GEN_pt"].array()
W = [1]*len(input_vars[0])



############################
#### CREATING DNN MODEL ####
############################

### Defining Baseline Model

def baseline_model():
    model = Sequential() # 20, 60, 30, 15, 1
    model.add(Dense(60, input_dim=len(INPUT_VAR_NAMES), activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

### Splitting into Training and Testing Sets
X_train, X_test, Y_train, Y_test, W_train, W_test = train_test_split(X, Y, W, test_size=.5, random_state=123)


### Defining our Keras Model

dtrain = xgb.DMatrix(data = X_train, label = Y_train, weight = W_train)
dtest = xgb.DMatrix(data = X_test, label = Y_test, weight = W_test)

xg_reg = xgb.XGBRegressor(objective = 'reg:squarederror', 
                        learning_rate = .1, 
                        max_depth = 5, 
                        n_estimators = 400,
                        max_bins = 1000,
                        nthread = 30)

xg_reg.fit(X_train, Y_train, sample_weight = W_train)

#############################
#### PLOTTING RESOLUTION ####
#############################

predictions = xg_reg.predict(X_test)

resolution = [(Y_test[i] - predictions[i])/Y_test[i] for i in range(0,len(Y_test))]
res_binned, res_bins = np.histogram(resolution, 100, (-2,2))

pdf_pages = PdfPages("./dnn_history_tutorial3.pdf")
fig, ax = plt.subplots(1)
fig.suptitle("Model Resolution")
ax.errorbar([res_bins[i]+(res_bins[i+1]-res_bins[i])/2 for i in range(0, len(res_bins)-1)],
                    res_binned, xerr=[(res_bins[i+1] - res_bins[i])/2 for i in range(0, len(res_bins)-1)],
                    linestyle="", marker=".", markersize=3, elinewidth = .5)
ax.set_ylabel('$N_{events}$')
ax.set_xlabel("$(p_T^{GEN} - p_T^{NN})/(p_T^{GEN})$")
fig.set_size_inches(6,6)


pdf_pages.savefig(fig)
pdf_pages.close()
