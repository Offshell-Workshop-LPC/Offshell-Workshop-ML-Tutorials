


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

INPUT_FILE_NAME = "../rootfiles/EMTF_ntuple_slimmed.root"

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
#### CREATING BDT MODEL ####
############################

### Splitting into Training and Testing Sets
X_train, X_test, Y_train, Y_test, W_train, W_test = train_test_split(X, Y, W, test_size=.5, random_state=123)

X_train_split, X_val, Y_train_split, Y_val, W_train_split, W_val = train_test_split(X_train, Y_train, W_train, test_size=.5, random_state=321)


### Defining our XGBoost Model

xg_reg = xgb.XGBRegressor(objective = 'reg:squarederror', 
                        learning_rate = .15, 
                        max_depth = 3, 
                        n_estimators = 400,
                        nthread = 30)

xg_reg.fit(X_train_split, Y_train_split, eval_metric=["rmse"], sample_weight = W_train_split, eval_set=[(X_train_split, Y_train_split), (X_val, Y_val)])

results = xg_reg.evals_result()



##############################
#### PLOTTING PERFORMANCE ####
##############################

pdf_pages = PdfPages("./bdt_history_tutorial3.pdf")

fig, ax = plt.subplots(1)
fig.suptitle("Model Loss")
ax.plot(results['validation_0']['rmse'])
ax.plot(results['validation_1']['rmse'])
ax.set_ylabel('RMSE')
ax.set_xlabel('Epoch')
ax.legend(['train','test'], loc='upper left')
fig.set_size_inches(6,6)

pdf_pages.savefig(fig)


#############################
#### PLOTTING RESOLUTION ####
#############################

predictions = xg_reg.predict(X_test)
predictions_train = xg_reg.predict(X_train_split)

resolution = [(Y_test[i] - predictions[i])/Y_test[i] for i in range(0,len(Y_test))]
resolution_train = [(Y_train_split[i] - predictions_train[i])/Y_train_split[i] for i in range(0,len(Y_train_split))]

res_binned, res_bins = np.histogram(resolution, 100, (-2,2), density=True)
res_binned_train, res_bins = np.histogram(resolution_train, 100, (-2,2),density=True)

fig2, ax2 = plt.subplots(1)
fig2.suptitle("Model Resolution")
ax2.errorbar([res_bins[i]+(res_bins[i+1]-res_bins[i])/2 for i in range(0, len(res_bins)-1)],
                    res_binned, xerr=[(res_bins[i+1] - res_bins[i])/2 for i in range(0, len(res_bins)-1)],
                    linestyle="", marker=".", markersize=3, elinewidth = .5,label="Test")

ax2.errorbar([res_bins[i]+(res_bins[i+1]-res_bins[i])/2 for i in range(0, len(res_bins)-1)],
                    res_binned_train, xerr=[(res_bins[i+1] - res_bins[i])/2 for i in range(0, len(res_bins)-1)],
                    linestyle="", marker=".", markersize=3, elinewidth = .5,label="Train")

ax2.set_ylabel('$N_{events}$')
ax2.set_xlabel("$(p_T^{GEN} - p_T^{BDT})/(p_T^{GEN})$")
ax2.legend()
fig2.set_size_inches(6,6)


pdf_pages.savefig(fig2)

pdf_pages.close()
