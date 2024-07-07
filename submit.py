import numpy as np
import sklearn
from scipy.linalg import khatri_rao
from sklearn import linear_model

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y0_train, y1_train ):
################################
#  Non Editable Region Ending  #
################################
    feat = my_map(X_train)
    clf = sklearn.linear_model.LogisticRegression(fit_intercept=True,max_iter=1500,C = 50, random_state=0)
    clf1 = sklearn.linear_model.LogisticRegression(fit_intercept=True, max_iter=1500 ,C =50 , random_state=0)
    # training the response 0 set 
    clf.fit(feat, y0_train)
    # training the response 1 set f
    clf1.fit(feat , y1_train)
    # parameters of response 0
    w0 =  clf.coef_.T.flatten()
    b0 =  clf.intercept_
    # parameters of response 1 
    w1 = clf1.coef_.T.flatten()
    b1 = clf1.intercept_
    
	# Use this method to train your models using training CRPs
	# X_train has 32 columns containing the challenge bits
	# y0_train contains the values for Response0
	# y1_train contains the values for Response1

	
	# THE RETURNED MODELS SHOULD BE TWO VECTORS AND TWO BIAS TERMS
	# If you do not wish to use a bias term, set it to 0
    return w0, b0, w1, b1


################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to create features.
      d = 1 - 2*X 
      z = np.zeros((d.shape[0] , 32) , dtype = np.int32)
      z[ :,31] = d[:, 31]
      i = 0 
      while i< 31 :
           z[ :, 30-i] = z[:, 31-i] * d[:,30-i]
           i = i + 1
        
      X1 = np.zeros((d.shape[0] , 32) , dtype = np.int32)
      X1[: , 31] = X[:, 31] 

      i = 0
      while i<31 :
           X1[:, 30-i] = X1[:, 31-i] + X[:, 30-i] 
           i = i + 1

      feat = np.zeros((z.shape[0] , 63), dtype= np.int32) 

      k = 0     
      while k<32:
        feat[:,k]=X[:,k]  
        k = k+1
      n=31
      for j in range(n):
          feat[:, 32 + j ] = z[:, j] * X1[:, j+1] 
      
	# It is likely that my_fit will internally call my_map to create features for train points
	
      return feat
