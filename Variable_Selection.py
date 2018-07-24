                         # Backward_Elimination #
# Backward elimination is one of the variable selection method used to choose the most significant variable and build the final model
                     
                       # Function inputs or arguments

# First parameter - result # Example result = model.fit() # full model 
# Second paramter - threshold # Alpha value like 0.1,0.05 etc
# train_x # DataFrame with only predictor variables
# train_y # DataFrame with only independent variables
    
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm

rem=[] 
def Backward_Elimination(result,threshold,train_y,train_x):
    while np.amax(result.pvalues) > threshold:
      value_name=pd.Series(list(result.pvalues.values),index=result.pvalues.index)
      rem.append(value_name.idxmax())
      train_xx=train_x.drop(rem,axis=1)
      model=sm.Logit(train_y,train_xx)
      result=model.fit()
    return result
