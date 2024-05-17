from scipy.stats import pearsonr # Pearson R best
from keras.metrics import MeanAbsoluteError
from typing import cast
import keras
import math

def regression_metrics(y_label, y_pred):
  print("++++++++++++++++++++++++++Regression++++++++++++++++++++++++++")
  PearsonR, _ = pearsonr(y_label, y_pred)
  PearsonR = cast(float, PearsonR)
  print('Pearson Correlation Coefficient: ' + str(PearsonR))
  MSE = MeanAbsoluteError()
  MSE.update_state(y_label, y_pred)
  MSE = MSE.result().numpy() # type: ignore
  print('Mean Absolute Error: ' + str(MSE))
  # RMSE = math.sqrt(MSE)
  # print('Root Mean Absolute Error: ' + str(RMSE))

  return PearsonR, MSE

def classification_metric(y_label, y_pred):
    print("+++++++++++++++++++++++Classification+++++++++++++++++++++++")
    # auc = keras.metrics.AUC()
    tp = keras.metrics.TruePositives(thresholds= 0.9)
    tn = keras.metrics.TrueNegatives(thresholds= 0.9)
    fp = keras.metrics.FalsePositives(thresholds= 0.9)
    fn = keras.metrics.FalseNegatives(thresholds= 0.9)
      
    # auc.update_state(y_label, y_pred)
    tp.update_state(y_label, y_pred)
    tn.update_state(y_label, y_pred)
    fp.update_state(y_label, y_pred)
    fn.update_state(y_label, y_pred)
    
    # auc = auc.result().numpy()
    tp_result = tp.result().numpy() # type: ignore
    tn_result = tn.result().numpy() # type: ignore
    fp_result = fp.result().numpy() # type: ignore
    fn_result = fn.result().numpy() # type: ignore
    
    tp_result = cast(float, tp_result)
    tn_result = cast(float, tn_result)
    fp_result = cast(float, fp_result)
    fn_result = cast(float, fn_result)

    print("TP Result:", tp_result)
    print("FP Result:", fp_result)
    precision = tp_result/ (tp_result+fp_result) # PPV
    print('Precision: ' + str(precision))

    # recall = Recall()
    # recall.update_state(y_label, y_pred)
    # recall = recall.result().numpy()
    recall = tp_result/(tp_result+fn_result) # Recall - TPR
    print('Recall: ' + str(recall))

    specificity = tn_result/(tn_result+fp_result)
    print('Specificity: ' + str(specificity))

    NPV = tn_result/(tn_result+fn_result)
    print('NPV: ' + str(NPV))

    # accuracy = Accuracy()
    # accuracy.update_state(y_label, y_pred)
    # accuracy = accuracy.result().numpy()
    # print('AUC: ' + str(auc))

    # f1_score = 2 * (precision * recall) / (precision + recall)
    # print('F1_Score: ' + str(f1_score))

    MCC = (tp_result*tn_result - fp_result*fn_result)/ math.sqrt((tp_result+fp_result)*(tp_result+fn_result)*(tn_result+fp_result)*(tn_result+fn_result)  ) # Phi coefficient
    print("Phi coefficient:" + str(MCC))

    return precision, recall, specificity, NPV, MCC