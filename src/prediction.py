'''
The file is used to perform predictions on the test set.
'''

import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import random
import pandas as pd
import itertools
from sklearn.preprocessing import RobustScaler
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor
import os

if __name__ == "__main__":

    # Invert the log transformation.
    predictions_best_casual = pd.DataFrame(predictions_best_casual)
    predictions_best_casual = pd.DataFrame(np.round(np.exp(predictions_best_casual) - 1)).stack()

    # Combine both model predictions.
    final_model_predictions_val = np.round((predictions_best_registered + predictions_best_casual))
    validation_y_count = pd.DataFrame(validation_y_count)
    correct_validation = np.exp(validation_y_count.stack()) - 1

    # Locally evaluate the model performance.
    eval_final_model = np.sqrt((sle((np.round(correct_validation)), final_model_predictions_val) ** 2).mean())
    print('Final model performance:' + str(eval_final_model))
    final_predictions = np.round((predictions_registered + predictions_casual))

    # Save submission file.
    submission_file = pd.read_csv('data/sampleSubmission.csv')
    submission_file['count'] = final_predictions.astype(np.int)
    submission_file.to_csv('submission_file.csv', index=False)

    print('EOF')

