'''
The file is used to perform all preprocessing steps and identify the best xgb model.
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


def bin_hours_registered(data):
    """
    Registered users showed a certain daily activity pattern which we used to bin hours.
    """
    hour_bin = data['hour']

    hour_bin[data['hour'] < 6] = 0
    hour_bin[data['hour'] == 6] = 1
    hour_bin[data['hour'] == 7] = 2
    hour_bin[data['hour'] == 8] = 3
    hour_bin[data['hour'] == 9] = 4
    hour_bin.ix[set(data[data['hour'] > 9].index.values).intersection(set(data[data['hour'] < 17].index.values))] = 5
    hour_bin.ix[set(data[data['hour'] > 16].index.values).intersection(set(data[data['hour'] < 19].index.values))] = 6
    hour_bin.ix[set(data[data['hour'] > 18].index.values).intersection(set(data[data['hour'] < 21].index.values))] = 7
    hour_bin[data['hour'] > 21] = 8

    data['hour_bin_reg'] = hour_bin
    return data


def bin_hours_casual(data):
    """
    Casual users showed a certain daily activity pattern which we used to bin hours.
    """
    hour_bin = data['hour']

    hour_bin[data['hour'] < 8] = 0
    hour_bin[data['hour'] > 22] = 0
    hour_bin.ix[set(data[data['hour'] > 7].index.values).intersection(set(data[data['hour'] < 11].index.values))] = 1
    hour_bin.ix[set(data[data['hour'] > 9].index.values).intersection(set(data[data['hour'] < 17].index.values))] = 2
    hour_bin[data['hour'] == 17] = 3
    hour_bin.ix[set(data[data['hour'] > 17].index.values).intersection(set(data[data['hour'] < 21].index.values))] = 4
    hour_bin.ix[set(data[data['hour'] > 20].index.values).intersection(set(data[data['hour'] < 23].index.values))] = 5

    data['hour_bin_cas'] = hour_bin
    return data


def extract_date_information(train_x):
    """
    We are extracting the date time information.
    :param train_x: Original input with single datetime string.
    :return: train_x: Modified training cases which additional time features.
    """
    # We are extract the hour, day, month, year of the datetime string.
    date_time_info = pd.DataFrame()
    date_time_info['year'] = pd.DatetimeIndex(train_x['datetime']).year
    date_time_info['month'] = pd.DatetimeIndex(train_x['datetime']).month
    date_time_info['day'] = pd.DatetimeIndex(train_x['datetime']).day
    date_time_info['hour'] = pd.DatetimeIndex(train_x['datetime']).hour

    # Here, we indicate which day of the year/month it is.
    date_time_info['day_of_year'] = pd.DatetimeIndex(train_x['datetime']).strftime('%j').astype(np.int)
    date_time_info['day_of_month'] = pd.DatetimeIndex(train_x['datetime']).strftime('%m').astype(np.int)
    date_time_info['day_of_week'] = pd.DatetimeIndex(train_x['datetime']).strftime('%w').astype(np.int)

    #  We add the new datatime information and delete the original one.
    train_x = pd.concat([train_x, date_time_info], axis=1)
    train_x = train_x.drop(['datetime'], axis=1)

    return train_x


def read_in_training_data():
    """
    We are reading in the data and we split the outcome variables.
    :return: train_x: Training data for the algorithm.
    :return: train_y: All outcome variables for the training data.
    """

    #  We are finding the the folder where the data is stored.
    path_training_data = os.path.normpath(os.getcwd() + os.sep + os.pardir) + '/data/training_data/train.csv'

    training_data = pd.read_csv(path_training_data)
    train_y = pd.DataFrame()
    train_y['count'] = training_data['count']
    train_y['casual'] = training_data['casual']
    train_y['registered'] = training_data['registered']
    train_x = training_data.drop(['count', 'casual', 'registered'], axis=1)

    return train_x, train_y


def read_in_testing_data():
    """
    We are reading in the test data.
    :return: test_x: The cases which need to be predicted.
    """
    path_testing_data = os.path.normpath(os.getcwd() + os.sep + os.pardir) + '/data/training_data/test.csv'
    test_x = pd.read_csv(path_testing_data)

    return test_x


def add_variable_free_day(data_set):
    """
    We compute an additional variable 'free day'.
    """
    is_holiday = data_set['holiday'] == 1
    is_sunday = data_set['day'] == 0
    is_saturday = data_set['day'] == 6

    day_off = is_holiday + is_saturday + is_sunday
    day_off = day_off.astype(np.int)
    data_set['day_off'] = day_off

    return data_set


def add_variable_weekend(data_set):
    """
    We compute an additional variable to indicate the weekend.
    """
    is_sunday = data_set['day'] == 0
    is_saturday = data_set['day'] == 6

    is_weekend = is_saturday + is_sunday
    is_weekend = is_weekend.astype(np.int)
    data_set['is_weekend'] = is_weekend

    return data_set


def add_variable_type_day(data_set):
    """
    We are computing a variable to categorize days.
    """
    # The additional categorization looks like this:
    # 1: normal working day, non-holiday
    # 2. working day, holiday
    # 3. non-working day, holiday
    # 4. non-working day, non-holiday.

    holiday_working_day = (data_set['holiday'] == 1) + (data_set['workingday'] == 1)
    holiday_not_working_day = (data_set['holiday'] == 1) + (data_set['workingday'] == 0)
    non_holiday_working_day = (data_set['holiday'] == 0) + (data_set['workingday'] == 1)
    non_holiday_not_working_day = (data_set['holiday'] == 1) + (data_set['workingday'] == 1)

    data_set['holiday_working_day'] = holiday_working_day.astype(np.int)
    data_set['holiday_not_working_day'] = holiday_not_working_day.astype(np.int)
    data_set['non_holiday_working_day'] = non_holiday_working_day.astype(np.int)
    data_set['non_holiday_not_working_day'] = non_holiday_not_working_day.astype(np.int)

    return data_set


def bin_hours(data_set):
    """
    Here, we bin the hour variable.
    """
    data_set['hours_categorized_2'] = np.floor(data_set['hour'] / 2)
    data_set['hours_categorized_3'] = np.floor(data_set['hour'] / 3)
    data_set['hours_categorized_4'] = np.floor(data_set['hour'] / 4)
    data_set['hours_categorized_5'] = np.floor(data_set['hour'] / 5)
    return data_set


def cluster_month(data_set):
    """
    Here, we split the months into different subgroups.
    """

    data_set['month_chunks_2'] = (np.ceil(data_set['year'] / min(data_set['year'])) * 10) + (np.floor(data_set['month'] / 2))
    data_set['month_chunks_3'] = (np.ceil(data_set['year'] / min(data_set['year'])) * 10) + (np.floor(data_set['month'] / 3))
    data_set['month_chunks_4'] = (np.ceil(data_set['year'] / min(data_set['year'])) * 10) + (np.floor(data_set['month'] / 4))

    return data_set


def add_log_noise(train_y, noise_level):
    """
    We are adding a small random noise to our trainig labels to improve generalization.
    :param train_y: Trainig data labels.
    :param noise_level: The amount of noise we are adding.
    :return: train_y: Training data labels with noise.
    """
    # We are adding noise between 0 and 10 for 2/3 of the y rows.

    #  We identify rows with will have noise and which will not noise.
    number_rows = list(range(0, train_y.shape[0]))
    noise_to_add = pd.Series([(random.random() / (30/noise_level)) for current_row in number_rows])
    no_noise_rows = random.sample(list(range(0, train_y.shape[0])), int(np.floor(train_y.shape[0]/3)))

    #  We apply a logarithm transformation to the noise.
    noise_to_add = np.log(noise_to_add + 1)
    noise_to_add = 1 + (noise_to_add - np.mean(noise_to_add))
    noise_to_add[no_noise_rows] = 1
    train_y = (train_y.stack().values * noise_to_add.values)
    train_y[train_y < 0] = 0
    return train_y


def rmsle(estimator, x, y):
    #  The code was obtained from: https://www.snip2code.com/Snippet/68562/Kaggle-Bike-Sharing-Demand-Competition--
    predictions = estimator.predict(x)

    return np.sqrt((sle(y, predictions)**2).mean())


def sle(actual, predicted):
    """
    Taken from benhamner's Metrics library.
    Computes the squared log error.

    This function computes the squared log error between two numbers,
    or for element between a pair of lists or numpy arrays.

    Parameters
    ----------
    actual : int, float, list of numbers, numpy array
             The ground truth value
    predicted : same type as actual
                The predicted value

    Returns
    -------
    score : double or list of doubles
            The squared log error between actual and predicted

    """
    return (np.power(np.log(np.array(actual)+1) -
                     np.log(np.array(predicted)+1), 2))


def produce_random_parameters():
    """
    We are producing randomized hyperparameters for the XGB regressor.
    :return: Pandas Series containing randmized hyperparamters.
    """
    parameters = pd.Series()
    parameters['max_depth'] = random.randint(3, 20)
    parameters['n_estimators'] = random.randint(100, 20000)
    parameters['learning_rate'] = random.randint(1, 100) / 10000
    parameters['subsample'] = random.randint(1, 100)/100
    parameters['colsample_by_level'] = random.randint(0, 100)/100
    parameters['min_child_weight'] = random.randint(1, 30)
    parameters['early_stopping'] = random.randint(1, 100)
    parameters['size_xgb_val'] = random.randint(1, 99)/100

    return parameters


def train_xgb_classifier(train_x, train_y, user_group):
    """
    We are training an xgb classifier with randomized hyperparameters.
    :param train_x: Feature-engineered training data.
    :param train_y: Training labels after logarithm and noise modifications.
    :param user_group: Name of the user group ('registered', 'casual').
    :return: xgb_models: A number of xgb models.
    """
    #  We can load old models or train new xgb models.
    retrain = True

    if retrain:
        #  We are developing many xgb classifiers which all have randomized hyperparameters.
        different_iterations = range(1, 3)
        xgb_models = dict()
        for iteration in different_iterations:
            parameters = produce_random_parameters()

            # Develop the model.
            x_train, x_test, y_train, y_test = train_test_split(train_x, train_y,
                                                                random_state=iteration,
                                                                test_size=parameters['size_xgb_val'])

            xgb_model = xgb.XGBRegressor(n_estimators=int(parameters['n_estimators']),
                                         max_depth=int(parameters['max_depth']),
                                         min_child_weight=parameters['min_child_weight'],
                                         learning_rate=parameters['learning_rate'],
                                         subsample=parameters['subsample'])

            xgb_model = xgb_model.fit(x_train, y_train, verbose=False,
                                      early_stopping_rounds=parameters['early_stopping'],
                                      eval_metric='rmse', eval_set=[(x_test, y_test)])

            # We are storing the model for evaluation at a later moment.
            xgb_models[iteration] = xgb_model

    else:
        #  We are loading models which have already been developed.
        xgb_path = os.path.normpath(os.getcwd() + os.sep + os.pardir) + '/data/xgb_models/' + user_group + '/'
        xgb_model_names = os.listdir(xgb_path)
        xgb_models = dict()

        for current_model_name in xgb_model_names:
            xgb_model = joblib.load(xgb_path + current_model_name)
            xgb_models[current_model_name] = xgb_model
        print('XGB models loaded.')

    return xgb_models


def perform_prediction_on_validation(xgb_models, validation_x, validation_y):
    """
    We are performing predictions on the test se
    :param xgb_models:
    :param validation_x:
    :param validation_y:
    :return:
    """
    validation_y = np.exp(validation_y) - 1
    results_rmlse = dict()
    for current_model_name in xgb_models.keys():
        xgb_model = xgb_models[current_model_name]

        predicted_values = xgb_model.predict(validation_x)
        predicted_values = np.exp(predicted_values) - 1
        results_rmlse[current_model_name] = np.sqrt((sle(validation_y.stack(), predicted_values) ** 2).mean())

    print('Best performing model for registered users: ' + str(min(results_rmlse.values())))
    return results_rmlse


def predict_test_x(model, test_x):
    """
    We are predicting the value on the test set using the best performing model.
    :param model: The best performing xgb classifier.
    :param test_x: The test set for which we need to predict the labels.
    :return: predictions: Predicted labels for the test set.
    """
    predictions = model.predict(test_x)
    predictions = np.exp(predictions) - 1
    predictions = np.round(predictions)
    return predictions


def bin_dew_values(data):
    """
    We decided to bin the dew values into bins of equal comfort.
    """
    bin_container = data['dew'].copy()
    bin_container[data['dew'] < 0] = 0
    bin_container[(data['dew'] > 0).astype(int) + (data['dew'] <= 12).astype(int) == 2] = 1
    bin_container[(data['dew'] > 12).astype(int) + (data['dew'] <= 17).astype(int) == 2] = 2
    bin_container[(data['dew'] > 16).astype(int) + (data['dew'] <= 19).astype(int) == 2] = 3
    bin_container[(data['dew'] > 18).astype(int) + (data['dew'] <= 22).astype(int) == 2] = 4
    bin_container[(data['dew'] > 22).astype(int) + (data['dew'] <= 25).astype(int) == 2] = 5
    bin_container[(data['dew'] > 25).astype(int) + (data['dew'] <= 27).astype(int) == 2] = 6
    bin_container[(data['dew'] > 2)] = 7

    data['dew_binned'] = bin_container
    data['bew_bin_3'] = np.floor(data['dew'] / 3)
    return data


def good_day(data):
    """
    We create some additional weather factors.
    """
    #  A certain interaction of weather factors provides the 'best' opportunity
    #  to rent out a bike.
    data['good_day'] = data['holiday']
    data['good_day'] = 0

    #  We are finding moments when the windspeed is optimal.
    windspeed_above_10 = np.where(data['windspeed'] > 10)[0]
    windspeed_below_25 = np.where(data['windspeed'] < 25)[0]
    comfort_windspeed = np.intersect1d(windspeed_above_10, windspeed_below_25)

    #  We are finding moments when the humidity is appropriate.
    humidity_above_10 = np.where(data['humidity'] > 10)[0]
    humidity_below_25 = np.where(data['humidity'] < 25)[0]
    comfort_humidity = np.intersect1d(humidity_above_10, humidity_below_25)

    #  We are finding the best moments in terms of temperature.
    temp_above_10 = np.where(data['temp'] > 25)[0]
    temp_below_25 = np.where(data['temp'] < 30)[0]
    comfort_temp = np.intersect1d(temp_above_10, temp_below_25)

    good_day = np.intersect1d(comfort_windspeed, comfort_humidity)
    good_day = np.intersect1d(good_day, comfort_temp)
    data['good_day'].values[good_day] = 1

    #  Apart from good days, there can be other types of days.
    #  (by https://github.com/logicalguess/kaggle-bike-sharing-demand/blob/master/code/main.py)
    data['ideal'] = data[['temp', 'windspeed']].apply(lambda x: (0, 1)[x['temp'] > 27 and
                                                                       x['windspeed'] < 30], axis=1)
    data['sticky'] = data[['humidity', 'workingday']].apply(lambda x: (0, 1)[x['workingday'] == 1 and
                                                                             x['humidity'] >= 60], axis=1)

    return data


def feature_engineering(data):
    """
    We are performing a number of feature engineering operations.
    :param data: Original training cases.
    :return: data: Original training cases with additional features.
    """
    data = round_given_columns(data)
    data = add_variable_free_day(data)
    data = add_variable_weekend(data)
    data = add_variable_type_day(data)
    data = bin_hours(data)
    data = good_day(data)
    data = calculate_dew_point(data)
    data = bin_dew_values(data)

    return data


def cluster_given_columns(data):
    # Here, we create data bins.
    data['temp_bins_3'] = np.floor(data['temp'] / 3)
    data['atemp_bins_3'] = np.floor(data['atemp'] / 3)
    data['humidity_5'] = np.floor(data['humidity'] / 5)
    data['windspeed_5'] = np.floor(data['windspeed'] / 5)

    return data


def round_given_columns(data):
    """
    We are rounding a number of columns to facilitate branching of the decision tree.
    :param data: Original training data.
    :return: data: Training data with rounded weather variables.
    """
    data['temp'] = data['temp'].round()
    data['atemp'] = data['atemp'].round()
    data['windspeed'] = data['windspeed'].round()
    data['humidity'] = data['humidity'].round()

    return data


def extract_validation_set(train_x, train_y):
    """
    We are creating a train-test split.
    :param: train_x: Full training data.
    :param: train_y: Labels of training data.
    :return: training_x: Training data split.
    :return: trainig_y: Training data labels split.
    :return: validation_x: Validation data split.
    :return: validation_y: Validation data labels split.
    """

    #  The first 4 days of a month are used as the validation set.
    cut_off_value = 4
    train_y = pd.DataFrame(train_y)
    training_x = train_x.loc[train_x['day'] >= cut_off_value, :]
    training_y = train_y.loc[train_x['day'] >= cut_off_value]

    validation_x = train_x.loc[train_x['day'] < cut_off_value, :]
    validation_y = train_y.loc[train_x['day'] < cut_off_value]
    return training_x, training_y, validation_x, validation_y


def perform_log_on_output(train_y):
    """
    Outliers were present. Here, we apply log transformation.
    :param train_y:
    :return:
    """
    #  We add 1 to avoid that 0s become -inf.
    train_y = np.log(train_y + 1)
    return train_y


def interpolate_missing_data(train_x):
    """
    We interpolate missing (or unusual) cases of weather factors.
    :param train_x: Original training data.
    :return: train_x: Training data with interpolated weather data.
    """
    train_x["weather"] = train_x["weather"].interpolate(method='time').apply(np.round)
    train_x["temp"] = train_x["temp"].interpolate(method='time')
    train_x["atemp"] = train_x["atemp"].interpolate(method='time')
    train_x["humidity"] = train_x["humidity"].interpolate(method='time').apply(np.round)
    train_x["windspeed"] = train_x["windspeed"].interpolate(method='time')

    return train_x


def calculate_dew_point(train_x):
    """
    We calculate the dew point (an absolute measure of humidity).
    """
    #  Background information regarding the dew measure.
    #  According to https://en.wikipedia.org/wiki/Dew_point a certain binning of the weather might make sense.
    #  Temperature corresponds to the soil temperature and atemp is the air temperature.
    #  The soil temperature becomes somewhat warmer during the day and stays warm in the night.
    #  The airtemp is highly variable and can become hot during the day and cold during the night.


    # The code was obtained from: http://www.meteo-blog.net/2012-05/dewpoint-calculation-script-in-python/

    dew = np.round(dewpoint_approximation(train_x['atemp'], train_x['humidity']), decimals=5)
    dew = dew.fillna(np.round(dew.mean()))
    dew = np.round(dew)
    train_x['dew'] = dew

    return train_x


def dewpoint_approximation(temperature, rhumidity):
    """
    Compute dew point approximation.
    """
    # obtained from: http://www.meteo-blog.net/2012-05/dewpoint-calculation-script-in-python/constants
    a = 17.271
    b = 237.7 # degC
    Td = (b * gamma(temperature,rhumidity)) / (a - gamma(temperature,rhumidity))
    return Td


def gamma(temperature, rhumidity):
    """
    We are computing gamma.
    """
    # obtained from: http://www.meteo-blog.net/2012-05/dewpoint-calculation-script-in-python/constants
    a = 17.271
    b = 237.7
    g = (a * temperature / (b + temperature)) + np.log(rhumidity/100.0)
    return g


def interpolate_missing_y(train_y):
    """
    We are interpolating missing values of the outcome variable.
    """
    train_y = train_y.interpolate(method='time').apply(np.round)
    return train_y


def standardize_columns(data):
    """
    We decided to standardize the weather factor due to outliers.
    """
    columns_to_standardize = ['temp', 'atemp', 'humidity', 'windspeed']
    min_max_scaler = RobustScaler()

    for column in columns_to_standardize:
        data[column] = min_max_scaler.fit_transform(data[column])
    return data


def reverse_log(predictions):
    """
    We are reversing the logarithm transformation.
    :param predictions: The original predictions.
    :return: predictions_reversed: Log-reversed predictions.
    """
    predictions_reversed = np.exp(predictions) - 1
    return predictions_reversed


def select_best_model(xgb_models, results_rsmle):
    """
    We are identifying the best performing model
    :param xgb_models: All XGB models.
    :param results_rsmle: Performance of xgb models on holdout set.
    :return: best_xgb: Best xgb classifier.
    """
    best_model = min(results_rsmle, key=results_rsmle.get)
    best_xgb = xgb_models[best_model]

    return best_xgb


def preprocess_save_data(perform_preprocessing):
    """
    We are reading in the data and performing all preprocessing steps.
    :param perform_preprocessing: Repeat preprocessing or load data.
    :return train_x: Training data for the algorithm.
    :return train_y: Labels for the training data.
    :return test_x: Test cases which need to be predicted.
    """

    #  We are reading in the data and performing all preprocessing
    #  steps from scratch.
    train_x, train_y = read_in_training_data()
    test_x = read_in_testing_data()

    test_x = extract_date_information(test_x)
    train_x = extract_date_information(train_x)

    size_training = train_x.shape[0]
    data = pd.concat([train_x, test_x], axis=0)
    data.index = range(0, data.shape[0])

    data = interpolate_missing_data(data)
    data = feature_engineering(data)

    # Separate the data again into train and real test
    training_rows = range(0, size_training)
    testing_rows = range(size_training, (data.shape[0]))
    train_x = data.iloc[training_rows, :]
    test_x = data.iloc[testing_rows, :]

    train_y = perform_log_on_output(train_y)

    return train_x, train_y, test_x


if __name__ == "__main__":  
    """
    The function contains the complete data preparation and model building steps.
    """
    perform_preprocessing = True
    train_x, train_y, test_x = preprocess_save_data(perform_preprocessing)

    # Separate training and local validation set for the three dependant variables.
    train_x_casual, train_y_casual, validation_x_casual, validation_y_count = \
        extract_validation_set(train_x, train_y['count'])
    train_x_registered, train_y_registered, validation_x_registered, validation_y_registered = \
        extract_validation_set(train_x, train_y['registered'])
    train_x_casual, train_y_casual, validation_x_casual, validation_y_casual = \
        extract_validation_set(train_x, train_y['casual'])

    # Add noise to dependent variable.
    noise_level = 1
    train_y_casual = add_log_noise(train_y_casual, noise_level)
    noise_level = 1
    train_y_registered = add_log_noise(train_y_registered, noise_level)

    # First, we develop the model for registered users.
    user_group = 'registered'
    xgb_models = train_xgb_classifier(train_x_registered, train_y_registered, user_group)

    # After training many models, we identify the best by their rmsle result on the validation set.
    results_rsmle = perform_prediction_on_validation(xgb_models, validation_x_registered, validation_y_registered)
    best_xgb = select_best_model(xgb_models, results_rsmle)
    predictions_best_registered = best_xgb.predict(validation_x_registered)
    predictions_registered = predict_test_x(best_xgb, test_x)

    # Invert the log transformation.
    predictions_best_registered = pd.DataFrame(predictions_best_registered)
    predictions_best_registered = pd.DataFrame(np.round(np.exp(predictions_best_registered) - 1)).stack()

    # Secondly, we develop the model for casual users.
    user_group = 'casual'
    xgb_models = train_xgb_classifier(train_x_casual, train_y_casual, user_group)

    # After training many models, we identify the best by their rmsle result on the validation set.
    results_rsmle = perform_prediction_on_validation(xgb_models, validation_x_casual, validation_y_casual)
    best_xgb = select_best_model(xgb_models, results_rsmle)
    predictions_best_casual = best_xgb.predict(validation_x_casual)
    predictions_casual = predict_test_x(best_xgb, test_x)

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

