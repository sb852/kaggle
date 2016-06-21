import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import random
import pandas as pd
import itertools
from sklearn.preprocessing import RobustScaler
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor


def bin_hours_registered(data):
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
    date_time_info = pd.DataFrame()
    # Extract hour, day, month, year of the datetime string.
    date_time_info['year'] = pd.DatetimeIndex(train_x['datetime']).year
    date_time_info['month'] = pd.DatetimeIndex(train_x['datetime']).month
    date_time_info['day'] = pd.DatetimeIndex(train_x['datetime']).day
    date_time_info['hour'] = pd.DatetimeIndex(train_x['datetime']).hour

    # Here, we indicate which day of the year/month it is.
    date_time_info['day_of_year'] = pd.DatetimeIndex(train_x['datetime']).strftime('%j').astype(np.int)
    date_time_info['day_of_month'] = pd.DatetimeIndex(train_x['datetime']).strftime('%m').astype(np.int)
    date_time_info['day_of_week'] = pd.DatetimeIndex(train_x['datetime']).strftime('%w').astype(np.int)

    train_x = pd.concat([train_x, date_time_info], axis=1)
    train_x = train_x.drop(['datetime'], axis=1)

    return train_x


def read_in_training_data():
    path_training_data = "data/train(1).csv"
    training_data = pd.read_csv(path_training_data)
    train_y = pd.DataFrame()
    train_y['count'] = training_data['count']
    train_y['casual'] = training_data['casual']
    train_y['registered'] = training_data['registered']

    train_x = training_data.drop(['count', 'casual', 'registered'], axis=1)

    train_x = extract_date_information(train_x)
    return train_x, train_y


def read_in_testing_data():
    path_testing_data = "data/test.csv"
    testing_data = pd.read_csv(path_testing_data)

    testing_data = extract_date_information(testing_data)
    return testing_data


def add_variable_free_day(data_set):
    # Here, we compute an additional variable for days which are off.

    is_holiday = data_set['holiday'] == 1
    is_sunday = data_set['day'] == 0
    is_saturday = data_set['day'] == 6

    day_off = is_holiday + is_saturday + is_sunday
    day_off = day_off.astype(np.int)
    data_set['day_off'] = day_off

    return data_set


def add_variable_weekend(data_set):
    is_sunday = data_set['day'] == 0
    is_saturday = data_set['day'] == 6

    is_weekend = is_saturday + is_sunday
    is_weekend = is_weekend.astype(np.int)
    data_set['is_weekend'] = is_weekend

    return data_set


def add_variable_type_day(data_set):
    # Here, we add a variable to see if it is a
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


def categorize_hours(data_set):
    data_set['hours_categorized_2'] = np.floor(data_set['hour'] / 2)
    data_set['hours_categorized_3'] = np.floor(data_set['hour'] / 3)
    data_set['hours_categorized_4'] = np.floor(data_set['hour'] / 4)
    data_set['hours_categorized_5'] = np.floor(data_set['hour'] / 5)
    return data_set


def cluster_month(data_set):
    # Here, we split the months into different subgroups.

    data_set['month_chunks_2'] = (np.ceil(data_set['year'] / min(data_set['year'])) * 10) + (np.floor(data_set['month'] / 2))
    data_set['month_chunks_3'] = (np.ceil(data_set['year'] / min(data_set['year'])) * 10) + (np.floor(data_set['month'] / 3))
    data_set['month_chunks_4'] = (np.ceil(data_set['year'] / min(data_set['year'])) * 10) + (np.floor(data_set['month'] / 4))

    return data_set


def add_noise(train_y):
    # Here, we add noise between 0 and 10 for 2/3 of the y rows.

    number_rows = list(range(0, train_y.shape[0]))
    noise_to_add = pd.Series([random.randrange(0, 10, 1) for current_row in number_rows])
    no_noise_rows = random.sample(list(range(0, train_y.shape[0])), int(np.floor(train_y.shape[0]/3)))
    noise_to_add[no_noise_rows] = 0
    train_y = train_y + noise_to_add
    train_y[train_y < 0] = 0
    print(train_y)
    return train_y


def add_log_noise(train_y, noise_level):
    # Here, we add noise between 0 and 10 for 2/3 of the y rows.

    number_rows = list(range(0, train_y.shape[0]))
    noise_to_add = pd.Series([(random.random() / (30/noise_level)) for current_row in number_rows])
    no_noise_rows = random.sample(list(range(0, train_y.shape[0])), int(np.floor(train_y.shape[0]/3)))
    noise_to_add = np.log(noise_to_add + 1)
    noise_to_add = 1 + (noise_to_add - np.mean(noise_to_add))
    noise_to_add[no_noise_rows] = 1
    train_y = (train_y.stack().values * noise_to_add.values)
    train_y[train_y < 0] = 0
    print(train_y)
    return train_y


def check_boundary_noise(intended_noise, maximum_noise):
    if intended_noise < 0 and abs(intended_noise) > maximum_noise:
        intended_noise = -maximum_noise
    elif intended_noise > 0 and abs(intended_noise) > maximum_noise:
        intended_noise = maximum_noise
    return intended_noise


def create_submission_data(submission_dates, prediction_to_be_submitted):
    prediction_to_be_submitted = prediction_to_be_submitted.flatten()
    submission_dates = np.hstack(['datetime', submission_dates])
    prediction_to_be_submitted = np.hstack(['count', prediction_to_be_submitted])
    prediction_to_be_submitted = np.vstack([submission_dates, prediction_to_be_submitted])
    prediction_to_be_submitted = np.transpose(prediction_to_be_submitted)

    return prediction_to_be_submitted


def create_submission_file(prediction_to_be_submitted, file_name):
    np.savetxt(file_name, prediction_to_be_submitted, delimiter=",", fmt='%s')
    print("Submission file saved!" + file_name)


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
    parameters = pd.Series()
    parameters['max_depth'] = random.randint(3, 20)
    parameters['n_estimators'] = random.randint(500, 20000)
    parameters['learning_rate'] = random.randint(1, 100) / 10000
    parameters['subsample'] = random.randint(1, 100)/100
    parameters['colsample_by_level'] = random.randint(0, 100)/100
    parameters['min_child_weight'] = random.randint(1, 30)
    parameters['early_stopping'] = random.randint(1, 100)
    parameters['size_xgb_val'] = random.randint(1, 99)/100

    return parameters


def train_xgb_bagging_classifier(train_x, train_y):
    different_iterations = range(1, 300)
    xgb_models = dict()

    for iteration in different_iterations:
        parameters = produce_random_parameters()

        # Develop the model.
        x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, random_state=iteration, test_size=parameters['size_xgb_val'])
        xgb_model = xgb.XGBRegressor(n_estimators=int(parameters['n_estimators']), max_depth=parameters['max_depth'], min_child_weight=parameters['min_child_weight'],
                                     learning_rate=parameters['learning_rate'], subsample=parameters['subsample'])
        xgb_model = xgb_model.fit(x_train, y_train, verbose=0, early_stopping_rounds=parameters['early_stopping'], eval_metric='rmse', eval_set=[(x_test, y_test)])

        # Dump the model.
        save_xgb = 'xgb_models/model' + str(iteration) + '.pkl'
        joblib.dump(xgb_model, save_xgb)
        xgb_models[iteration] = save_xgb

        print('xbg fitted, iteration ',  iteration)

    return xgb_models


def train_sklearn_bagging_classifier(train_x, train_y, parameters):
    different_iterations = range(1, 2)

    xgb_models = dict()
    for iteration in different_iterations:
        # Develop the model.
        x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, random_state=iteration, test_size=parameters['size_xgb_val'])

        clf = RandomForestRegressor(n_estimators= int(parameters['n_estimators']), max_depth=int(parameters['max_depth']), min_samples_split = int(parameters['min_child_weight']))

        clf = clf.fit(x_train, y_train)

        # Dump the model.
        save_xgb = 'xgb_models/model' + str(iteration) + '.pkl'
        joblib.dump(clf, save_xgb)
        xgb_models[iteration] = save_xgb

        print('xbg fitted, iteration ',  iteration)

    return xgb_models


def perform_prediction_on_validation(xgb_models, validation_x, validation_y):

    validation_y = np.exp(validation_y) - 1
    results_rmlse = dict()
    for element in xgb_models:
        xgb_model = joblib.load(xgb_models[element])

        predicted_values = xgb_model.predict(validation_x)
        predicted_values = np.exp(predicted_values) - 1
        results_rmlse[element] = np.sqrt((sle(validation_y.stack(), predicted_values) ** 2).mean())

    print(results_rmlse)
    return results_rmlse


def predict_test_x(model, test_x):
    predictions = model.predict(test_x)
    predictions = np.exp(predictions) - 1
    predictions = np.round(predictions)
    return predictions


def create_list_of_all_combinations():
    in_list = [1, 2, 3, 4]
    out_list = []
    for i in range(1, len(in_list) + 1):
        out_list.extend(itertools.combinations(in_list, i))
    out_list = [x[0] if len(x) == 1 else list(x) for x in out_list]
    return out_list


def bin_dew_values(data):
    bin_container = data['dew']
    bin_container[data['dew'] < 0] = 0
    bin_container[(data['dew'] > 0).astype(int) + (data['dew'] < 12).astype(int) == 2 ] = 1
    bin_container[(data['dew'] > 12).astype(int) + (data['dew'] < 17).astype(int) == 2 ] = 2
    bin_container[(data['dew'] > 16).astype(int) + (data['dew'] < 19).astype(int) == 2 ] = 3
    bin_container[(data['dew'] > 18).astype(int) + (data['dew'] < 22).astype(int) == 2 ] = 4
    bin_container[(data['dew'] > 22).astype(int) + (data['dew'] < 25).astype(int) == 2 ] = 5
    bin_container[(data['dew'] > 25).astype(int) + (data['dew'] < 27).astype(int) == 2 ] = 6
    bin_container[(data['dew'] > 26)] = 7

    data['dew_binned'] = bin_container
    data['bew_bin_3'] = np.floor(data['dew'] / 3)
    return data


def good_day(data):
    # Here, we identify and encode the characteristics of a particularily good day.

    data['good_day'] = data['holiday']
    data['good_day'] = 0
    windspeed_rows = ((data['windspeed'] > 10).astype(int) + (data['windspeed'] < 25).astype(int) == 2)
    humidity_rows = ((data['humidity'] > 10).astype(int) + (data['windspeed'] < 40).astype(int) == 2)
    temp_rows = ((data['temp'] > 25).astype(int) + (data['temp'] < 30).astype(int) == 2)
    good_day_rows = ((windspeed_rows.astype(int) + humidity_rows.astype(int) + temp_rows.astype(int)) == 3)
    data['good_day'].where(good_day_rows == True, other=1)

    data.ix[data['good_day'], 'good_day'] = 1

    # This code is inspired by https://github.com/logicalguess/kaggle-bike-sharing-demand/blob/master/code/main.py
    data['ideal'] = data[['temp', 'windspeed']].apply(lambda x: (0, 1)[x['temp'] > 27 and x['windspeed'] < 30], axis=1)
    data['sticky'] = data[['humidity', 'workingday']].apply(lambda x: (0, 1)[x['workingday'] == 1 and x['humidity'] >= 60], axis=1)

    return data


def feature_engineering(data):
    data = round_given_columns(data)
    data = add_variable_free_day(data)
    data = add_variable_weekend(data)
    data = add_variable_type_day(data)
    data = categorize_hours(data)
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
    columns_rounding = ['temp', 'atemp', 'windspeed', 'humidity']
    for given_column in columns_rounding:
        data[given_column].round()

    return data


def extract_validation_set(train_x, train_y):
    cut_off_value = 4
    train_y = pd.DataFrame(train_y)
    training_x = train_x.loc[train_x['day'] > cut_off_value, :]
    training_y = train_y.loc[train_x['day'] > cut_off_value]

    validation_x = train_x.loc[train_x['day'] < cut_off_value, :]
    validation_y = train_y.loc[train_x['day'] < cut_off_value]
    return training_x, training_y, validation_x, validation_y


def perform_log_on_output(train_y):
    # Here, we apply log transformation to avoid that 0s become - inf.
    train_y = np.log(train_y + 1)
    return train_y


def interpolate_missing_data(train_x):
    train_x["weather"] = train_x["weather"].interpolate(method='time').apply(np.round)
    train_x["temp"] = train_x["temp"].interpolate(method='time')
    train_x["atemp"] = train_x["atemp"].interpolate(method='time')
    train_x["humidity"] = train_x["humidity"].interpolate(method='time').apply(np.round)
    train_x["windspeed"] = train_x["windspeed"].interpolate(method='time')

    return train_x


def bin_weather(train_x):
    # According to https://en.wikipedia.org/wiki/Dew_point a certain binning of the weather might make sense.
    # Temperature corresponds to the soil temperature and atemp is the air temperature.
    # The soil temperature becomes somewhat warmer during the day and stays warm in the night.
    # The airtemp is highly variable and can become hot during the day and cold during the night.
    # Here, I chose the atemp for the binning.

    train_x["temp"] = train_x["temp"].interpolate(method='time')
    train_x["atemp"] = train_x["atemp"].interpolate(method='time')
    train_x["humidity"] = train_x["humidity"].interpolate(method='time').apply(np.round)
    return train_x


def calculate_dew_point(train_x):
    # obtained from: http: // www.meteo - blog.net / 2012 - 05 / dewpoint - calculation - script - in -python /

    dew = np.round(dewpoint_approximation(train_x['atemp'], train_x['humidity']), decimals=5)
    dew = dew.fillna(np.round(dew.mean()))
    dew.to_pickle('dew.pkl')
    train_x['dew'] = dew

    return train_x


def dewpoint_approximation(T,RH):
    # obtained from: http: // www.meteo - blog.net / 2012 - 05 / dewpoint - calculation - script - in -python /
    # constants
    a = 17.271
    b = 237.7 # degC

    Td = (b * gamma(T,RH)) / (a - gamma(T,RH))

    return Td


def gamma(T,RH):
    # obtained from: http: // www.meteo - blog.net / 2012 - 05 / dewpoint - calculation - script - in -python /
    # constants
    a = 17.271
    b = 237.7 # degC

    g = (a * T / (b + T)) + np.log(RH/100.0)

    return g


def interpolate_missing_y(train_y):
    train_y = train_y.interpolate(method='time').apply(np.round)
    return train_y


def standardize_columns(data):
    columns_to_standardize = ['temp', 'atemp', 'humidity', 'windspeed']
    min_max_scaler = RobustScaler()

    for column in columns_to_standardize:
        data[column] = min_max_scaler.fit_transform(data[column])

    return data


def reverse_log(y_values):
    y_values = np.exp(y_values) - 1
    return y_values


def select_best_model(xgb_models, results_rsmle):
    min_value = min(results_rsmle.values())
    for element in results_rsmle:
        if results_rsmle[element] == min_value:
            min_key = element
            break
    best_xgb = joblib.load(xgb_models[min_key])

    return best_xgb


def preprocess_save_data():
    #  Read (and preprocess) the training data.
    train_x, train_y = read_in_training_data()
    size_training = train_x.shape[0]
    test_x = read_in_testing_data()
    data = pd.concat([train_x, test_x], axis=0)
    data.index = range(0, data.shape[0])

    data = interpolate_missing_data(data)
    data = feature_engineering(data)
    data = standardize_columns(data)

    # Separate the data again into train and real test
    training_rows = range(0, size_training)
    testing_rows = range(size_training, (data.shape[0]))
    train_x = data.iloc[training_rows, :]
    test_x = data.iloc[testing_rows, :]

    train_y = perform_log_on_output(train_y)

    # Save the data.
    train_x.to_pickle('train_x')
    train_y.to_pickle('train_y')
    test_x.to_pickle('test_x')
    return train_x, train_y, test_x


def develop_model():
    redo_preprocessing = 0
    if redo_preprocessing:
        train_x, train_y, test_x = preprocess_save_data()
    else:
        train_x = pd.read_pickle('train_x')
        train_y = pd.read_pickle('train_y')
        test_x = pd.read_pickle('test_x')

    # Separate training and local validation set for the three dependant variables.
    train_x_casual, train_y_casual, validation_x_casual, validation_y_count = extract_validation_set(train_x, train_y['count'])
    train_x_registered, train_y_registered, validation_x_registered, validation_y_registered = extract_validation_set(train_x, train_y['registered'])
    train_x_casual, train_y_casual, validation_x_casual, validation_y_casual = extract_validation_set(train_x, train_y['casual'])

    # Add noise to dependent variable.
    noise_level = 1
    train_y_casual = add_log_noise(train_y_casual, noise_level)
    noise_level = 1
    train_y_registered = add_log_noise(train_y_registered, noise_level)

    # First, we develop the model for registered users.
    xgb_models = train_xgb_bagging_classifier(train_x_registered, train_y_registered)

    # After training many models, we identify the best by their rmsle result on the validation set.
    results_rsmle = perform_prediction_on_validation(xgb_models, validation_x_registered, validation_y_registered)
    best_xgb = select_best_model(xgb_models, results_rsmle)
    predictions_best_registered = best_xgb.predict(validation_x_registered)
    predictions_registered = predict_test_x(best_xgb, test_x)

    # Invert the log transformation.
    predictions_best_registered = pd.DataFrame(predictions_best_registered)
    predictions_best_registered = pd.DataFrame(np.round(np.exp(predictions_best_registered) - 1)).stack()

    # Secondly, we develop the model for casual users.
    xgb_models = train_xgb_bagging_classifier(train_x_casual, train_y_casual)

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

develop_model()