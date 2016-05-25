import numpy as np
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import csv
import datetime
import os
import random
import pandas as pd


def cluster_hours_registered(hours):
    counter = 0
    for element in hours:
        if element < 8:
            hours[counter] = 0
        elif element == 8 :
            hours[counter] = 1
        elif element == 9:
            hours[counter] = 2
        elif element < 18:
            hours[counter] = 3
        elif element <= 21:
            hours[counter] = 4
        elif element < 24:
            hours[counter] = 5
        counter = counter + 1

    return hours


def cluster_hours_casual(hours):
    counter = 0
    for element in hours:
        if element < 6.5:
            hours[counter] = 0
        elif element < 8.5:
            hours[counter] = 1
        elif element < 9.5:
            hours[counter] = 2
        elif element < 10.5:
            hours[counter] = 3
        elif element < 19.5:
            hours[counter] = 4
        elif element < 21.5:
            hours[counter] = 5
        elif element < 24:
            hours[counter] = 6
        counter = counter + 1

    return hours


def extract_date_information(data):
    replacement_list_hour = []
    replacement_list_day = []
    replacement_list_month = []
    replacement_list_year = []

    for element in data[0]:
        try:
            extract_hour = datetime.datetime.strptime(element.split()[1], '%H:%M:%S').time()
            extract_hour = extract_hour.hour
        except:
            extract_hour = []
        replacement_list_hour.append(extract_hour)

        try:
            day = datetime.datetime.strptime(element.split()[0], '%Y-%m-%d').strftime('%w')
        except:
            day = []
        replacement_list_day.append(day)

        try:
            extract_month = datetime.datetime.strptime(element.split()[0], '%Y-%m-%d')
            extract_month = extract_month.month
        except:
            extract_month = []
        replacement_list_month.append(extract_month)

        try:
            extract_year = datetime.datetime.strptime(element.split()[0], '%Y-%m-%d')
            extract_year = extract_year.year
        except:
            extract_year = []
        replacement_list_year.append(extract_year)

    replacement_list_day = [int(x) for x in replacement_list_day]

    replacement_list_hour = np.asarray(replacement_list_hour)
    replacement_list_day = np.asarray(replacement_list_day)
    replacement_list_month = np.asarray(replacement_list_month)
    replacement_list_year = np.asarray(replacement_list_year)

    date_information = np.vstack([replacement_list_hour, replacement_list_day, replacement_list_month, replacement_list_year])

    return date_information


def read_data(rel_path, type_of_data):
    if 'train' in type_of_data:
        number_of_columns = 12
    elif 'test' in type_of_data:
        number_of_columns = 9

    script_dir = os.path.dirname(__file__)
    abs_file_path = os.path.join(script_dir, rel_path)

    content = [[] for x in range(number_of_columns)]

    with open(abs_file_path) as csvfile: # , newline=''
        contreader = csv.reader(csvfile,delimiter= ',')
        starting_row = 1
        for row in contreader:
            if starting_row == 1:
                starting_row = 0
            else:
                i = 0
                first_element = 1
                for element in row:
                    if first_element == 1:
                        first_element = 0
                    else:
                        element = float(element)

                    content[i].append(element)
                    i = i + 1
    return content


def create_train_x_train_y(data):
    number_of_columns = len(data)
    trainY = data[number_of_columns-1]
    del data[number_of_columns-1]

    trainX = data
    return trainX, trainY


def read_in_training_data(model_type):

    source = "data/train(1).csv"

    columns_to_be_deleted = [12, 13]

    data = read_data(source, type_of_data='train')

    train_x, train_y = create_train_x_train_y(data)
    date_information = extract_date_information(train_x)

    del train_x[0]

    train_x = np.array(train_x)
    train_x = np.vstack([date_information, train_x])
    train_x = train_x.transpose()

    train_y = np.array(train_y)
    train_y.astype(np.int)

    cluster_hours = 0

    if 'registered' in model_type:
        count_cas_reg = train_x[:, 13]
        if cluster_hours == 1:
            train_x[:, 0] = cluster_hours_registered(train_x[:, 0])

    elif 'casual' in model_type:
        count_cas_reg = train_x[:, 12]

    train_x = delete_not_allowed_info(train_x, columns_to_be_deleted)

    train_y = count_cas_reg
    return train_x, train_y


def read_in_test_data(model_type, args):
    if len(args) == 0:
        columns_to_be_deleted = []
    else:
        columns_to_be_deleted = args

    source = "data/test.csv"
    data = read_data(source, type_of_data='test')

    date_information = extract_date_information(data)
    submission_dates = data[0]
    del data[0]

    train_x = np.array(data)
    train_x = np.vstack([date_information, train_x])
    train_x = train_x.transpose()

    cluster_hours = 0
    if cluster_hours == 1:
        if 'registered' in model_type:
            print('clustering hours deactivated!')

        elif 'casual' in model_type:
            train_x[:, 0] = cluster_hours_casual(train_x[:, 0])

    train_x = delete_not_allowed_info(train_x, columns_to_be_deleted)
    return train_x, submission_dates


def delete_not_allowed_info(data, columns_to_be_deleted):
    columns_to_be_deleted = columns_to_be_deleted[::-1]
    for element in columns_to_be_deleted:
        if not(element > data.shape[1]-1):
            data = np.delete(data, element, 1)
        else:
            print("Wrong index for delete.")
    return data


def run_single_model_registered():
    columns_to_delete_test = []

    model_type = 'registered'
    train_x, train_y = read_in_training_data(model_type)
    test_x, submission_dates = read_in_test_data(model_type, columns_to_delete_test)

    train_x = feature_engineering(train_x)
    test_x = feature_engineering(test_x)

    percentage_noise = 1
    train_y = add_noise(train_y, percentage_noise)

    np.save('registered_users_trainX.npy', train_x)
    np.save('registered_users_trainY.npy', train_y)
    np.save('registered_users_testX.npy', test_x)
    np.save('csv_submission_dates.npy', submission_dates)


def categorize_years(data_set, *args):
    new_variable = np.zeros([data_set[:, 0].shape[0], 1])
    first_chunk = [0, 1, 2, 3]
    second_chunk = [4, 5, 6, 7]
    third_chunk = [8, 9, 10, 11]

    counter = 0
    for year in data_set[:, 3]:
        if year == 2011 and data_set[counter, 2] in first_chunk:
            new_variable[counter] = 1
        elif year == 2011 and data_set[counter, 2] in second_chunk:
            new_variable[counter] = 2
        elif year == 2011 and data_set[counter, 2] in third_chunk:
            new_variable[counter] = 3
        elif year == 2012 and data_set[counter, 2] in first_chunk:
            new_variable[counter] = 4
        elif year == 2012 and data_set[counter, 2] in second_chunk:
            new_variable[counter] = 5
        elif year == 2012 and data_set[counter, 2] in third_chunk:
            new_variable[counter] = 6
        counter = counter + 1

    return_separated_ds = 0
    if return_separated_ds == 1:
        index_chunk1 = np.where(new_variable == 1)[0]
        index_chunk2 = np.where(new_variable == 2)[0]
        index_chunk3 = np.where(new_variable == 3)[0]
        index_chunk4 = np.where(new_variable == 4)[0]
        index_chunk5 = np.where(new_variable == 5)[0]
        index_chunk6 = np.where(new_variable == 6)[0]


        dataset_1 = data_set[index_chunk1, :]
        dataset_1 = pd.DataFrame(dataset_1)

        dataset_2 = data_set[index_chunk2, :]
        dataset_2 = pd.DataFrame(dataset_2)

        dataset_3 = data_set[index_chunk3, :]
        dataset_3 = pd.DataFrame(dataset_3)

        dataset_4 = data_set[index_chunk4, :]
        dataset_4 = pd.DataFrame(dataset_4)

        dataset_5 = data_set[index_chunk5, :]
        dataset_5 = pd.DataFrame(dataset_5)

        dataset_6 = data_set[index_chunk6, :]
        dataset_6 = pd.DataFrame(dataset_6)

        if 'args' in locals():
            train_y = np.asarray(args[0])
            train_y_1 = train_y[index_chunk1]
            train_y_2 = train_y[index_chunk2]
            train_y_3 = train_y[index_chunk3]
            train_y_4 = train_y[index_chunk4]
            train_y_5 = train_y[index_chunk5]
            train_y_6 = train_y[index_chunk6]

            index_new_dim = dataset_1.shape[1] + 1
            dataset_1[index_new_dim] = train_y_1

            index_new_dim = dataset_2.shape[1] + 1
            dataset_2[index_new_dim] = train_y_2

            index_new_dim = dataset_3.shape[1] + 1
            dataset_3[index_new_dim] = train_y_3

            index_new_dim = dataset_4.shape[1] + 1
            dataset_4[index_new_dim] = train_y_4

            index_new_dim = dataset_5.shape[1] + 1
            dataset_5[index_new_dim] = train_y_5

            index_new_dim = dataset_6.shape[1] + 1
            dataset_6[index_new_dim] = train_y_6

        return dataset_1, dataset_2, dataset_3, dataset_4, dataset_5, dataset_6

    else:
        data_set = np.hstack([data_set, new_variable])
        return data_set


def add_var_free_or_not(data_set):
    weekend = [0, 6]
    new_variable = np.zeros([data_set[:, 0].shape[0], 1])

    counter = 0
    for element in data_set[:, 1]:
        if element in weekend or data_set[counter, 6] == 0:
            new_variable[counter] = 1
        counter = counter + 1

    data_set = np.hstack([data_set, new_variable])
    print('Variable added for the type of day')
    return data_set


def add_var_daytipe(data_set):
    new_variable = np.zeros([data_set[:, 0].shape[0], 1])
    counter = 0
    for element in data_set[:, 5]:
        if element == 1:
            new_variable[counter] = 0

        elif element == 1 and data_set[counter, 6] == 0:
            new_variable[counter] = 1

        elif element == 0 and data_set[counter, 6] == 1:
            new_variable[counter] = 2

        counter = counter + 1

    data_set = np.hstack([data_set, new_variable])
    print('Variable added for the type of day')
    return data_set


def add_var_day_or_we(data_set):
    weekend = [0, 6]
    new_variable = np.zeros([data_set[:, 0].shape[0], 1])

    counter = 0
    for element in data_set[:, 1]:
        if element in weekend:
            new_variable[counter] = 1
        counter = counter + 1

    data_set = np.hstack([data_set, new_variable])
    print('Variable added for whether day is weekday or weekend.')
    return data_set


def feature_engineering(data_set):
    data_set = add_var_free_or_not(data_set)
    data_set = add_var_day_or_we(data_set)
    data_set = add_var_daytipe(data_set)
    data_set = categorize_years(data_set)
    data_set = categorize_years_more(data_set)
    data_set = categorize_hours(data_set)
    return data_set


def categorize_hours(data_set):
    container1 = [0, 1]
    container2 = [2, 3]
    container3 = [4, 5]
    container4 = [6, 7]
    container5 = [8, 9]
    container6 = [10, 11]
    container7 = [12, 13]
    container8 = [14, 15]
    container9 = [16, 17]
    container10 = [18, 19]
    container11 = [20, 21]
    container12 = [22, 23]

    counter = 0
    new_var = np.zeros([data_set[:, 0].shape[0], 1])
    for element in data_set[:, 0]:
        if element in container1:
            new_var[counter] = 1
        elif element in container2:
            new_var[counter] = 2
        elif element in container3:
            new_var[counter] = 3
        elif element in container4:
            new_var[counter] = 4
        elif element in container5:
            new_var[counter] = 5
        elif element in container6:
            new_var[counter] = 6
        elif element in container7:
            new_var[counter] = 7
        elif element in container8:
            new_var[counter] = 8
        elif element in container9:
            new_var[counter] = 9
        elif element in container10:
            new_var[counter] = 10
        elif element in container11:
            new_var[counter] = 11
        elif element in container12:
            new_var[counter] = 12
        counter = counter + 1

    data_set = np.hstack([data_set, new_var])
    return data_set


def categorize_years_more(data_set, *args):
    new_variable = np.zeros([data_set[:, 0].shape[0], 1])
    # column Year = 4
    first_chunk = [0, 1]
    second_chunk = [2, 3]
    third_chunk = [4, 5]
    fourth_chunk = [6, 7]
    fifth_chunk = [8, 9]
    sixth_chunk = [10, 11]

    counter = 0
    for year in data_set[:, 3]:
        if year == 2011 and data_set[counter, 2] in first_chunk:
            new_variable[counter] = 1
        elif year == 2011 and data_set[counter, 2] in second_chunk:
            new_variable[counter] = 2
        elif year == 2011 and data_set[counter, 2] in third_chunk:
            new_variable[counter] = 3
        elif year == 2011 and data_set[counter, 2] in fourth_chunk:
            new_variable[counter] = 4
        elif year == 2011 and data_set[counter, 2] in fifth_chunk:
            new_variable[counter] = 5
        elif year == 2011 and data_set[counter, 2] in sixth_chunk:
            new_variable[counter] = 6

        elif year == 2012 and data_set[counter, 2] in first_chunk:
            new_variable[counter] = 7
        elif year == 2012 and data_set[counter, 2] in second_chunk:
            new_variable[counter] = 8
        elif year == 2012 and data_set[counter, 2] in third_chunk:
            new_variable[counter] = 9
        elif year == 2012 and data_set[counter, 2] in fourth_chunk:
            new_variable[counter] = 10
        elif year == 2012 and data_set[counter, 2] in fifth_chunk:
            new_variable[counter] = 11
        elif year == 2012 and data_set[counter, 2] in sixth_chunk:
            new_variable[counter] = 12
        counter = counter + 1

    data_set = np.hstack([data_set, new_variable])
    return data_set


def run_single_model_casual():
    columns_to_delete_train = [12, 13]
    columns_to_delete_test = []

    model_type = 'casual'
    train_x, train_y = read_in_training_data(model_type, columns_to_delete_train)
    test_x, submission_dates = read_in_test_data(model_type, columns_to_delete_test)

    train_x = feature_engineering(train_x)
    test_x = feature_engineering(test_x)

    percentage_noise = 90
    train_y = add_noise(train_y, percentage_noise)

    np.save('casual_users_trainX.npy', train_x)
    np.save('casual_users_trainY.npy', train_y)
    np.save('casual_users_testX.npy', test_x)


def add_noise(y_train, percentage_noise):
    percentage_noise = 1 - percentage_noise
    counter = 0
    maximum_noise = 10

    for element in y_train:
        random_number = np.random.randint(0, 100)
        if random_number > percentage_noise:
            multiplicator = (np.random.randint(0, 200) - 100) / float(1000)
            noise_to_be_added = multiplicator * y_train[counter]
            noise_to_be_added = check_boundary_noise(noise_to_be_added, maximum_noise)
            y_train[counter] = round(y_train[counter] + noise_to_be_added)
            if y_train[counter] < 0:
                y_train[counter] = 0
        counter = counter + 1

    return y_train


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


def remove_negative_predictions(pred):
    counter = 0
    for element in pred:
        if element < 0:
            pred[counter] = 0
        counter = counter + 1
    return pred


def prepare_input_registered():
    train_x = np.load('registered_users_trainX.npy')
    train_y = np.load('registered_users_trainY.npy')
    test_x = np.load('registered_users_testX.npy')

    train_y = train_y.reshape(1, -1)
    train_y = np.transpose(train_y)

    return train_x, train_y, test_x


def prepare_input_casual():
    train_x = np.load('casual_users_trainX.npy')
    train_y = np.load('casual_users_trainY.npy')
    test_x = np.load('casual_users_testX.npy')

    train_y = train_y.reshape(1, -1)
    train_y = np.transpose(train_y)

    return train_x, train_y, test_x


def train_model_registered(train_x_full, train_y_full):
    model = xgb.XGBRegressor(n_estimators=100000, silent=0, max_depth=6, learning_rate=0.005, min_child_weight=10,
                             subsample=0.8, colsample_bylevel=0.8)

    xgb.XGBModel()
    train_y_full = np.log(train_y_full)

    x_train, x_test, y_train, y_test = train_test_split(train_x_full, train_y_full, random_state=1, test_size=0.10)
    y_train = remove_below_zero_entries(y_train)
    y_test = remove_below_zero_entries(y_test)

    model = model.fit(x_train, y_train,  eval_set=[(x_test, y_test)], verbose=True, early_stopping_rounds=10)
    return model


def remove_below_zero_entries(data):
    counter = 0
    for element in data:
        if element < 0:
            data[counter] = 0
        counter = counter + 1

    return data


def train_model_casual(train_x_full, train_y_full):
    model = xgb.XGBRegressor(n_estimators=10000, silent=0, max_depth=5, learning_rate=0.005, min_child_weight=10,
                             subsample=0.8, colsample_bylevel=0.8)

    xgb.XGBModel()
    train_y_full = np.log(train_y_full)

    x_train, x_test, y_train, y_test = train_test_split(train_x_full, train_y_full, random_state=1, test_size=0.10)
    y_train = remove_below_zero_entries(y_train)
    y_test = remove_below_zero_entries(y_test)

    model = model.fit(x_train, y_train,  eval_set=[(x_test, y_test)], verbose=True, early_stopping_rounds=10)
    return model


def predict_test_x(model, test_x):
    predictions = model.predict(test_x)
    predictions = np.exp(predictions)
    predictions = np.round(predictions)
    return predictions


def main():
    #  First, we create the input data.
    run_single_model_registered()
    run_single_model_casual()

    #  Here, we build the model for registered users.
    train_x, train_y, text_x = prepare_input_registered()
    model = train_model_registered(train_x, train_y)
    predictions_registered = predict_test_x(model, text_x)

    #  Here, we build the model for casual users.
    train_x, train_y, text_x = prepare_input_casual()
    model = train_model_casual(train_x, train_y)
    predictions_casual = predict_test_x(model, text_x)

    #  We create the final submission file.
    predictions = np.sum(np.vstack([predictions_registered, predictions_casual]), axis=0)
    submission_dates = np.load('csv_submission_dates.npy')
    submission_data = create_submission_data(submission_dates, predictions)
    filename = 'rf_submission.csv'
    create_submission_file(submission_data, filename)

    print('EOF')

main()
