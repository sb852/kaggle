import unittest
import os
import sys
import numpy as np
src_path = os.path.normpath(os.getcwd() + os.sep + os.pardir) + '/src'
sys.path.append(src_path)

class TestAdd(unittest.TestCase):
    """
    Test the functions from the bike sharing repository.
    """
 
    def test_read_in_training_data(self):
        """
        Test that the addition of two integers returns the correct total
        """
        from training import read_in_training_data
        training_x, training_y = read_in_training_data()

        self.assertEqual(training_x._typ, 'dataframe')
        self.assertEqual(training_y._typ, 'dataframe')


    def test_read_in_testing_data(self):
        """
        Test that the addition of two integers returns the correct total
        """
        from training import read_in_testing_data
        training_x = read_in_testing_data()

        self.assertEqual(training_x._typ, 'dataframe')


    def test_preprocess_save_data(self):
        """
        We are testing the entire preprocessing pipeline. This is an integration test.
        """

        from training import preprocess_save_data

        perform_preprocessing = True
        train_x, train_y, test_x = preprocess_save_data(perform_preprocessing)

        #  We expect to have three label colunms and 10866 observations.
        self.assertEqual(train_y.shape[1], 3)
        self.assertEqual(train_y.shape[0], 10886)

        self.assertEqual(train_x.shape[0], 10886)

        #  We expect the following columns to be present in the training data.
        present_columns = ['hour', 'day_of_year', 'day_of_month', 'day_of_week', 'day_off', 'day',
                           'is_weekend', 'hours_categorized_2']

        test = [column_name in train_x.columns for column_name in present_columns]
        self.assertEqual(np.unique(test), True)

        self.assertEqual(np.unique(train_x['hour']).size, 24)
        self.assertEqual(np.unique(train_x['day_of_month']).size, 12)
        self.assertEqual(np.unique(train_x['day_of_week']).size, 7)

        self.assertEqual(np.unique(train_x['day_off']).size, 2)
        self.assertEqual(np.unique(train_x['is_weekend']).size, 2)
        self.assertEqual(np.unique(train_x['day']).size, 19)

        self.assertEqual(np.unique(train_x['dew_binned']).size, 3)

        #  We expect the following columns to be present in the testing data.
        present_columns = ['hour', 'day_of_year', 'day_of_month', 'day_of_week', 'day_off', 'day',
                           'is_weekend', 'hours_categorized_2']

        test = [column_name in test_x.columns for column_name in present_columns]
        self.assertEqual(np.unique(test), True)

        self.assertEqual(np.unique(test_x['hour']).size, 24)
        self.assertEqual(np.unique(test_x['day_of_month']).size, 12)
        self.assertEqual(np.unique(test_x['day_of_week']).size, 7)

        self.assertEqual(np.unique(test_x['day_off']).size, 2)
        self.assertEqual(np.unique(test_x['is_weekend']).size, 1)
        self.assertEqual(np.unique(test_x['day']).size, 19)

        self.assertEqual(np.unique(test_x['dew_binned']).size, 3)


    def test_produce_random_parameters(self):
        """
        We are testing the production of randomized hyperparameters.
        """
        from training import produce_random_parameters

        #  We expect the following columns to be present.
        columns_expected = ['max_depth', 'n_estimators', 'learning_rate', 'subsample', 'colsample_by_level',
                            'min_child_weight', 'min_child_weight', 'early_stopping', 'size_xgb_val']
        parameters = produce_random_parameters()

        test = [column_name in parameters.keys() for column_name in columns_expected]
        self.assertEqual(np.unique(test), True)


    def test_perform_log_on_output(self):
        """
        We are testing if the log transformation worked.
        """
        from training import perform_log_on_output

        example_value = np.random.randint(0, 1000)
        output_value = perform_log_on_output(example_value)

        expected_value = np.log(example_value + 1)

        self.assertEqual(output_value, expected_value)


if __name__ == '__main__':
    unittest.main()
