import unittest
import os
import sys
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
        We are testing the entire preprocessing pipeline.
        """

        from training import preprocess_save_data

        perform_preprocessing = True
        train_x, train_y, test_x = preprocess_save_data(perform_preprocessing)

        print('x')



if __name__ == '__main__':
    unittest.main()
