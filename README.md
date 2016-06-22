# Bike Sharing Demand
Kaggle competition on bike sharing demand. https://www.kaggle.com/c/bike-sharing-demand

In the competition, bike rentals of some given time points need to be predicted. The training set consists of bike rentals per hour for the first 20 days of each calendar month for 2012 and 2013. The test set comprises of the remaining days of each month. The test set seems to be quite different from the training set as the local rmsle was always much lower (0.33) than the public score (0.40).

Three dependent variables are given. Count is the total number of bike rentals. Moreover, a separate rental count is given for registered and casual users. Both users have very different characteristics. Registered users follow a very stable usage pattern. They take their bike to work in the morning and also driven back in the evening. Casual users mostly rent bikes on the weekend (preferrably Sundays). Weather, season, holidays and year have an effect on the number of rents. 

For registered and casual users, a separate xgb boost is estimated. XGB hyperparameters are found using random initialisations and selecting the best based on their performance on a separate validation set. Some feature engineering is performed (see feature engineering method).

The final score on the PL is: 0.40693 (within the top10% submissions).


![alt tag](https://github.com/drawer87/kaggle/blob/master/kaggle_score.jpg)

