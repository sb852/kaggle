# Bike Sharing Demand
This is a project page for the kaggle competition on bike sharing demand. (https://www.kaggle.com/c/bike-sharing-demand)

#  Background
In the competition, bike rentals per hour need to be predicted. The training set consists of bike rentals per hour for the first 20 days of each calendar month for 2012 and 2013. The test set consists of the remaining days of each month. 

The following independent variables are given (taken from the project page):
'datetime' - hourly date + timestamp  
'season' -  1 = spring, 2 = summer, 3 = fall, 4 = winter 
'holiday' - whether the day is considered a holiday
'workingday' - whether the day is neither a weekend nor holiday
'weather - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 
'temp' - temperature in Celsius
'atemp' - "feels like" temperature in Celsius
'humidity' - relative humidity
'windspeed' - wind speed

Three dependent variables are given. 
'count' is the total number of bike rentals per hour. 
'registered' is the number of bike rentals per hour for registered users.
'casual' is the number of bike rentals per hour for casual users.

#  Exploratory data analysis

In order to get a better understanding of the dataset, I created a number of visualization.
 
`test`
Conclusion:

Both users have very different characteristics. Registered users follow a very stable usage pattern. They take their bike to work in the morning and also driven back in the evening. Casual users mostly rent bikes on the weekend (preferrably Sundays). Weather, season, holidays and year have an effect on the number of rents. 

#  Feature Engineering


#  Model Development


For registered and casual users, a separate xgb boost is estimated. XGB hyperparameters are found using random initialisations and selecting the best based on their performance on a separate validation set.

#  Model performance

The test set seems to be quite different from the training set as the local rmsle was always much lower (0.33) than the public score (0.40). The final score on the public leaderboard is: 0.40693 (within the top10% of all submissions).


![alt tag](https://github.com/drawer87/kaggle/blob/master/kaggle_score.jpg)

