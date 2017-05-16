# Bike Sharing Demand
This is a project page for the kaggle competition on bike sharing demand. (https://www.kaggle.com/c/bike-sharing-demand)

#  Background
In the competition, bike rentals per hour need to be predicted. The training set consists of bike rentals per hour for the first 20 days of each calendar month for 2012 and 2013. The test set consists of the remaining days of each month of both years.

The following independent variables are given (taken from the project page):  
__'datetime'__ - hourly date + timestamp  
__'season'__ -  1 = spring, 2 = summer, 3 = fall, 4 = winter  
__'holiday'__ - whether the day is considered a holiday  
__'workingday'__ - whether the day is neither a weekend nor holiday  
__'weather'__ - 1: Clear, Few clouds, Partly cloudy, Partly cloudy  
2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist  
3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds  
4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog   
__'temp'__ - temperature in Celsius  
__'atemp'__ - "feels like" temperature in Celsius  
__'humidity'__ - relative humidity  
__'windspeed'__ - wind speed  

The dataset has a low number of features. Feature engineering was performed to drastically improve the model performance.

Three dependent variables are given. 
'count' is the total number of bike rentals per hour ('registered' + 'casual'). 
'registered' is the number of bike rentals per hour of registered users.
'casual' is the number of bike rentals per hour of casual users.

#  Exploratory data analysis

In order to get a better understanding of the dataset, I created a number of visualization. Some visualizations
might not look very interesting but they are helpful to double-check the data (even though kaggle datasets
are normally rather clean).
 
__Is the depent variable normally distributed?__  
The count variable is skewed to the right. There are many hours in which very few bikes were rented out and 
there are fewer hours where a lot of bikes are rented out. Decision trees are a good choice for unbalanced datasets. It should also be noted that there a number of outliers present (e.g. values above 800, see boxplot outliers). These might be holidays with especially good weather. We will take care of these outliers in a later step. 

![alt tag](https://github.com/drawer87/kaggle/blob/master/dependent_var.jpg)

__Do both years show a similar pattern?__  
The dataset has rental records of 2011 and 2012. It is worthwile to check if people show similar behaviour in both years. In case of a stark difference between the years, it might be necesarry to build separate models for each year. This
does not seem to be needed with the present dataset.

![alt tag](https://github.com/drawer87/kaggle/blob/master/year_comparison_rentals.jpg)


The dataset contains rentals per hour for the casual and registered group separately. In the following, we will check if
it is needed to develop separate models for each group.

__Do casual and registered users account for the same number of rentals?__  
Registered users seem to make up for a much larger proportion of the total rentals per hour. Moreover,
the spread in rentals per hour is larger. 


![alt tag](https://github.com/drawer87/kaggle/blob/master/rentals_registered_casual.jpg)

__Is there a difference per calendar day?__  
There does not seem to be a huge difference per hourly rentals for different days of a given month. Registered and casual
users show different behaviours again.

![alt tag](https://github.com/drawer87/kaggle/blob/master/rentals_per_calendar_day.jpg)


__Is there a difference in rentals per week day?__  
We can see a clear difference in the behaviour of both groups. Registered users follow a more predictable pattern. They
might rent the bike to travel to work and they rent fewer bikes on the weekend. Casual users seem to be users who rent their bikes mostly on (or shortly before/after) weekends.

![alt tag](https://github.com/drawer87/kaggle/blob/master/rentals_per_weekday.jpg)


__Does the daily usage pattern differ between registered and casual users?__  

![alt tag](https://github.com/drawer87/kaggle/blob/master/rentals_per_hour_casual.jpg)

Both groups substantially differ regarding when they rent a bike. Casual users rent bikes in the late morning and they mostly use them during the daytime. Registered users seem to rent bikes to get to work in the morning and return to their homes in the late afternoon.

__Conclusion:__  

Both user groups have very different characteristics. Registered users follow a very stable usage pattern. They take their bike to work in the morning and also driven back in the evening. Casual users mostly rent bikes on the weekend (preferrably Sundays). Weather, season, holidays and year have an effect on the number of rents. 

__What is the influence of the weather?__  

In the 'weather' variable the weather is categorized into 4 different conditions. Not surprisingly, people
rent more bikes when the weather is good but even during light snow or thunderstorm people still rent bikes.
![alt tag](https://github.com/drawer87/kaggle/blob/master/weather_condition_rentals.jpg)


I have visualized the total bike rentals for each continous weather factor as well. It seems to hold that
the higher the temperature/air temperature, the more bikes are rented out. Regarding windspeed, there seems to be an
optimum between 20-30. The most comfortable humidity seems to be between 20-30 and continuously decreases the more humid it becomes.
![alt tag](https://github.com/drawer87/kaggle/blob/master/weather_factors.jpg)


#  Feature Engineering

Based on the EDA (and the help of kaggle forums), I decided to try to develop the following additional features.

__datetime__: I have separated the original datetime string into year, months, days and hours. This is very important. Usage behaviours differ for each hour, day, month and year. Decision trees would have a much worse performance without these additional branching factors. The decision tree can now build separate predictors for different timepoints.  
__datetime_binned__ : Casual and registered users show a stable, idiosyncratic pattern of activity during the day. I decided
to bin together hours of similar renting behaviour.  
__is_weekend__: A vector indicating if it is weekend.  
__is_free__: A vector indicating if the day is off (weekend or holiday).  
__day_type__: Type of the day  
1: normal working day, non-holiday  
2. working day, holiday  
3. non-working day, holiday  
4. non-working day, non-holiday.  
__cluster_month__: Clustering together months.  
__cluster_temp__: Clustering the temperature.  
__cluster_atemp__: Clustering the atemp.  
__cluster_humidity__: Clustering the humidity.  
__cluster_windspeed__: Clustering the windspeed.  
 __good_day__: Some days have great weather and more people are renting out bikes. (10 < windspeed < 25; 10 < humidity < 40; 25 < temp < 30)  
__dew__: The dew measure is a more reliable way of measuring humidity (because it is an absolute measure as opposed to relative humidity) and comfort for humans.

We also a small percentage of noise to our outcome variable to improve generalization performance.

#  Model Development

For registered and casual users, a separate xgb boost was developed. The last two days of each month were used as a private test set. XGB classifiers were developed on 90% the remaining days per month and the validation set consisted of the remaining 10%. The validation set was used to perform early stopping to prevent overfitting.  

The best combination of XGB hyperparameters (including max_tree_depths, learning_rate, min_child_weight, etc.) is not difficult to find. We performed a randomized search and developed hundreds of classifiers. All classifiers were tested on our test set. 
For each group (casual, registered), the best performing classifier was identified and predicted were added (i.e. predicted_casual_rentals + predicted_registered_rentals = predicted_total_rentals). The best performing classifier was used to predict the labels for the real test set and results were submitted to kaggle. 

#  Model performance  

The test set seems to be quite different from the training set as the local rmsle was always much lower (0.33) than the public score (0.40). The test set consists of the last 10 days of each month and in this time period more holidays or days with a special characteristics might have occured. The final score on the public leaderboard is: 0.40693 (within the top 10% of all submissions).  


![alt tag](https://github.com/drawer87/kaggle/blob/master/kaggle_score.jpg)


#  Future directions  

The randomized hyperparamter search was only run for two hours on a quad-core i7 laptop. A better final score is likely
if the hyperparameter search is run for a longer time. A more diverse set of classifiers could also be tested including logistic regression, svr and neural networks.
