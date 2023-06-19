"""
A function that receives as an input two dates, the start and the end date, 
and samples a day inbetween

1st Requirement
The mode and mean of the probablity density function should be the date
exactly in the middle between start_date and end_date.

2nd Requirement
The probablity density function according to which the function samples a
date should be symmetric around the mean.

3rd Requirement
Sampling dates before start_date or after end_date should not possible.

"""

import numpy as np
import matplotlib.pyplot as plt
import datetime
import time # for plot in skeleton code
    
# Set random seed for reproducibility
np.random.seed(42)   

# Define a function that receives as an input two dates, 
# the start and the end date, and samples a day inbetween
def getRandomDate(start_date: str, end_date: str):
    
    # Convert str dates to datetime
    start_date_datetime = datetime.datetime.strptime(start_date,'%d.%m.%Y').date()
    end_date_datetime = datetime.datetime.strptime(end_date,'%d.%m.%Y').date()
    
    # Count number of days in between
    delta = end_date_datetime - start_date_datetime
    dates_inbetween = delta.days - 1

    # Get median date
    mdn = np.floor(dates_inbetween/2)
    mdn_datetime =  start_date_datetime + datetime.timedelta(days=mdn)

    # Set mean and sigma of normal distribution
    mean = 0 
    sigma = 1
    
    # Sample dates inside specific range
    while True:
        # Sample from normal distribution and shift to mdn (!!!!!)
        sample = np.random.normal(mean, sigma, size=1) + mdn
        
        # Round sample to nearest integer
        n_sample = int(np.round(sample))
        
        # Convert n_sample to datetime and check if it's within range
        sample_datetime = start_date_datetime + datetime.timedelta(days=n_sample)
        if start_date_datetime <= sample_datetime <= end_date_datetime:
            break
    
    # Return sample date in specific format
    sample_str = sample_datetime.strftime('%d.%m.%Y')
    return sample_str

# Plotting of distribution of dates
dates = [getRandomDate("1.1.2016", "10.01.2016") for _ in range(10000)]
dates.sort(key=lambda x: time.mktime(time.strptime(x,'%d.%m.%Y')))
fig, ax = plt.subplots(figsize=(10, 6))
plt.hist(dates, bins=9)
ax.set_ylabel('Frequency')
ax.set_xlabel('Date')
plt.show()