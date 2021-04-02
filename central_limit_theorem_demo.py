# Last amended: 2nd April, 2021
# Objective: Drawing kernel density curves
#
# Illustrates Central limit theorem
#     Steps:
#           i) Let students assume any discrete
#              prob distribution. For example a pdf
#              that creates an array, such as: [10,20
#              20,21,20]  
#           ii)Draw random samples from this population 
#          iii)Take a mean of the sample
#           iv)Draw a density plot of these means
#            v)As no of samples (or no of sample means)
#              increase the density plot more and more
#              resembles normal distribution. 






# Ref: Wikipedia: https://en.wikipedia.org/wiki/Kernel_density_estimation#Example
#
# standard deviation of normal
# curve on each point is calculated as:
# value = min(std, iqr/1.34)

import numpy as np
from scipy import stats
r =  np.array([-2.1, -1.3, -0.4,  1.9,  5.1,  6.2])
r_std = np.std(r)
r_std        # 3.1515428320462697
iqr = stats.iqr(r)
iqr/1.34         # 4.01
std = 0.9 * (iqr/1.34) * (len(r) ** -0.2)
std     # 2.5228180500429214
#########################
# Illustration of Central Limit Theorem
# Wikipedia:
#   https://en.wikipedia.org/wiki/Illustration_of_the_central_limit_theorem#Illustration_of_the_discrete_case
# #####################

# Begin with asking students
# any discrete probability distribution

#   Let us say, our distribution
#    has random values from  7, 30 or 20
#     (or may, be: 7,7,7,30,20)

import seaborn as sns
s = np.array([7,7,30,20])
np.random.choice(s)

s_means = []       # collects 'j' sample-means of samples, each having 'i' items
for j in range(30):
    item= []
    for i in range(30):
        item.append(np.random.choice(s))
    s_means.append(np.mean(item))

# So plot now density curve of sample means
sns.kdeplot(s_means) ;
########################################
