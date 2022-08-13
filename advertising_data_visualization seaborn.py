# Last amended: 10th Feb 2021
# Data source: Kaggle: https://www.kaggle.com/fayomi/advertising
#
# Spyder note: Use F9 in Spyder to execute a line and advance
#
# objectives:
#           Discover relationships and existence of pattern
#           in data
#              i)  Feature Engineering for categorical variables
#             ii)  Extracting datetime components
#             iii) Behaviour of 'apply' vs 'map' for Series
#              iv)  Learning to draw various types of graphs
#               v)  Conditional plots using catplot
#              vi)  Relationship plots using relplot
#             vii)  Learning seaborn plots
#            viii)  Looking at structure in data
#                       a. Parallel coordinates plots
#                       b. Radviz plots
#                       c. Andrews curves

# Good reference: https://seaborn.pydata.org/introduction.html

# 1.0 Call libraries
%reset -f
# 1.1 For data manipulations
import numpy as np
import pandas as pd
# 1.2 For plotting
import matplotlib.pyplot as plt
#import matplotlib
#import matplotlib as mpl     # For creating colormaps
import seaborn as sns
# 1.3 For data processing
from sklearn.preprocessing import StandardScaler
# 1.4 OS related
import os

# 1.5
%matplotlib qt5
#%matplotlib inline


# 1.6 Go to folder containing data file
#os.chdir("D:\\data\\OneDrive\\Documents\\advertising")
#os.chdir("/home/ashok/datasets/advertising")
# os.chdir("D:\\data\\OneDrive\\Documents\\advertising")
os.chdir("C:\\Users\\Administrator\\OneDrive\\Documents\\advertising")

os.listdir()            # List all files in the folder

# 1.7 Read file and while reading file,
#      convert 'Timestamp' to datetime time
ad = pd.read_csv("advertising.zip",
                  parse_dates = ['Timestamp']    # especial for date parsing
                  )


# 1.8 Check data types of attributes
ad.dtypes

# 1.9 Some more dataset related information
pd.options.display.max_columns = 100
pd.set_option("display.max.columns" , 100)

ad.head(3)
ad.info()               # Also informs how much memory dataset takes
                        #   and status of nulls
ad.shape                # (1000, 10)
ad.columns.values
len(ad)                 # Also give number of rows

# 1.10 Categorical data value counts
#     Or number of levels per category
len(ad.City.unique())                   # 969 cities out of 1000
ad.City.value_counts()

# 1.10.1 How many conutries
len(ad.Country.unique())                # 237 countries
ad.Country.value_counts()               # Mostly 2 per country

# 1.10.2 Distribution of gender
ad.Male.value_counts()                  # 519:481

# 1.10.3 Distribution of clicks
ad['Clicked on Ad'].value_counts()      # 1 and 0 in the ratio of 500:500
                                        # This is highly optimistic. Genrally clicks may be 1%


#############################
# 2.0 Engineering  features
#############################

## Discretisation
#   a. Cut column 'Age' into 3 equal parts--age_cat
#   b. Cut 'Area Income' into 3 equal parts
#   c. Create a column of length of  'Ad Topic Line'
#   d. Create a column of no of words in  'Ad Topic Line'
## Replace by Category count
#   e. Create a column of 'City' count
#   f. Create a column of 'Country' count
#   g. Create a column of 'City' & 'Country' count
## Date and time columns
#   h. Extract from Timestamp, hour-of-day, weekday, months
## Using 'map'
#   i.Transform 'hour_of_day' to "earlymorning", "morning", "afternoon", "evening", "night","latenight"
#   j. Transform 'weekday' to 0,1,2,3,4,5,6
#   k. Transform months to Qtr1, Qtr2, Qtr3, Qtr4
## Rename columns
#   l. Assign new and shorter column names to a few columns


# 2.1 Descretise continuos columns
#     These are equal width bins as against
#     equal data-points bins (quantile) or kmeans clusters
#     Alternatively use KBinsDiscretizer of sklearn
ad["age_cat"] = pd.cut(
                       ad['Age'],
                       bins = 3,           # Else devise your bins: [0,20,60,110]
                       labels= ["y", "m", "s"]
                      )

ad["area_income_cat"] = pd.cut(
                               ad['Area Income'],
                               bins = 3,
                               labels= ["l", "m", "h"]
                               )

# 2.2 Create a new column as per length of each ad-line
#     Both the following lines do the same thing
ad['AdTopicLineLength'] = ad['Ad Topic Line'].apply(lambda x : len(x))
ad['AdTopicLineLength'] = ad['Ad Topic Line'].map(lambda x : len(x))


# 2.3 Create a new column as per number of words in each ad-line
# Try "good boy".split(" ")  and len("good boy.split(" "))
"good boy".split(" ")             # ['good', 'boy']
len("good boy".split(" "))        # 2
ad['Ad Topic Line'].map(lambda x : len(x.split(" ")))

# 2.3.1 Note the use of apply(). This apply() works on complete Series
#       to transform it rather than to summarise it as in groupby.
ad['AdTopicNoOfWords'] = ad['Ad Topic Line'].apply(lambda x : len(x.split(" ")))   # Note the use of apply()
                                                                                   # This apply works on complete Series

# 2.4 A column that has countd of City and
#       another column with count of Country columns
#       Note the use of transform method here:
#grouped = ad.groupby(['City'])
#ad['City_count'] = grouped['City'].transform('count')   # count is a groupby method

# 2.4.1 Same way for country
#grouped = ad.groupby(['Country'])
#ad['Country_count'] = grouped['Country'].transform('count')   # count is a groupby method


# 2.5 Extract date components using Series.dt accessor
#     https://pandas.pydata.org/pandas-docs/stable/reference/series.html#api-series-dt
#     https://pandas.pydata.org/pandas-docs/stable/reference/series.html#datetime-properties

# 2.6 What is the type of 'dt'
type(ad['Timestamp'].dt)    # Accessor like get()
                            # pandas.core.indexes.accessors.DatetimeProperties

# 2.7 Extract hour, weekday and month
ad['hourOfDay']    = ad['Timestamp'].dt.hour
ad['weekday']      = ad['Timestamp'].dt.weekday
ad['quarter']      = ad['Timestamp'].dt.month # First we get month. Then we map month to quarter
                                              #   See below

# 2.8 Cut hour to morning, evening, night etc
#     For example 0 to 6am is earlymorning

# 2.8.1 For easy interpretation of graphs, use l1
l1 = ["earlymorning", "morning", "afternoon", "evening", "night","latenight"]
# 2.8.2 For Radviz plot and Parallel charts use l2
l2 = [1,2,3,4,5,6]

# 2.8.3
#ad["hour"] = pd.cut(ad['hourOfDay'], bins = [-1,6,12,17,20,22,24], labels = l1)
ad["hour"] = pd.cut(ad['hourOfDay'], bins = [-1,6,12,17,20,22,24], labels = l2)


# 3.0 Similarly for weekdays
#     Map weekday numbers to weekday names
#     We use Series.map() method

mymap = {0 : 'Monday', 1 : 'Tuesday', 2: 'Wednesday',
         3: 'Thursday',4: 'Friday',   5: 'Saturday', 6: 'Sunday' }


# 3.0.1 For easy interpretation of weekdays in graphs
#ad['weekday'] = ad['weekday'].map(mymap)

ad['weekday'].head(2)

# 4.0 We use Series.map() method again but this time instead of supplying
#      a dictionary to dictate transformation, we use a function for
#        transformation
"""
### map vs apply in Series
# https://stackoverflow.com/questions/19798153/difference-between-map-applymap-and-apply-methods-in-pandas
What is desired result here?
Examine each value of month and transform it to
Qtr-1 or Qtr-2 etc. Thus our function is not operating
on a whole Series (like fillna() or like median()) but on
value.
'map' method takes one value from Series at a time.
'map' operates in the following manner:
    a. Pass an element to function
    b. Function processes that value and returns a value
    c. map 'appends' that processed value to earlier processed values
    d. Repeat steps a to c for all values in the Series
    e. At the end, return the processed Series
 Refer: https://stackoverflow.com/a/19798528/3282777

For example:
    ad['month'].map(lambda x: x.max() - x.min())
    Gives an error:

        'int' object has no attribute 'max'
    Same error happens if I use, 'apply'

    ad['month'].apply(lambda x: x.max() - x.min())

Thus, both 'apply' and 'map' methods of Series take a value as
input at a time.

### apply in DataFrame
map() method does not work for DataFrame. 'apply' does.

This 'normalization' works:

    ad[['month','Age']].apply(lambda x: x/(x.max() - x.min()))

That is, apply method of DataFrame takes a Series AND NOT value as input.

"""

# 4.01  For easy interprettaion in charts, use month1(), else month2()
def month1(x):
    if 0 < x <= 3:
        return "Q1"            # Quarter 1
    if 3 < x <= 6:
        return "Q2"            # Quarter 2
    if 6 < x <= 9:
        return "Q3"            # Quarter 3
    if 9 < x <= 12:
        return "Q4"            # Quarter 4

# 4.02
def month2(x):
    if 0 < x <= 3:
        return 1            # Quarter 1
    if 3 < x <= 6:
        return 2            # Quarter 2
    if 6 < x <= 9:
        return 3            # Quarter 3
    if 9 < x <= 12:
        return 4            # Quarter 4


#ad['quarter'] = ad['quarter'].map(lambda x : month1(x))   # Which quarter clicked
ad['quarter'] = ad['quarter'].map(lambda x : month2(x))   # Which quarter clicked
ad[['Timestamp','quarter']].head(2)   # Just check

# 4.1 So finally what are col names?
ad.columns.values
ad.shape               # (1000, 18)  Earlier shape was (1000, 10)

# 4.2 Let us rename some columns; remove spaces

new_col_names  = {
                 'Daily Time Spent on Site' :  'DailyTimeSpentonSite',
                 'Area Income'              : 'AreaIncome',
                 'Daily Internet Usage'     : 'DailyInternetUsage',
                 'Clicked on Ad'            : 'Clicked_on_ad',
                 'Male'                     : 'Gender'
              }
# 4.2.1
ad.rename(
         columns = new_col_names,
         inplace = True,
         #axis = 1             # Note the axis keyword. By default it is axis = 0
         )

ad.head(3)
ad.columns.values

##################
# 5 Plotting
##################
# A summary of syntax of important plots
#---------------------------------------
# 1. sns.displot()
#    (note: sns.distplot() is depreciated)
# https://seaborn.pydata.org/generated/seaborn.distplot.html
# displot(data=None, *, x=None, y=None, hue=None, row=None,
#         col=None, weights=None, kind='hist', rug=False,
#         rug_kws=None, log_scale=None, legend=True, palette=None,
#         hue_order=None, hue_norm=None, color=None, col_wrap=None,
#         row_order=None, col_order=None, height=5, aspect=1, facet_kws=None,
#         **kwargs)
#         kind: 'hist', 'kde', 'ecdf'
###$$$$
# Note: 'displot' DOSES not take 'ax' argument. So if you need to use, 'ax'
#       argument, use: histplot, kdeplot, ecdfplot
###$$$$

# 2. sns.jointplot()
# http://seaborn.pydata.org/generated/seaborn.jointplot.html
# jointplot(x, y, data=None, kind='scatter',
#          stat_func=None, color=None, height=6, ratio=5, space=0.2,
#          dropna=True, xlim=None, ylim=None, joint_kws=None, marginal_kws=None,
#          annot_kws=None, **kwargs)
#         kind: { “scatter” | “kde” | “hist” | “hex” | “reg” | “resid” }
#
###$$$$
# Note: 'jointplot' DOSES not take 'ax' argument. So if you need to use, 'ax' argument
#       use: scatterplot, kdelot, histplot, regplot, residplot
###$$$$
#
# 3. sns.replplot()
# https://seaborn.pydata.org/generated/seaborn.relplot.html
# relplot(x=None, y=None, hue=None, size=None, style=None, data=None, row=None,
#        col=None, col_wrap=None, row_order=None, col_order=None, palette=None,
#        hue_order=None, hue_norm=None, sizes=None, size_order=None, size_norm=None,
#        markers=None, dashes=None, style_order=None, legend='brief', kind='scatter',
#        height=5, aspect=1, facet_kws=None, **kwargs)
#        kind: 'scatter' or 'line'
#
# 4. sns.catplot()
# https://seaborn.pydata.org/generated/seaborn.catplot.html
# catplot(*, x=None, y=None, hue=None, data=None, row=None,col=None, col_wrap=None,
#         estimator=<function mean at 0x7fecadf1cee0>, height=5, aspect=1, orient=None,
#         ci=95, n_boot=1000, units=None, seed=None, order=None, hue_order=None,
#         row_order=None, col_order=None, kind='strip',sharex=True, sharey=True,
#         color=None, palette=None, legend=True, legend_out=True,
#         margin_titles=False, facet_kws=None, **kwargs )
#         kind:  “strip”, “swarm”, “box”, “violin”, “boxen”, “point”, “bar”, or “count”
#
###$$$$
# Note: 'catplot' DOSES not take 'ax' argument. So if you need to use, 'ax' argument
#       use: stripplot, swarmplot, boxplot, violinplot,boxenplot,barplot, countplot
###$$$$
#
# 5. sns.barplot()
#    https://seaborn.pydata.org/generated/seaborn.barplot.html
# barplot(*, x=None, y=None, hue=None, data=None, order=None, hue_order=None,
#         estimator=<function mean at 0x7fecadf1cee0>, ci=95, n_boot=1000,
#         units=None, seed=None, orient=None, color=None, palette=None,
#         saturation=0.75, errcolor='.26', errwidth=None, capsize=None,
#         dodge=True, ax=None, **kwargs)
#
###$$$$
# Note: For plotting counts of a single cat feature, use 'countplot'
#       For summarising another continuous function, against cat-feature
#       use barplot with estimator of np.sum, np.mean etc
###$$$$
#


####################################
## Plotting questions that we will answer
####################################
#
## 1 Understand your numeric data
##   How is it distributed.

# Question 1: How is Age distributed?
# Question 2: How is DailyTimeSpentonSite distributed?
# Question 3: How is AreaIncome distributed?
# Question 4: Use for loop to draw the distribution plots for the following
#             columns = ['Age', 'AreaIncome', 'DailyInternetUsage', 'DailyTimeSpentonSite']

# 2.0 Relationship of numeric variable with a categorical variable

# Question 5: How is 'Age' related to clicking?
# Question 6: How is DailyInternetUsage related to clicking?
# Question 7: How is 'AreaIncome' related to clicking?
# Question 8: Draw all the following relationship plots at one go:
#               columns = ['Age', 'AreaIncome', 'DailyInternetUsage', 'DailyTimeSpentonSite']
#               catVar = ['Clicked_on_ad', 'age_cat' ]

# 3.0 Relationship of numeric to numeric variables
#     Using jointplots:

# Question 9:  Show joint distribution of DailyTimeSpentonSite and AreaIncome
# Question 10: Show joint distribution of DailyInternetUsage and DailyTimeSpentonSite
# Question 11: Show these plots as kernel density as also 'hex' as also
#              draw regression line

# 4.0 Relationship of a categorical to another categorical variable

# Question 12: What relationship exist between 'Clicked_on_ad' and 'Gender'?
# Question 13: What relationship exist between 'DailyTimeSpentonSite' and 'Gender'?
# Question 14: Relationship between Gender and Clicked_on_ad, subset by 'age_cat wise

# 5.0 Relationship between two categorical and one numeric variable

# Question 15: Hour and weekday wise when are clicks most
# Question 16: Quarter wise and weekday wise when are clicks most
# Question 17: Quarter wise and weekday wise when are DailyInternetUsage max and min

# 6.0 Structure in data
# Question 18: Does data exhibit any pattern with respect to 'Clicked_on_ad'
#              Explore how good the patterns are. Stronger patterns will lead
#              to better classifications
#


###### Start answering questions

# 5.0.1 Sample data:
#       This step is academic here. But for large datasets,
#       there is a need to sample data before plotting so
#       that they do not crowd limited X-Y space

dn = ad.sample(frac = 0.5)    # Extract 50% sample of data
dn.shape      # (500,20)


## Task 1 Understand your numeric data
##         How is it distributed.

# Question 1: How is Age distributed?
# Question 2: How is DailyTimeSpentonSite distributed
# Question 3: How is AreaIncome distributed

# 5.1 Distribution of each continuous value using distplot()
#     https://seaborn.pydata.org/generated/seaborn.distplot.html
#     (Does not have **kwargs)

# 5.1.1 Age is slight skewed to right. Naturally density of younger
#       persons is high
sns.displot(ad.Age)


# 5.1.2 Add more plot configurations
# Refer: https://matplotlib.org/api/axes_api.html#matplotlib-axes
ax= sns.displot(ad.Age)
ax.set( xlim =(10,80),                     #  sns.distplot does not have **kwargs
        xlabel= "age of persons",
        ylabel = "counts",
        title= "Histogram of Age",
        xticks = list(range(0,80,5))
        )


# 5.1.2 Distribution of DailyTimeSpentonSite
sns.displot(ad.DailyTimeSpentonSite)
sns.displot(ad.AreaIncome)
sns.displot(ad.DailyInternetUsage)


# 5.1.3 Using for loop to plot all at once
columns = ['Age', 'AreaIncome', 'DailyInternetUsage', 'DailyTimeSpentonSite']
fig,ax = plt.subplots(2,2, figsize = (10,10))
ax = ax.flatten()
for i in range(len(columns)):
    sns.histplot(ad[columns[i]],ax = ax[i])


# 6.0 Relationship of numeric variable with a categorical variable
# Question 4: How is 'Age' related to clicking?
# Question 5: How is DailyInternetUsage related to clicking?
# Question 6: How is 'AreaIncome' related to clicking?
#

# 6.1 One demo plot of relationship of 'Age' with 'Clicked_on_ad'
#     https://seaborn.pydata.org/generated/seaborn.boxplot.html#seaborn.boxplot

sns.boxplot(x = 'Clicked_on_ad',       # Discrete
            y = 'Age',                 # Continuous
            data = ad
            )

sns.boxplot(x = 'Clicked_on_ad',       # Discrete
            y = 'Age',                 # Continuous
            data = ad,
            notch = True               # **kwargs. Not all kwargs are permitted
                                       #   From https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.boxplot.html#matplotlib.axes.Axes.boxplot
            )

# 6.2 More such relationships through for-loop
columns = ['Age', 'AreaIncome', 'DailyInternetUsage', 'DailyTimeSpentonSite']
catVar = ['Clicked_on_ad', 'age_cat' ]


# 6.3 Now for loop. First create pairs of cont and cat variables
mylist = [(cont,cat)  for cont in columns  for cat in catVar]
mylist

# 6.4 Now run-through for-loop
#     For boxen plots, see:
#     https://stackoverflow.com/a/65894078/3282777

fig,ax = plt.subplots(4,2,figsize = (10,10))
ax = ax.flatten()
for j in range(len(mylist)):
    sns.boxenplot(x = mylist[j][1], y = mylist[j][0], data = ad, ax = ax[j])


# 7.0 Relationship of numeric to numeric variables
#     Using jointplots:
#           While jointplots may not show any linear relationship,
#           they can show by use of contour plots, given X, probable
#           Y through high density areas.

# Question 7: Show joint distribution of DailyTimeSpentonSite and AreaIncome
# Question 8: Show joint distribution of DailyInternetUsage and DailyTimeSpentonSite
# Question 9: Show these plots as kernel density as also 'hex' as also
#             draw regression line
#
# A jointplot = Scatterplot + Density plots

# 7.1 Open first the following
sns.jointplot(ad.DailyTimeSpentonSite, ad.AreaIncome)
# 7.2  and then this plot to understand meaning of colour intensity
#         in contour plots?
#      The graph shows that when DailyTimeSpentonSite is around 80
#      there is high probability that AreaIncome is around 60000/-

sns.jointplot(ad.DailyTimeSpentonSite, ad.AreaIncome, kind = "kde")

# 7.3  Clearly two clusters are evident here
sns.jointplot(ad.DailyInternetUsage,
              ad.DailyTimeSpentonSite,
              kind = "kde"
              )

# 7.4 Or plot hex plot
sns.jointplot(ad.DailyInternetUsage,
              ad.DailyTimeSpentonSite,
              kind = "hex"
              )


# 7.5 Add regression and kernel density fits:
sns.jointplot(ad.DailyInternetUsage,
              ad.DailyTimeSpentonSite,
              kind = "reg"
              )


# 8.0 Relationship of a categorical to another categorical variable

#     For example per category, count of other categories
#     And relationship of categorical to numeric variables
#     For example, compare per category mean(numeric) or sum(numeric)

# Question 10: What relationship exist between 'Clicked_on_ad' and 'Gender'?
# Question 11: What relationship exist between 'DailyTimeSpentonSite' and 'Gender'?
# Question 12: Relationship between Gender and Clicked_on_ad, subset by 'age_cat wise

# 8.1 Note how seaborn uses estimator function
#     Barplots are grouped summaries, category wise
#     'estimator' is a summary function
#       For errobars, see this wikpedia on bootstrap statistics
#         https://en.wikipedia.org/wiki/Bootstrapping_(statistics)
#          Bootstrap statistics:
#          Repeatedly draw equal-sized samples of data (bootstrapping)
#           as in RandomForest & using these samples calculate 95% conf
#            interval, for example, np.sum and np.mean in the following cases

sns.barplot(x = 'Gender',
            y = 'Clicked_on_ad',
            estimator = np.sum,      # As there are multiple occurrences of Gender, sum up 'Clicked_on_ad'
            ci = 95,                 # Estimate default confidence interval using bootstrapping
            data = ad,
            #capsize = 1
            )


# 8.2 Multiple ways of plotting similar information
sns.barplot(x = 'Clicked_on_ad',
            y = 'DailyTimeSpentonSite',
            estimator = np.mean,
            ci = 95,
            data =ad
            )


# 8.3 Multiple ways of plotting similar information
fig = plt.figure(figsize = (10,8))
sns.barplot(x = 'Gender',
            y = 'Clicked_on_ad',
            hue = 'age_cat',       # Age-cat wise plots
            estimator = np.mean,
            ci = 68,
            data =ad)


# 9.0 Relationship between two categorical and one numeric variable
#     Numeric variable has to be some summary measure. So, we have
#     to first calculate this summary measure
#
#     Matrix plots or heatmap
#    #########################

# Question 13: Hour and weekday wise when are clicks most
# Question 14: Quarter wise and weekday wise when are clicks most
# Question 15: Quarter wise and weekday wise when are DailyInternetUsage max and min


# 9.1 When are total clicks more
#     Heatmap of hour vs weekday
#     X and Y labels are DataFrame indexes

grouped = ad.groupby(['hour', 'weekday'])
df_wh = grouped['Clicked_on_ad'].sum().unstack()
df_wh

# 9.2 Draw quickly the heatmap. For drawing heatmap,
#     When Pandas DataFrame is provided, the index & column
#     of DataFrame will be used to label the columns and rows
#      of heatmap.
#
sns.heatmap(df_wh)

# 9.2.1 For list of ready-made cmaps (plt.cm...), see:
#       https://matplotlib.org/tutorials/colors/colormaps.html
sns.heatmap(df_wh, cmap = plt.cm.OrRd)
sns.heatmap(df_wh, cmap = plt.cm.GnBu)


# 9.3 Quarter vs weekday
grouped = ad.groupby(['weekday','quarter'])
df_wq = grouped['Clicked_on_ad'].sum().unstack()
sns.heatmap(df_wq, cmap = plt.cm.coolwarm)

# 9.4 In which quarter daily Internet usage is more
#     It is single categorical feature vs numeric summary
#     Appropriate plot is boxplot. So we add, one more
#     feature of 'weekday'
grouped = ad.groupby([ 'weekday','quarter'])
df_wqd = grouped['DailyInternetUsage'].mean().unstack()
df_wqd
sns.heatmap(df_wqd, cmap = plt.cm.Spectral)


# 10.0 Faceted plots: Show facets of relationships between
#      by numerous categorical variables
#      Facet plots
#      READ 'catplot' AS CONDITIONAL PLOTS

# 10.1
sns.catplot(x = 'Gender',
            y = 'DailyInternetUsage',
            row = 'age_cat' ,
            col = 'area_income_cat',
            kind = 'box',
            estimator = np.sum,
            data = ad)

# 10.2
sns.catplot(x = 'age_cat',
            y = 'DailyInternetUsage',
            row = 'area_income_cat',
            col = 'Clicked_on_ad',
            estimator = np.mean ,
            kind = 'box',
            data =ad)


# 10.3 Faceted scatter plots or relationship plots
sns.relplot(x = 'Age', y = 'DailyInternetUsage', row = 'area_income_cat', col = 'weekday', kind = 'scatter', data = ad)
sns.relplot(x = 'Age', y = 'DailyInternetUsage', hue = 'area_income_cat',  kind = 'scatter', data = ad, cmap = 'winter')
sns.relplot(x = 'Age', y = 'DailyInternetUsage', hue = 'area_income_cat', size = 'weekday', kind = 'scatter', data = ad)
sns.relplot(x = 'Age', y = 'DailyInternetUsage', hue = 'hour', kind = 'scatter', data = ad)
sns.relplot(x = 'Age', y = 'DailyInternetUsage', row = 'hour', kind = 'scatter', data = ad)


########################
# 11. Discover Structure in data
#     Pandas plotting functions take care
#     of NaNs but sklearn, t-sne requires
#     NaN to be eliminated. Maybe, use fillna(-1)
#     for NaN points

# Question 16: Does data display any pattern so that
#              'Clicked_on_ad' can be classified?

ad.dtypes
ad.dtypes.value_counts()

# 11.0 Select only numeric columns for the purpose
num_data = ad.select_dtypes(include = ['float64', 'int64']).copy()
num_data.head()
num_data.shape       # (1000, 11)
num_data.columns



#11.1 Columns in num_data that are either discrete (with few levels)
#     or numeric

cols=['DailyTimeSpentonSite', 'Age','AreaIncome',
      'DailyInternetUsage','Gender', 'AdTopicLineLength',
      'AdTopicNoOfWords', 'hourOfDay', 'quarter', 'weekday','Clicked_on_ad' ]


# 11.2.1 Create an instance of StandardScaler object
ss= StandardScaler()

# 11.2.2 Use fit and transform method
nc = ss.fit_transform(num_data.loc[:,cols])

# 11.2.3
nc.shape     # (1000,9)

# 11.2.4 Transform numpy array back to pandas dataframe
#        as we will be using pandas plotting functions
nc = pd.DataFrame(nc, columns = cols)
nc.head(2)

# 11.2.5 Add/overwrite few columns that are discrete
#        These columns were not to be scaled

nc['Gender'] = ad['Gender']
nc['quarter'] = ad['quarter']
nc['hourOfDay'] = ad['hourOfDay']
nc['weekday'] = ad['weekday']
nc['Clicked_on_ad'] = ad['Clicked_on_ad']


nc.shape    # (1000,11)


# 11.3 Also create a dataframe from random data
#      for comparison
rng = np.random.default_rng()
nc_rand = pd.DataFrame(rng.normal(size = (1000,11)),
                       columns = cols    # Assign column names, just like that
                       )

# 11.3.1 Add/overwrite these columns also
#
nc_rand['Clicked_on_ad'] = np.random.randint(2, size= (1000,))   # [0,1]
nc_rand['Gender']        = np.random.randint(2, size= (1000,))   # [0,1]
nc_rand['quarter']       = np.random.randint(1,4, size= (1000,)) # [1,2,3]
nc_rand['hourOfDay']     = np.random.randint(24, size= (1000,))  # [0 to 23]
nc_rand['weekday']       = np.random.randint(7, size= (1000,))   # [0 to 6]

nc_rand.shape    # (1000,11)


# 11.4   Now start plotting
#        https://pandas.pydata.org/docs/reference/api/pandas.plotting.parallel_coordinates.html


# Parallel coordinates with random data
fig1 = plt.figure()
pd.plotting.parallel_coordinates(nc_rand,
                                 'Clicked_on_ad',    # class_column
                                  colormap='winter'
                                  )
plt.xticks(rotation=90)
plt.title("Parallel chart with random data")


# 11.4.1 Parallel coordinates with 'ad' data
fig2 = plt.figure()
ax = pd.plotting.parallel_coordinates(nc,
                                 'Clicked_on_ad',
                                  colormap= plt.cm.winter
                                  )

plt.xticks(rotation=90)
plt.title("Parallel chart with ad data")



# 11.4.2 Andrews charts with random data
fig3 = plt.figure()
pd.plotting.andrews_curves(nc_rand,
                           'Clicked_on_ad',
                           colormap = 'winter')

plt.title("Andrews plots with random data")


# 11.4.3 Andrews plots with ad data
fig4 = plt.figure()
pd.plotting.andrews_curves(nc,
                           'Clicked_on_ad',
                            colormap = plt.cm.winter
                           )
plt.xticks(rotation=90)
plt.title("Andrews curve with ad data")


# 11.4.4 Radviz plot
# https://pandas.pydata.org/docs/reference/api/pandas.plotting.radviz.html

fig5 = plt.figure()
pd.plotting.radviz(nc,
                   class_column ='Clicked_on_ad',
                   colormap= plt.cm.winter,
                   alpha = 0.4
                   )



# 11.5 See the power of t-sne
#      (t-distributed Stochastic Neighbor Embedding)

from sklearn.manifold import TSNE

# 11.5.1 Project all data but 'Clicked_on_ad' on two axis
#        Also just replace nc with nc_rand and try again

X_embedded = TSNE(n_components=2).fit_transform(nc.iloc[:,:-1])
X_embedded.shape    # (1000,2), numpy array
df = pd.DataFrame(X_embedded, columns=['X','Y'])

# 11.5.2 No two plots will be the same
sns.relplot(x = "X",
            y = "Y",
            hue = nc.Clicked_on_ad,    # Colur each point as per 1 or 0
            data = df
            )



################### I am done ########################

# 12.0 Conditional Density plots

## AA. Plot density plots and boxplots to show which
##     attributes will be able to predict/classify
##     target attribute, Clicked_on_ad
## BB. Draw Boxen plots also
#      See Moodle under Machine Learning II to
#      know what is lvplot or boxen plots

sns.boxplot(x = 'Clicked_on_ad',y = 'DailyInternetUsage', data = ad)
 # For boxenplot, refer: https://chartio.com/learn/charts/box-plot-complete-guide/
sns.boxenplot(x = 'Clicked_on_ad', y = 'Age', data = ad)

# Draw conditional density plots
#  You have to draw overlapping plots as below
df = ad[ad['Clicked_on_ad'] == 0]
df1 = ad[ad['Clicked_on_ad'] == 1]
ax = sns.kdeplot(df.DailyInternetUsage, shade = True)
sns.kdeplot(df1.DailyInternetUsage, ax = ax, shade = True)
# Here is a conditional density plot
#  for a column with random values
ad['rand'] = np.random.randn(ad.shape[0])
df = ad[ad['Clicked_on_ad'] == 0]
df1 = ad[ad['Clicked_on_ad'] == 1]
ax = sns.kdeplot(df.rand, shade = True)
sns.kdeplot(df1.rand, ax = ax, shade = True)
sns.kdeplot(df1.rand, ax = ax, shade = True)
# And this is conditional boxplot for the 'rand' column
sns.boxplot(x = 'Clicked_on_ad', y = 'rand', data =ad)


####################
# A matplotlib colormap maps the numerical range between 0 and 1 to a range of colors.
# https://stackoverflow.com/a/47699278/3282777
import matplotlib.cm as cm
cm.register_cmap(name='mycmap',
                 data={'red':   [(0.,0,0),
                                 (1.,0,0)],

                       'green': [(0.,0.6,0.6),
                                 (1.,0.6,0.6)],

                       'blue':  [(0.,0.4,0.4),
                                 (1.,0.4,0.4)],

                       'alpha': [(0.,0,0),
                                 (1,1,1)]})

sns.heatmap(df_wqd, cmap = 'mycmap', vmin = 169, vmax = 193)
help(cm.register_cmap)

'alpha': [(0.,0,0),
          (0.2,0.4,0.5),
          (1,1,1)]})
