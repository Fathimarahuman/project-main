from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render,HttpResponseRedirect,redirect
from django.db import connection
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import sklearn
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from datetime import date

import csv
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor

import catboost as cb
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor


# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn.preprocessing import StandardScaler
# from sklearn.feature_selection import VarianceThreshold
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import ExtraTreesRegressor

from datetime import datetime
import time
import calendar
import random

import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import Word
import shap
import IPython

data=pd.read_csv('data_dictionary.csv')
data
print(data)
# df=pd.read_csv('sample.csv')
# print(df)
d=pd.read_csv('train.csv')
print(d)
dg=pd.read_csv('test.csv')
print(dg)

import warnings
warnings.filterwarnings('ignore')
nltk.download('stopwords')
nltk.download('wordnet')
sources= ['Facebook', 'GooglePlus', 'LinkedIn']
topics = ['Economy','Microsoft', 'Obama', 'Palestine']
master_df = pd.read_csv('News_Final.csv')

df = {}
for source in sources:
  for topic in topics:
    file_name = f'{source}_{topic}.csv'
    file_path = f'{file_name}'
    df[f'{source}_{topic}'] = pd.read_csv(file_path)


df['Facebook_Microsoft'].head()

df['Facebook_Microsoft'].tail()
for d in df:
    print(f'Null Values of {d}:',sum(df[d].isna().sum()))


news_df = pd.read_csv('News_Final.csv')
master_df = news_df.copy()
master_df.head()

master_df.tail()

master_df.shape

master_df.info()

master_df.describe()

for col in sources:
    master_df[col] = master_df[col].apply(lambda x:x+1)

for idf in df:
    for col in df[idf]:
        if col == 'IDLink':
            continue
        df[idf][col] += 1

df['Facebook_Economy'].head()

print(master_df.isna().sum())
print(master_df.Source.value_counts()[:3])
master_df['Source'].fillna('Bloomberg', inplace = True)
master_df.dropna(inplace = True)
def convert_to_dt(df):
  df['PublishDate'] = pd.to_datetime(df['PublishDate'])
  df['PublishTime'] = df['PublishDate'].dt.time
  df['PublishDate'] = df['PublishDate'].dt.date

convert_to_dt(master_df)
master_df['Weekday'] = master_df.PublishDate.apply(lambda x: calendar.day_name[x.weekday()])

master_df.head()

print('No. of Distinct Sources in Original Data set:',len(list(master_df['Source'].unique())))
master_df['Source'] = master_df['Source'].apply(lambda x:x.lower())
print('No. of Distinct Sources After Conversion:',len(list(master_df['Source'].unique())))
print('Start Date:',min(master_df['PublishDate']))
print('End Date:',max(master_df['PublishDate']))
master_df = master_df[(master_df['PublishDate']>=pd.to_datetime('2015-11-01').date()) & (master_df['PublishDate']<=pd.to_datetime('2016-08-01').date())]
print('Start Date:',min(master_df['PublishDate']))
print('End Date:',max(master_df['PublishDate']))

master_df.head()

index_to_drop = master_df[(master_df['Facebook']==0) & (master_df['GooglePlus']==0) & (master_df['LinkedIn']==0)].index
print('Shape Before Dropping:',master_df.shape)
master_df.drop(index_to_drop,inplace=True)
print('Shape After Dropping:',master_df.shape)
print(master_df[master_df.duplicated()])
master_df.drop_duplicates(inplace=True)


def show_outliers(df, three_d = False):
    fig, axes = plt.subplots(2,3,figsize=(15, 8))
    fig.tight_layout(pad=4.0)
    for source in range(len(sources)):
        sns.distplot(ax=axes[0][source],x=df[sources[source]])
        axes[0][source].set_xlabel(sources[source],fontdict={'fontsize': 12})
        axes[1][source].set_ylabel('Density',fontdict={'fontsize': 12})

        sns.boxplot(ax=axes[1][source],y=df[sources[source]] )    
        axes[1][source].set_xlabel(sources[source],fontdict={'fontsize': 12})
        axes[1][source].set_ylabel('Popularity',fontdict={'fontsize': 12})

        axes[0][source].set_title('Z - Score')
        axes[1][source].set_title('Outliers')

        plt.suptitle('Popularity of Social Media Platforms',fontsize = 22,fontweight = 'bold')

    plt.show()
    
    # if three_d == True:
    #     fig = px.scatter_3d(df, x='Facebook', y='GooglePlus', z='LinkedIn', title='Dependent Variables')
    #     fig.show()


show_outliers(master_df, True)

def outliers_treatment(df, sources):
    for source in sources:
        # tenth_percentile = np.percentile(df[source], 10)
        ninetieth_percentile = np.percentile(df[source], 90)
        # df[source] = np.where(df[source] < tenth_percentile, tenth_percentile, df[source])
        df[source] = np.where(df[source] > ninetieth_percentile, ninetieth_percentile, df[source])

    return df

master_df = outliers_treatment(master_df, sources)
show_outliers(master_df, False)

scaler = StandardScaler()

for source in sources:
    master = master_df[source].apply(lambda x:x if x!=(0) else np.nan)  # We excluded level 0 because actually it's level -1, which we are not considering. 
    master_df[f'{source}_scaled'] = scaler.fit_transform(master.values.reshape(-1,1))

    master_df[f'{source}_scaled'].fillna(0,inplace=True) # Reversed our first step.
    master_df[source].fillna(0,inplace=True)
    show_outliers(master_df, True)

master_df.reset_index(inplace=True, drop=True)
master_df.head(2)
master_df['SentimentTitle_Category'] = master_df['SentimentTitle'].apply(lambda x: 'neutral' if x == 0 else 'positive' if x > 0 else 'negative')

master_df['SentimentHeadline_Category'] = master_df['SentimentHeadline'].apply(lambda x: 'neutral' if x == 0 else 'positive' if x > 0 else 'negative')
master_df.head()

def show_no_of_news_sentiment_title(df):
    print('******** No. of News items of SentimentTitle ********')
    print(df.SentimentTitle_Category.value_counts(),'\n')

    plt.figure(figsize=(10,8))
    df.SentimentTitle_Category.value_counts().plot(kind='bar')
    plt.title('News Items Distribution of Each Sentiment Title', fontdict={'size':20, 'fontweight' : 'bold'})
    plt.xlabel('Sentiment Type',fontdict={'size':15})
    plt.ylabel('No. of News Items',fontdict={'size':15})

def show_no_of_news_sentiment_headline(df):
    print('******** No. of News items of SentimentHeadline ********')
    print(df.SentimentHeadline_Category.value_counts(),'\n')

    plt.figure(figsize=(10,8))
    df.SentimentHeadline_Category.value_counts().plot(kind='bar')
    plt.title('News Items Distribution of Each Sentiment Headline', fontdict={'size':20, 'fontweight' : 'bold'})
    plt.xlabel('Sentiment Type',fontdict={'size':15})
    plt.ylabel('No. of News Items',fontdict={'size':15})

show_no_of_news_sentiment_title(master_df)
show_no_of_news_sentiment_headline(master_df)


# q1 = np.percentile(master_df['Source'].value_counts().unique(), 25, interpolation = 'midpoint')
# q2 = np.percentile(master_df['Source'].value_counts().unique(), 50, interpolation = 'midpoint')
# q3 = np.percentile(master_df['Source'].value_counts().unique(), 75, interpolation = 'midpoint')
# print('----- Quaters -----')
# print('> q1: ',q1,'\n> q2:',q2,'\n> q3:',q3)


# source_df = pd.DataFrame(master_df['Source'].value_counts())
# master_df['Source_type'] = master_df['Source'].apply(lambda x: 'A' if source_df['Source'][x]<=q1 else 'B' 
#                                                      if source_df['Source'][x]<=q2 else 'C' if source_df['Source'][x]<=q3 else 'D')
# master_df.drop(columns=['Source'], inplace = True)

def show_news_distribution_in_sources(df):
    print('******** No. of News items in Sources ********')
    print(df.Source_type.value_counts(),'\n')

    plt.figure(figsize=(14,8))
    df.Source_type.value_counts().plot(kind='bar')
    plt.title('News Items Distribution in Each Source Type', fontdict={'size':20,'fontweight' : 'bold'})
    plt.xlabel('Source Type',fontdict={'size':15})
    plt.ylabel('No. of News Items',fontdict={'size':15})

    show_news_distribution_in_sources(master_df)

def show_popularities(df):
    facebook = abs(df.groupby(['Topic'])['Facebook_scaled'].sum())
    googleplus = abs(df.groupby(['Topic'])['GooglePlus_scaled'].sum())
    linkedin = abs(df.groupby(['Topic'])['LinkedIn_scaled'].sum())

    ind = np.arange(len(topics)) 


    plt.figure(figsize=(14,8))
    plt.bar(ind-0.25,facebook,width=0.25,label='Facebook')
    plt.bar(ind,googleplus,width=0.25,label='GooglePlus')
    plt.bar(ind+0.25,linkedin,width=0.25,label='LinkedIn')

    plt.xticks(ind, topics)
    plt.legend()
    plt.title('Topic-wise Popularity On Each Social Media Platform', fontdict={'size':20, 'fontweight' : 'bold'})
    plt.xlabel('Topics', fontdict={'size':15})
    plt.ylabel('Popularity Level', fontdict={'size':15})
    plt.show()

    #show_popularities(master_df,True)

def mean_cal(data):
    count = len(data)-data.count(0) # because level 0 means, the news item hasn't landed on the platform yet.
    if count==0:
        return 0
    avg = (sum(data))/count

    return avg

rows = df['Facebook_Economy'].loc[:,'TS1':].columns

def mean_news_popularity(df_dict, social_media, rows):
    mean_df = pd.DataFrame(index = rows, columns = ['Economy','Microsoft','Obama','Palestine'])
    mean_df['Economy'] = df_dict[f'{social_media}_Economy'].apply(lambda x: mean_cal(list(x)))
    mean_df['Microsoft'] = df_dict[f'{social_media}_Microsoft'].apply(lambda x: mean_cal(list(x)))
    mean_df['Obama'] = df_dict[f'{social_media}_Obama'].apply(lambda x: mean_cal(list(x)))
    mean_df['Palestine'] = df_dict[f'{social_media}_Palestine'].apply(lambda x: mean_cal(list(x)))

    return mean_df

def show_news_popularity_with_time(df, Platform, rows):
    sns.set(rc={'figure.figsize':(25,7)})
    sns.lineplot(data = mean_news_popularity(df, Platform, rows), dashes=False)
    plt.title(f'News Popularity With Time On {Platform}',fontdict={'size':20, 'fontweight' : 'bold'})
    plt.xlabel('Time',fontdict={'size':15})
    plt.ylabel('Popularity',fontdict={'size':15})
    plt.show()

def compare_news_popularity(df, Platform, rows):
    sns.set(rc={'figure.figsize':(10,7)})
    sns.barplot(data = mean_news_popularity(df, Platform, rows))
    plt.title(f'Popularity Comparison On {Platform}',fontdict={'size':20,'fontweight' : 'bold'})
    plt.xlabel('Topics',fontdict={'size':15})
    plt.ylabel('Popularity',fontdict={'size':15})
     

def mean_suitable_platform(df_dict, topic, rows):
    mean_df = pd.DataFrame(index=rows, columns = ['LinkedIn','GooglePlus','Facebook'])
    mean_df['LinkedIn'] = df_dict[f'LinkedIn_{topic}'].apply(lambda x: mean_cal(list(x)))
    mean_df['GooglePlus'] = df_dict[f'GooglePlus_{topic}'].apply(lambda x: mean_cal(list(x)))
    mean_df['Facebook'] = df_dict[f'Facebook_{topic}'].apply(lambda x: mean_cal(list(x)))

    return mean_df

def show_best_platform_for_news(df_dict, topic, rows):
    sns.set(rc={'figure.figsize':(25,7)})
    sns.lineplot(data = mean_suitable_platform(df_dict, topic, rows), dashes=False)
    plt.title(f'Best Social Media for {topic}',fontdict={'size':20,'fontweight' : 'bold'})
    plt.xlabel('Time',fontdict={'size':15})
    plt.ylabel('Popularity',fontdict={'size':15})

def compare_platforms(df_dict, topic, rows):
    sns.set(rc={'figure.figsize':(10,7)})
    sns.barplot(data = mean_suitable_platform(df_dict, topic, rows))
    plt.title(f'Platform Comparison for {topic}',fontdict={'size':20,'fontweight' : 'bold'})
    plt.xlabel('Social Media',fontdict={'size':15})
    plt.ylabel('Popularity',fontdict={'size':15})
def mean_source_popularity(df, df_dict, platform, rows):
    source_a_df, source_b_df, source_c_df, source_d_df = df[df['Source_type'] == 'A'],df[df['Source_type'] == 'B'],df[df['Source_type'] == 'C'],df[df['Source_type'] == 'D']
    
    platform_df = df_dict[f'{platform}_Economy'].append(df_dict[f'{platform}_Microsoft'].append(df_dict[f'{platform}_Obama'].append(df_dict[f'{platform}_Palestine'])))

    pt_sa = source_a_df.join(platform_df, on = 'IDLink',how = 'inner',lsuffix='l',rsuffix='r')
    pt_sb = source_b_df.join(platform_df, on = 'IDLink',how = 'inner',lsuffix='l',rsuffix='r')
    pt_sc = source_c_df.join(platform_df, on = 'IDLink',how = 'inner',lsuffix='l',rsuffix='r')
    pt_sd = source_d_df.join(platform_df, on = 'IDLink',how = 'inner',lsuffix='l',rsuffix='r')

    pt_sa = pt_sa.loc[:,'TS1':]
    pt_sb = pt_sb.loc[:,'TS1':]
    pt_sc = pt_sc.loc[:,'TS1':]
    pt_sd = pt_sd.loc[:,'TS1':]


    mean_df = pd.DataFrame(index=rows, columns = ['Source_A','Source_B','Source_C', 'Source_D'])
    mean_df.Source_A = pt_sa.apply(lambda x: mean_cal(list(x)))
    mean_df.Source_B = pt_sb.apply(lambda x: mean_cal(list(x)))
    mean_df.Source_C = pt_sc.apply(lambda x: mean_cal(list(x)))
    mean_df.Source_D = pt_sd.apply(lambda x: mean_cal(list(x)))

    return mean_df

def show_source_popularity_with_time(df, df_dict, platform, rows):
    sns.set(rc={'figure.figsize':(25,7)})
    sns.lineplot(data = mean_source_popularity(df, df_dict, platform, rows), dashes=False)
    plt.title(f'Source-wise Popularity on {platform}',fontdict={'size':20,'fontweight' : 'bold'})
    plt.xlabel('Time',fontdict={'size':15})
    plt.ylabel('Popularity',fontdict={'size':15})

def show_source_popularity_comparison(df, df_dict, platform, rows):

    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(20,8))
    df1 = mean_source_popularity(df, df_dict, platform, rows)
    plt.suptitle(f'Source-wise Popularity Comparison on {platform}',fontsize = 20,fontweight = 'bold')
    sns.barplot(ax = axes[0],data = df1)
    sns.boxenplot(ax = axes[1],data = df1)
    axes[0].set_ylabel('Popularity', fontsize = 15)
    axes[0].set_xlabel('Sources', fontsize = 15)
    axes[1].set_ylabel('Popularity', fontsize = 15)
    axes[1].set_xlabel('Sources', fontsize = 15)
def show_groupwise_popularity_distribution(df):
    temp_df = df.copy()
    temp_df['Facebook'] = np.log10(temp_df['Facebook'][temp_df['Facebook']>0])
    temp_df['GooglePlus'] = np.log10(temp_df['GooglePlus'][temp_df['GooglePlus']>0])
    temp_df['LinkedIn'] = np.log10(temp_df['LinkedIn'][temp_df['LinkedIn']>0])

    fig, axes = plt.subplots(2, 3, sharex=True, figsize=(20,10))
    fig.suptitle('Group-wise News Popularity Distribution', fontdict={'size':20, 'fontweight' : 'bold'})
    sns.scatterplot(ax = axes[0,0],x="Source_type",y="Facebook", data=temp_df,hue='Source_type')
    sns.scatterplot(ax = axes[0,1],x="Source_type",y="GooglePlus", data=temp_df,hue='Source_type')
    sns.scatterplot(ax = axes[0,2],x="Source_type",y="LinkedIn", data=temp_df, hue='Source_type')

    sns.boxplot(ax = axes[1,0],x="Source_type",y="Facebook", data=temp_df,hue='Source_type')
    sns.boxplot(ax = axes[1,1],x="Source_type",y="GooglePlus", data=temp_df,hue='Source_type')
    sns.boxplot(ax = axes[1,2],x="Source_type",y="LinkedIn", data=temp_df, hue='Source_type')
     

def seperate_news_for_topic(df):

    obama_news = df[df['Topic'] == 'obama']
    microsoft_news = df[df['Topic'] == 'microsoft']
    economy_news = df[df['Topic'] == 'economy']
    palestine_news = df[df['Topic'] == 'palestine']

    return (obama_news, microsoft_news, economy_news, palestine_news)

topic_tuple = seperate_news_for_topic(master_df)
obama_news = topic_tuple[0]
microsoft_news = topic_tuple[1]
economy_news = topic_tuple[2]
palestine_news = topic_tuple[3]
     

def plot_news_vs_publish_date_month_wise(News_Final, sd, ed):
    startdate = pd.to_datetime(sd).date()
    enddate = pd.to_datetime(ed).date()
    news_final_new1 = News_Final[(News_Final['PublishDate']>=startdate) & (News_Final['PublishDate']<=enddate)] 
    group_news1 = news_final_new1.groupby('PublishDate' ).count()
    # mean_group_news1  = news_final_new1.groupby('PublishDate').mean()

    ax = plt.figure(figsize=(18,7))
    plt.plot(group_news1.index, group_news1['SentimentHeadline']) 
    plt.title('No. of News Items Publish Per Month',fontdict={'size':20, 'fontweight' : 'bold'})
    plt.xlabel('Published Date',fontdict={'size':15})
    plt.ylabel('Number of News',fontdict={'size':15})
def get_master_tsdf_min_after_pub(df, tsdf_name):
    # df['SOP'] = df.apply(lambda x:  sum([a for a in x[1:] if a>0]), axis = 1)

    social_med = tsdf_name.split('_')[0]
    columns_to_delete = df.columns[1:-1]
    tsdf_sum_fb = df.drop(columns=columns_to_delete)
    
    merge_master_tsdf_sum_fb = pd.merge(master_df[['IDLink','Topic','PublishDate', 'PublishTime', social_med]], tsdf_sum_fb, on ='IDLink')

    df['AppearedAfterMinutes'] = df.apply(lambda x:  sum([20 for a in x[1:] if a==0]), axis = 1)
    tsdf_map_fb = df.drop(columns=columns_to_delete)

    merge_master_tsdf_map_fb = pd.merge(master_df[['IDLink','Topic','PublishDate', 'PublishTime', social_med]], tsdf_map_fb, on ='IDLink')
    # merge_master_tsdf_map_fb = merge_master_tsdf_map_fb[merge_master_tsdf_map_fb[social_med] != -1]

    merge_master_tsdf_map_fb['PublishHour'] = merge_master_tsdf_map_fb['PublishTime'].apply(lambda x: str(x)[0:2])
    merge_master_tsdf_map_fb=merge_master_tsdf_map_fb.sort_values(by='PublishHour')
    return merge_master_tsdf_map_fb

def show_news_landed_on_platform_after_published(df_dict, Platform_Topic):
    platform_topic_list = Platform_Topic.split('_')
    merge_master_tsdf_map_fb = get_master_tsdf_min_after_pub(df[Platform_Topic], Platform_Topic)
    ax = plt.figure(figsize=(18,7))
    sns.lineplot(data=merge_master_tsdf_map_fb, x = 'PublishHour', y= 'AppearedAfterMinutes')
    plt.title(f'News First Appeared On {platform_topic_list[0]} of {platform_topic_list[1]} After Its Published',fontdict={'size':20,'fontweight' : 'bold'} )
def plot_topic_date_vs_news_count(df):
    group_news_count = df.groupby('PublishDate').count()
    ax = plt.figure(figsize=(18,7))
    plt.plot(group_news_count.index, group_news_count['SentimentHeadline'], label='All News')
    ax.legend(ncol=2, loc="upper right", frameon=True)
    plt.title('No. of News Items Published Per Day',fontdict={'size':20, 'fontweight' : 'bold'})
    plt.xlabel('Published Date',fontdict={'size':15})
    plt.ylabel('Number of News Published',fontdict={'size':15})
     

def plot_topic_date_vs_news_items_published():

    group_obama = obama_news.groupby('PublishDate').count()
    group_microsoft = microsoft_news.groupby('PublishDate').count()
    group_economy = economy_news.groupby('PublishDate').count()
    group_palestine = palestine_news.groupby('PublishDate').count()

    ax = plt.figure(figsize=(18,7))
    plt.plot(group_obama.index, group_obama['SentimentHeadline'], label='Obama')
    plt.plot(group_microsoft.index, group_microsoft['SentimentHeadline'], label='Microsoft')
    plt.plot(group_economy.index, group_economy['SentimentHeadline'] , label='Economy')
    plt.plot(group_palestine.index, group_palestine['SentimentHeadline'], label='Palestine')
    plt.title('No. of News Items Published Per Day on Various Topics',fontdict={'size':20, 'fontweight' : 'bold'})
    plt.xlabel('Published Date',fontdict={'size':15})
    plt.ylabel('Number of News Published',fontdict={'size':15})
    plt.legend(loc=2)
def plot_topic_date_vs_news_items_published():

    group_obama = obama_news.groupby('PublishDate').count()
    group_microsoft = microsoft_news.groupby('PublishDate').count()
    group_economy = economy_news.groupby('PublishDate').count()
    group_palestine = palestine_news.groupby('PublishDate').count()

    ax = plt.figure(figsize=(18,7))
    plt.plot(group_obama.index, group_obama['SentimentHeadline'], label='Obama')
    plt.plot(group_microsoft.index, group_microsoft['SentimentHeadline'], label='Microsoft')
    plt.plot(group_economy.index, group_economy['SentimentHeadline'] , label='Economy')
    plt.plot(group_palestine.index, group_palestine['SentimentHeadline'], label='Palestine')
    plt.title('No. of News Items Published Per Day on Various Topics',fontdict={'size':20, 'fontweight' : 'bold'})
    plt.xlabel('Published Date',fontdict={'size':15})
    plt.ylabel('Number of News Published',fontdict={'size':15})
    plt.legend(loc=2)
     

def plot_hour_vs_popularity(df, social_media):
    df['Hour'] = df['PublishTime'].apply(lambda x: str(x)[0:2])
    df=df.sort_values(by='Hour')
    ax = plt.figure(figsize=(18,7))
    sns.lineplot(data=df, y = social_media, x= 'Hour')
    plt.title(f'Popularity On {social_media} At Each Hour', fontdict={'fontsize': 22, 'fontweight' : 'bold'})
    plt.xlabel('Hour',fontsize = 15)
    plt.ylabel(f'{social_media}',fontsize = 15)
     

def plot_sentiment_vs_news_count(palestine_news, obama_news, microsoft_news, economy_news, sentiment_type ):
    fig, axes = plt.subplots(2, 2, sharex=True,figsize=(20, 10))

    sns.histplot(ax= axes[0][0],data = palestine_news[sentiment_type], kde=True, color='green')
    axes[0][0].set_title('Palestine News',fontdict={'fontsize': '15', 'fontweight' : 'bold'})


    sns.histplot(ax= axes[0][1],data = obama_news[sentiment_type], kde=True, color='green')
    axes[0][1].set_title('Obama News',fontdict={'fontsize': '15', 'fontweight' : 'bold'})


    sns.histplot(ax= axes[1][0],data = microsoft_news[sentiment_type], kde=True, color='green')
    axes[1][0].set_title('Microsoft News',fontdict={'fontsize': '15', 'fontweight' : 'bold'})


    sns.histplot(ax= axes[1][1],data = economy_news[sentiment_type], kde=True, color='green')
    axes[1][1].set_title('Economy News',fontdict={'fontsize': '15', 'fontweight' : 'bold'})

    fig.suptitle('News Sentiment Density', fontsize = '22',fontweight = 'bold')
     

def show_relation_between_sentiment_and_dependent_features(df, sentimentType, feature):
    df1 = df[['Facebook', 'GooglePlus','LinkedIn','Topic', 'SentimentTitle','SentimentHeadline']]
    fig = px.scatter(df1, x=sentimentType, y=feature, color="Topic", symbol="Topic", title=f'Popularity Based on {Sentiment_Type}')
    fig.show()

def show_sentiment_correlation(df):
    fig = px.scatter(df, x="SentimentTitle", y="SentimentHeadline", facet_col="Topic", title='Correlation Between Title Sentiment and Headline Sentiment',facet_row="Source_type", trendline="ols",trendline_color_override="red")
    fig.show()

def show_sentiment_heatmap(df):
    sns.heatmap(master_df[['SentimentHeadline','SentimentTitle','Facebook','GooglePlus','LinkedIn']].corr(),annot=True)
    plt.title('Sentiment HeatMap', fontdict={'size':20, 'fontweight' : 'bold'})
def senti_mean(df, df_dict, platform, rows):
    senti_p_df, senti_neu_df, senti_n_df = df[df['SentimentTitle_Category'] == 'positive'],df[df['SentimentTitle_Category'] == 'neutral'],df[df['SentimentTitle_Category'] == 'negative']
    platform_df = df_dict[f'{platform}_Economy'].append(df_dict[f'{platform}_Microsoft'].append(df_dict[f'{platform}_Obama'].append(df_dict[f'{platform}_Palestine'])))

    spos = senti_p_df.join(platform_df, on = 'IDLink',how = 'inner',lsuffix='l',rsuffix='r')
    sneu = senti_neu_df.join(platform_df, on = 'IDLink',how = 'inner',lsuffix='l',rsuffix='r')
    sneg = senti_n_df.join(platform_df, on = 'IDLink',how = 'inner',lsuffix='l',rsuffix='r')

    spos = spos.loc[:,'TS1':]
    sneu = sneu.loc[:,'TS1':]
    sneg = sneg.loc[:,'TS1':]


    senti_mean_df = pd.DataFrame(index=rows, columns = ['Positive','Neutral','Negative'])
    senti_mean_df.Positive = spos.apply(lambda x: mean_cal(list(x)))
    senti_mean_df.Neutral = sneu.apply(lambda x: mean_cal(list(x)))
    senti_mean_df.Negative = sneg.apply(lambda x: mean_cal(list(x)))

    return senti_mean_df

def show_sentiment_popularity_trend(df, df_dict, platform, rows):
    sns.set(rc={'figure.figsize':(25,7)})
    sns.lineplot(data = senti_mean(df, df_dict, platform, rows), dashes=False)
    plt.title(f'Sentiment-wise Popularity on {platform}',fontdict={'size':20, 'fontweight' : 'bold'})
    plt.xlabel('Time',fontdict={'size':15})
    plt.ylabel('Popularity',fontdict={'size':15})

def show_sentiment_popularity_comparison(df, df_dict, platform, rows):

    fig, axes = plt.subplots(1, 2, sharex=True, figsize=(20,8))
    df1 = senti_mean(df, df_dict, platform, rows)
    plt.suptitle(f'Popularity Comparison on {platform}',fontsize = 22, fontweight = 'bold')
    sns.barplot(ax = axes[0],data = df1)
    sns.boxenplot(ax = axes[1],data = df1)
    axes[0].set_ylabel('Popularity', fontsize = 15)
    axes[0].set_xlabel('Sentiment', fontsize = 15)
    axes[1].set_ylabel('Popularity', fontsize = 15)
    axes[1].set_xlabel('Sentiment', fontsize = 15)

     

# def show_popularities(df, sentiment):
#     platform_df = df[['Facebook_scaled', 'GooglePlus_scaled','LinkedIn_scaled']]
#     platform_df = abs(platform_df)
#     platform_df[f'{sentiment}_Category'] = df[f'{sentiment}_Category']
#     temp_df = pd.DataFrame(columns = ['Facebook','GooglePlus','LinkedIn'])
#     temp_df['Facebook'] = platform_df.groupby([f'{sentiment}_Category'])['Facebook_scaled'].sum()
#     temp_df['GooglePlus'] = platform_df.groupby([f'{sentiment}_Category'])['GooglePlus_scaled'].sum()
#     temp_df['LinkedIn'] = platform_df.groupby([f'{sentiment}_Category'])['LinkedIn_scaled'].sum()
    
    
#     bars=temp_df.columns

#     fig = go.Figure(data=[
#         go.Bar(name='Negative', x=bars, y=temp_df['Facebook']),
#         go.Bar(name='Neutral', x=bars, y=temp_df['GooglePlus']),
#         go.Bar(name='Positive', x=bars, y=temp_df['LinkedIn'])
#     ])
#     # Change the bar mode
#     fig.update_layout(barmode='group',title_text='Sentiment Distribution')
#     fig.show()

     

# def weekday_distribution(df):
#     df1 = df[['Facebook','LinkedIn','GooglePlus','Weekday']]
#     temp_df = pd.DataFrame([df1.groupby(['Weekday'])['Facebook'].mean(),df1.groupby(['Weekday'])['LinkedIn'].mean(),df1.groupby(['Weekday'])['GooglePlus'].mean()])

#     fig, axes = plt.subplots(1, 3, sharex=True, figsize=(25,7))
#     fig.suptitle('Weekdays Mean News Popularity Distribution', fontsize=22,fontweight='bold')
#     temp_df.T['Facebook'].plot(ax = axes[0],kind='bar', color= 'grey', title='Facebook', ylabel='Popularity')
#     temp_df.T['GooglePlus'].plot(ax = axes[1],kind='bar', color= 'lightgreen',title='GooglePlus',ylabel='Popularity')
#     temp_df.T['LinkedIn'].plot(ax = axes[2],kind='bar', color= 'red', title='LinkedIn',ylabel='Popularity')
#     axes[0].set_xlabel('Weekday', fontsize = 15)
#     axes[0].set_ylabel('Popularity', fontsize = 15)
#     axes[1].set_xlabel('Weekday', fontsize = 15)
#     axes[1].set_ylabel('Popularity', fontsize = 15)
#     axes[2].set_xlabel('Weekday', fontsize = 15)
#     axes[2].set_ylabel('Popularity', fontsize = 15)
     

# def work_with_time_and_date(df):
#     # Changing time in Seconds
#     df['PublishTime'] = df.PublishTime.apply(lambda x: (x.hour * 60 + x.minute) * 60 + x.second)
#     seconds_in_day = 24*60*60

#     # Making Time Cyclic
#     df['PublishSinTime'] = df['PublishTime'].apply(lambda x: np.sin(2*np.pi*x/seconds_in_day))
#     df['PublishCosTime'] = df['PublishTime'].apply(lambda x: np.cos(2*np.pi*x/seconds_in_day))

#     df.drop(columns='PublishTime',inplace=True)

#     # Categorizing Months
#     df['PublishMonth'] = master_df['PublishDate'].apply(lambda x: calendar.month_name[x.month])

#     return df

     

# def show_month_distribution(df):
#     fig, axes = plt.subplots(1, 3, figsize=(25,7))
#     fig.suptitle('Month-wise News Popularity Distribution', fontsize=20, fontweight = 'bold')

#     sns.barplot(ax = axes[0],x='PublishMonth',y='Facebook',data = master_df)
#     sns.barplot(ax = axes[1],x='PublishMonth',y='GooglePlus',data = master_df)
#     sns.barplot(ax = axes[2], x='PublishMonth',y='LinkedIn',data = master_df)
    
#     for i in range(3):
#         axes[i].tick_params(labelrotation=45)
#         axes[i].set_ylabel('Popularity', fontsize = 15)

#     axes[0].set_xlabel('Facebook', fontsize = 15)
#     axes[1].set_xlabel('GooglePlus', fontsize = 15)
#     axes[2].set_xlabel('LinkedIn', fontsize = 15)
     

# # Merge all the 12 Time Series Dataframe with master_df

# def change_df(df, n, col_name_prefix):
#     df['TS0'] = 0
#     new_df = df[['IDLink']]
#     for i,j in zip(range(1,(72//n)+1), range(n, 73, n)):
#         new_df[f'{col_name_prefix}_t{i}'] = df[f'TS{j}'] - df[f'TS{j-n}']

#     return new_df

# def merge_df(fb_df, gp_df, li_df, topic):
#     # Convert the dataframe into - "popularity in every 6 hours"
#     # 3*k: 3 -> TS1+TS2+TS3 = 1 hour, k -> hours
#     new_fb_df = change_df(fb_df, 3*6, 'fb')
#     new_gp_df = change_df(gp_df, 3*6, 'gp')
#     new_li_df = change_df(li_df, 3*6, 'li')

#     new_final_df = new_fb_df.merge(new_gp_df.merge(new_li_df, on = 'IDLink', how='outer'), on = 'IDLink', how='outer')
    
#     return new_final_df


# def merge_all_df(master_df, df):
    
#     # Step 1: Split master_df into Topics: Economy, Microsoft, Obama, Palestine
#     economy_df, microsoft_df, obama_df, palestine_df = master_df[master_df.Topic == 'economy'], master_df[master_df.Topic == 'microsoft'], master_df[master_df.Topic == 'obama'], master_df[master_df.Topic == 'palestine']

#     # Step 2: Combine all Topics dataframes: like: (Facebook_Economy, LinkedIn_Economy, GooglePlus_Economy), (Facebook_Obama, GooglePlus_Obama, LinkedIn_Obama), etc.
#     df_time_dict = {}
#     for topic in ['Economy','Microsoft','Obama','Palestine']:
#         df_time_dict[topic] = merge_df(df[f'Facebook_{topic}'], df[f'GooglePlus_{topic}'], df[f'LinkedIn_{topic}'], topic )

#     # Step 3: Merge splitted master_df dataframe and df_time_dict dataframes
#     economy_df = economy_df.merge(df_time_dict['Economy'], on='IDLink',how = 'left')
#     microsoft_df = microsoft_df.merge(df_time_dict['Microsoft'], on='IDLink',how = 'left')
#     obama_df = obama_df.merge(df_time_dict['Obama'], on='IDLink',how = 'left')
#     palestine_df = palestine_df.merge(df_time_dict['Palestine'], on='IDLink',how = 'left')

#     # Step 4: Append all the dataframes to make the final master dataframe
#     final_master_df = economy_df.append(microsoft_df.append(obama_df.append(palestine_df)))

#     # Removing Null values
#     print('Before: ',list(final_master_df.isna().sum()))

#     # Checking for the news who has some popularity on any social media and it's time series columns are null
#     temp = final_master_df[final_master_df.isnull().any(axis=1)]
#     null_indexes = temp[temp['Facebook'] != 1].index
#     for ind in null_indexes:
#         final_master_df['fb_t1'][ind] = 1  # Popularity of those news after 2 days is 1. So, at some point in time, the popularity difference would be one.

#     final_master_df.fillna(0, inplace=True)
#     print('After: ',list(final_master_df.isna().sum()))

#     return final_master_df


     

# def cols_to_remove(df):
#     cols = []
#     for col in df.columns:
#         try:
#             int(col)
#             cols.append(col)
#         except:
#             continue

#     return cols


# def title_headline_processing():
#     st = PorterStemmer()
#     master_df['title_headline'] = master_df['Title']+', '+master_df['Headline']
#     master_df['title_headline'] = master_df['title_headline'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
#     master_df['title_headline'] = master_df['title_headline'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

#     tfidfvectorizer = TfidfVectorizer(analyzer='word', stop_words='english', max_df=0.7, min_df = 150, max_features = 1000)
#     tfidf_wm = tfidfvectorizer.fit_transform(master_df['title_headline'])
#     tfidf_tokens = tfidfvectorizer.get_feature_names()
#     df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(), columns=tfidf_tokens)
#     df_tfidfvect['IDLink'] = master_df['IDLink'].reset_index(drop=True).copy()

#     trash_cols = cols_to_remove(df_tfidfvect)

#     # Removing all the numeric columns
#     df_tfidfvect.drop(columns=trash_cols, inplace=True)

#     return df_tfidfvect

     

# def add_doc_length():
#     countVecorizer = CountVectorizer(analyzer = 'word', stop_words = 'english', max_df=0.7, min_df = 150)
#     count_wm_title = countVecorizer.fit_transform(master_df['Title'])
    
#     count_tokens_title = countVecorizer.get_feature_names()
#     df1 = pd.DataFrame(data = count_wm_title.toarray(),columns = count_tokens_title)

#     count_wm_headline = countVecorizer.fit_transform(master_df['Headline'])
#     count_tokens_headline = countVecorizer.get_feature_names()
#     df2 = pd.DataFrame(data = count_wm_headline.toarray(),columns = count_tokens_headline)
#     master_df['Title_wl'] = df1.sum(axis = 1)
#     master_df['Headline_wl'] = df2.sum(axis = 1)

# def show_doc_length_desity_plot(df):
#     fig, axes = plt.subplots(1, 2, figsize=(25,7))
    
#     sns.distplot(df['Title_wl'], ax= axes[0], color='brown')
#     axes[0].set_title('Title',fontdict={'fontsize': '15', 'fontweight' : 'bold'})
#     axes[0].set_xlabel('Title Word Length', fontsize = 15)
#     axes[0].set_ylabel('Density', fontsize = 15)

#     sns.distplot(df['Headline_wl'],ax= axes[1], color='black')
#     axes[1].set_title('Headline',fontdict={'fontsize': '15', 'fontweight' : 'bold'})
#     axes[1].set_xlabel('Headline Word Length', fontsize = 15)
#     axes[1].set_ylabel('Density', fontsize = 15)

#     fig.suptitle('Title And Headline Document Length Distribution', fontsize = 22, fontweight = 'bold')

     

# def show_doc_length_relation_with_popularity(df):
#     fig, axes = plt.subplots(3, 2, figsize=(25,25))
    
#     sns.scatterplot(ax= axes[0][0],data = df, x = 'Title_wl', y='Facebook_scaled', color='green')
#     axes[0][0].set_title('Title on Facebook',fontdict={'fontsize': '18', 'fontweight' : 'bold'})

#     sns.scatterplot(ax= axes[0][1],data = df, x = 'Headline_wl', y='Facebook_scaled', color='green')
#     axes[0][1].set_title('Headline on Facebook',fontdict={'fontsize': '18', 'fontweight' : 'bold'})


#     sns.scatterplot(ax= axes[1][0],data = df, x = 'Title_wl', y='GooglePlus_scaled', color='red')
#     axes[1][0].set_title('Title on GooglePlus',fontdict={'fontsize': '18', 'fontweight' : 'bold'})


#     sns.scatterplot(ax= axes[1][1],data = df, x = 'Headline_wl', y='GooglePlus_scaled', color='red')
#     axes[1][1].set_title('Headline on GooglePlus',fontdict={'fontsize': '18', 'fontweight' : 'bold'})

    
#     sns.scatterplot(ax= axes[2][0],data = df, x = 'Title_wl', y='LinkedIn_scaled', color='blue')
#     axes[2][0].set_title('Title on LinkedIn',fontdict={'fontsize': '18', 'fontweight' : 'bold'})

#     sns.scatterplot(ax= axes[2][1],data = df, x = 'Headline_wl', y='LinkedIn_scaled', color='blue')
#     axes[2][1].set_title('Headline on LinkedIn',fontdict={'fontsize': '18', 'fontweight' : 'bold'})

#     fig.suptitle('Effect Of Title And Headline Length On Popularity', fontsize = 22, fontweight = 'bold')

#     for i in range(0,3):
#         for j in range(0,2):
#             if j==1:
#                 axes[i][j].set_xlabel('Headline Word Length', fontsize = 15)
#             else:
#                 axes[i][j].set_xlabel('Title Word Length', fontsize = 15)
#             axes[i][j].set_ylabel('Popularity', fontsize = 15)
    
#     Platform = "GooglePlus" #@param ["Facebook", "GooglePlus", "LinkedIn"]     
#     show_news_popularity_with_time(df, Platform, rows)
#     compare_news_popularity(df, Platform, rows)
def index(request):
    return render(request,'index.html')
def login(request):
    return render(request,'login.html')    
def user(request):
    return render(request,'user.html') 
def Rating(request):
    return render(request,'Rating.html') 
def userpage(request):
    return render(request,'userpage.html')
def profile(request):
    cursor=connection.cursor()
    uid=request.session['cid']
    sq="select * from customer where cid=%s"%(uid)
    cursor.execute(sq)
    cr=[]
    rs=cursor.fetchall()
    for rw in rs:
        x={'cid':rw[0],'name':rw[1],'email':rw[2],'contactno':rw[3]}
        cr.append(x)
    return render(request,'profile.html',{'cr':cr})
def profileaction(request):
    cursor=connection.cursor()
    cd=request.GET['cid']
    e=request.GET['email']
    c=request.GET['contact']
    sq="update customer set email='%s',contactno='%s' where cid=%s"%(e,c,cd)
    cursor.execute(sq)
    html="<script>alert('Successfully Updated');window.location='/userpage/';</script>"
    return HttpResponse(html)
def useraction(request):
    cursor=connection.cursor()
    n=request.GET['name']
    e=request.GET['email'] 
    u=request.GET['userid']
    c=request.GET['number']
    p=request.GET['password'] 
    sql="insert into customer(name,email,contactno)values('%s','%s','%s')"%(n,e,c)  
    cursor.execute(sql) 
    s="select max(cid) as uid from customer"
    cursor.execute(s)
    rs=cursor.fetchall()
    for rw in rs:
        x="insert into login(cid,username,password,user_type)values(%s,'%s','%s','user')"%(rw[0],u,p)
        cursor.execute(x)
        html="<script>alert('Registered Successfully');window.location='/index/';</script>"
    return HttpResponse(html)  

def Ratingaction (request):
    cursor=connection.cursor()
    p=request.GET['post']
    u=request.GET['userid']
    r=request.GET['Rating']
    dt=date.today()
    sql="insert into Rating(post,userid,Rating,date)values('%s','%s','%s','%s')"%(p,u,r,dt)
    cursor.execute(sql)
    ht="<script>alert('Added');window.location='/index/';</script>"
    return HttpResponse(ht)
def loginaction(request):
    cursor=connection.cursor()
    u=request.GET['uname']
    p=request.GET['password']
    sq="select * from login where username='%s' and password='%s'"%(u,p)
    cursor.execute(sq)
    rs=cursor.fetchall()
    for rw in rs:
        request.session['cid']=rw[1]
        request.session['username']=rw[2]
        request.session['password']=rw[3]
        request.session['user_type']=rw[4]
        if(request.session['user_type']=='user'):
            return render(request,'userpage.html')
        else:
            html="<script>alert('Invalid username or Password');window.location='/login/';</script>"
    return HttpResponse(html)
def accuracy(request):
    return render(request,'accuracy.html')
def uploaddataset(request):
    return render(request,'uploaddataset.html')
def uploadaction(request):
    cursor=connection.cursor()
    f=request.GET['file']
    data_points=[
    {"label":"Decision Tree","y":96.8},
    {"label":"SVM","y":97.6},
    {"label":"Logistic Regression","y":98.02},
    ]
    return render(request,'uploadaction.html',{'data_points':data_points})
