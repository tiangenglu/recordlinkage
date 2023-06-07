#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 09:57:04 2023

@author: Tiangeng Lu
"""

# STEP 1: DATA
import numpy as np
import pandas as pd
# Data 1: country/retion list from non-immigrant visa issuances monthly report from the Dept. of State
niv = pd.read_csv('nonimm_nationalities.csv')
# remove leading & trailing white space for all string columns
niv = niv.applymap(lambda x: x.strip() if isinstance(x, str) else x)
# Data 2: Schedule C country list (downloaded online from the U.S. Census Bureau)
ScheduleC_raw = pd.read_csv("country.txt", delimiter = "\t", skiprows = [0, 1, 2, 3, 4, 245, 246, 247, 248], names = ['V'])
ScheduleC = ScheduleC_raw['V'].str.split('|', expand = True).rename(columns = {0:'code', 1:'name', 2:'iso'}) 
ScheduleC = ScheduleC.applymap(lambda x: x.strip() if isinstance(x, str) else x)
# Save the cleaned schedule C country list to local disk
#ScheduleC.to_csv('scheduleC_codes.csv', index = False)
# set country names to upper case
ScheduleC['name'] = ScheduleC['name'].str.upper()
del ScheduleC_raw
# remove leading/trailing spaces for all columns. Only works for str columns

# STEP 2: EXACT MATCHES
exact_matches\
    = niv[['country']].merge(ScheduleC, left_on = 'country', right_on = 'name', how = 'inner')\
        .drop('name', axis = 1)
# 186 exact matches, now subset the non-matches from the niv country list 
niv_short = niv[['country']][~niv['country'].isin(exact_matches['country'])].reset_index(drop = True)
# to be cautious, remove duplicates
niv_short = niv_short.drop_duplicates(keep = 'first')

# STEP 3: Tf-Idf Matrix
from sklearn.feature_extraction.text import TfidfVectorizer
# for "richer" text data, it is recommended to add the `min_df = ` statement
tfidf_vectorizer = TfidfVectorizer(stop_words = 'english')
# fit the `tfidf_vectorizer` with the country names with country codes
corpus = tfidf_vectorizer.fit_transform(ScheduleC['name'])
# create the tfidf matrix dataframe
tfidf_matrix_df = pd.DataFrame(corpus.toarray(), columns = tfidf_vectorizer.get_feature_names_out())

# STEP 4: sklearn pairwise recordlinakage

from sklearn.metrics.pairwise import euclidean_distances
# keep top n best matches
n = 2

for i in range(len(niv_short['country'])):
    query = niv_short['country'].iloc[i]
    # fit the un-matched countries to the tfidf data frame
    query_vector = tfidf_vectorizer.transform([query])
    eucl_dis = pd.DataFrame(euclidean_distances(corpus, query_vector), columns = ['eucl_dist']\
                            , index = ScheduleC.index)
    eucl_dis = eucl_dis.sort_values(by = 'eucl_dist', ascending = True)   
    output = ScheduleC.loc[eucl_dis.index[0:n], :]
    output.index = ['match_1', 'match_2']
    dist = eucl_dis[0:n]
    
    if i == 0:
        all_outputs = output.copy(deep = True)
        all_dist = dist.copy(deep = True)
    else:
        all_outputs = pd.concat([all_outputs, output])
        all_dist = pd.concat([all_dist, dist])
    
# STEP 5: compile output    
all_outputs['rank'] = all_outputs.index
# Must reset index for all dataframes before merging
all_outputs = all_outputs.reset_index(drop = True)  
# Must reset index for all dataframes before merging
all_dist = all_dist.reset_index(drop = True)    
# duplicate query rows to join the queries to their results
all_matches = pd.concat([pd.DataFrame(np.repeat(niv_short[['country']].values, n, axis = 0)),\
                        all_outputs, all_dist], axis = 1) # query, potential matches, euclidean distance  
all_matches = all_matches.rename(columns = {0:'query', 'name':'match'})
all_matches = all_matches[all_matches['rank'] == 'match_1']
all_matches.to_csv('eucl_dist.csv', index = False)
## NOT DONE YET, STILL NEEDS EVALUATE THE MATCHING RESULTS ##