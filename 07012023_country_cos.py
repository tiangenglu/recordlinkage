#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 22:36:49 2023
Completed on Sat Jul 22 2023

@author: Tiangeng Lu

I started working on this in Atlanta but wasn't able to finish it before my return to Pennsylvania.
Compared to previous work: https://github.com/tiangenglu/recordlinkage/blob/main/05212023%20Country%20Names%20Record%20Linkage%20Part%20I%20(cosine-similarity).ipynb
This script adds multi-sheet excel output that includes several tables.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

countries = pd.read_csv('countries.csv')
ScheduleC_raw = pd.read_csv("country.txt", delimiter = "\t",\
                            skiprows = [0, 1, 2, 3, 4, 245, 246, 247, 248], names = ['V'])
# reshape data by spliting the column by |
ScheduleC = ScheduleC_raw['V'].str.split('|', expand = True).\
rename(columns = {0:'code', 1:'name', 2:'iso'})
# remove potential leading and trailing spaces in a dataframe
ScheduleC = ScheduleC.applymap(lambda x: x.strip() if isinstance(x, str) else x)
# delete the raw data since the cleaned & reshaped dataframe is ready
del ScheduleC_raw
# set to uppercase to be consistent with the other dataset
ScheduleC['name'] = ScheduleC['name'].str.upper()

### Exact Matches ###
exact_matches = countries.merge(ScheduleC, left_on = 'country', right_on = 'name', how = 'inner').drop('name',axis = 1)
print(exact_matches.shape[0] / countries.shape[0])

## Then, create a short dataframe with unmatched countries
countries_short = countries[~countries['country'].isin(exact_matches['country'])].reset_index(drop = True)

##### TFIDF #####
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words = 'english')
# fit the "dictionary/reference" data
corpus = tfidf_vectorizer.fit_transform(ScheduleC['name'])
# make the tfidf matrix/df
tfidf_matrix_df = pd.DataFrame(corpus.toarray(), columns = tfidf_vectorizer.get_feature_names_out())
print(tfidf_matrix_df.shape)

##### COSINE-SIMILARITY #####
from sklearn.metrics.pairwise import cosine_similarity
n = 2

for i in range(len(countries_short['country'])):
    # the ith entry in the unmatched countries
    query = countries_short['country'].iloc[i]
    # tokenize and vectorize each individual query row, must be wrap with []
    query_vector = tfidf_vectorizer.transform([query])
    # run cosine-similarity
    cosine_sim = pd.DataFrame(cosine_similarity(corpus, query_vector), \
                              columns = ['cosine_similarity'], \
                                  index = ScheduleC.index)
    cosine_sim = cosine_sim.sort_values(by = ['cosine_similarity'], ascending = False)
    # top n cosine similarity scores
    scores = cosine_sim[0:n]
    # keep top n best matches, and keep all columns in the ScheduleC (the data that provides matches and codes)
    # output is ScheduleC data that includes top n matches for the queries
    output = ScheduleC.loc[cosine_sim.index[0:n], :]
    # output is a data with n rows, set index
    output.index = ['match_1', 'match_2']
    
    # FINAL OUTPUTS
    ## for the very first iteration
    if i == 0:
        all_outputs = output.copy(deep = True)
        all_scores = scores.copy(deep = True)
    ## from the 2nd iteration onward, append rows to the existing data
    else:
        all_outputs = pd.concat([all_outputs, output])
        all_scores = pd.concat([all_scores, scores])
    ## the final output have 2*n rows
    # NOTE: all_outputs are the matched country names/codes with index of "match_1" and "match_2",
    #       all_scores doesn't have their indeces set up. The next step is to reset index for both datasets.

# convert the indeces to a column
all_outputs['rank'] = all_outputs.index
# MUST reset index for all dataframes before merging
all_outputs = all_outputs.reset_index(drop = True)
all_scores = all_scores.reset_index(drop = True)

# duplicate query rows to join the queries to their results
# merge: query, match, score
all_matches = pd.concat([pd.DataFrame(np.repeat(countries_short[['country']].values, n, axis = 0)),\
                         all_outputs, all_scores], axis = 1)

# From the merged final results, we can see that more work can be done to improve the output. 
# 1.Rename the query column whose current column name is 0. 
# 2.Remove rows with `cosine_similarity = 0` 
# 3.Set cut-off cosine-similarity score
# 4.Optional: reorder the columns

# More dataframe editions/enhancements
# After observing the results, I decided to set the cutoff at 0.5
all_matches = all_matches[all_matches['cosine_similarity'] > 0.5]
all_matches = all_matches.rename(columns = {0:'query', 'name':'match'})
all_matches = all_matches[['query','match','code','iso','cosine_similarity','rank']]
all_matches = all_matches.reset_index(drop = True)

# Continue filtering the matches by keeping match_1 only
best_match = all_matches[all_matches['rank'] == 'match_1']
best_match = best_match[['query','code','iso']].rename(columns = {'query':'country'})

# Concatenate exact and cosine-similarity matches
final_matches = pd.concat([exact_matches, best_match]).sort_values(by = 'country').reset_index(drop = True)

# The following failed to find matches from cosine-similarity. They need further efforts
unmatched = list(set(countries['country']) - set(final_matches['country']))
df_unmatched = countries[countries['country'].isin(unmatched)]

# The following are the unused dictionary/reference countries
unused_ScheduleC = list(set(ScheduleC['name']) - set(all_matches['match']) - set(exact_matches['country']))
df_unused_ScheduleC = ScheduleC[ScheduleC['name'].isin(unused_ScheduleC)]

####### EXCEL OUTPUT ########
with pd.ExcelWriter('CountryISO.xlsx') as file:
    final_matches.to_excel(file, sheet_name = 'all', index = False)
    all_matches.to_excel(file, sheet_name = 'cosine_sim', index = False)
    df_unmatched.to_excel(file, sheet_name = 'unmatched', index = False)
    df_unused_ScheduleC.to_excel(file, sheet_name = 'unused', index = False)