#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 23:26:31 2023

@author: Tiangeng Lu

1. Count unique country/region names
2. Assign ISO country codes via (a) exact matches, (b) sklearn.pairwise, and (c) thefuzz

Thoughts after the first draft:
    (1) Which characteristics of the dictionary/reference dataset can improve matching results?
    (2) How do the characteristics differ in different approaches (tokenized words vs spelling variations)?
"""
import pandas as pd
import numpy as np

##### STEP 1: HOW MANY COUNTRIES/REGIONS?

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


###### STEP 2 ######
### 2(a) Exact Matches ###
exact_matches = countries.merge(ScheduleC, left_on = 'country', right_on = 'name', how = 'inner').drop('name',axis = 1)
print(exact_matches.shape[0] / countries.shape[0])

## Then, create a short dataframe with unmatched countries
countries_short = countries[~countries['country'].isin(exact_matches['country'])].reset_index(drop = True)

##### 2(b) TFIDF #####
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words = 'english')
# fit the "dictionary/reference" data
corpus = tfidf_vectorizer.fit_transform(ScheduleC['name'])
# make the tfidf matrix/df
tfidf_matrix_df = pd.DataFrame(corpus.toarray(), columns = tfidf_vectorizer.get_feature_names_out())
print(tfidf_matrix_df.shape)

##### 2(b) COSINE-SIMILARITY #####
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

### 2(c) thefuzz
from thefuzz import process
print(unmatched)

## It's better to place alternative terms in as a separate item
[item.split('(')[-1].replace(')','') for item in unused_ScheduleC if '(' in item]

fuzz_results = [None]*len(unmatched)
for i in range(len(unmatched)):
    fuzz_results[i] = process.extract(unmatched[i], pd.Series(unused_ScheduleC), limit = 4)

print(fuzz_results)

## TBC
