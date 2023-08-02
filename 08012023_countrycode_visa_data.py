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

##### STEP 0: Execute another python script
with open('08022023_countrycodes_enhance.py') as script:
    exec(script.read())

##### STEP 1: HOW MANY COUNTRIES/REGIONS?
countries = pd.read_csv('countries.csv')

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
# After observing the results, I decided to set the cutoff at 0.65
all_matches = all_matches[all_matches['cosine_similarity'] > 0.65]
all_matches = all_matches.rename(columns = {0:'query', 'name':'match'})
all_matches = all_matches[['query','match','code','iso','cosine_similarity','rank']]
all_matches = all_matches.reset_index(drop = True)

# Continue filtering the matches by keeping match_1 only
best_match = all_matches[all_matches['rank'] == 'match_1']
best_match = best_match[['query','code','iso']].rename(columns = {'query':'country'})

### reduced all_matches w/ suggested edition column
suggested_names_cos = (all_matches[all_matches['rank'] == 'match_1'])[['query','match','code','iso']]

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

fuzz_results = [None]*len(unmatched)
print(fuzz_results)
results_dfs = [None]*len(unmatched)
for i in range(len(unmatched)):
    fuzz_results[i] = process.extract(unmatched[i], pd.Series(ScheduleC['name']), limit = 4)
    results_dfs[i] = pd.DataFrame(fuzz_results[i])[[0,1]]

fuzz_df = pd.concat([df for df in results_dfs]).\
    rename(columns = {0:'name',1:'score'})
fuzz_df['query'] = np.repeat(unmatched,4)
results_df = fuzz_df.merge(ScheduleC, on = 'name', how = 'left')
results_df = results_df[['query','name','code','iso','score']]
results_df = results_df[results_df['score'] > 88]

## Combine dataframes for suggested edits
results_df = results_df.rename(columns = {'name':'suggested'})
suggested_names_cos = suggested_names_cos.rename(columns = {'match': 'suggested'})
df_suggested = pd.concat([results_df[['query','suggested','code','iso']], suggested_names_cos])
final_unmatched = list(set(unmatched) - set(df_suggested['query']))
print(final_unmatched)
######### OUTPUT #################
df_suggested.to_csv('suggested_schedulec.csv', index = False)
exact_matches.to_csv('country_schedulec.csv', index = False)
with pd.ExcelWriter('visa_country_codes.xlsx') as file:
    ScheduleC.to_excel(file, sheet_name = 'ScheduleC', index = False)
    exact_matches.to_excel(file, sheet_name = 'exact', index = False)
    df_suggested.to_excel(file, sheet_name = 'suggest', index = False)
    countries.to_excel(file, sheet_name = 'countries', index = False)