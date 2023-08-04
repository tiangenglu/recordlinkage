#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 21:41:45 2023

@author: Tiangeng Lu

Assign country codes from the American Community Survey to monthly visa statistics
The acs country code list only has 168 items. Some entries will be assigned to more generic "other" categories.
"""
import numpy as np
import pandas as pd
import sys
# 3.9.13 (main, Aug 25 2022, 18:29:29)
print(sys.version)

# country code dictionary
acscode = pd.read_csv('acs_countrycodes.txt', delimiter= '\t', header= None).rename(columns = {0: 'code', 1:'label'})

# country list seeking for country code from visa statistics
countries = pd.read_csv('countries.csv')

### tokenize the dictionary, get tfidf matrix
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
corpus = tfidf_vectorizer.fit_transform(acscode['label'])
tfidf_matrix_df = pd.DataFrame(corpus.toarray(), columns = tfidf_vectorizer.get_feature_names_out()) # 168*188

### find best matches, w/o loops
from sklearn.metrics.pairwise import cosine_similarity

# query in rows, dictionary in columns
scores = pd.DataFrame(data = cosine_similarity(tfidf_vectorizer.transform(countries['country']), corpus),\
                      index = countries['country'], columns = acscode['label'])

best_score = [scores.iloc[i].sort_values(ascending = False)[:1][0] for i in range(len(scores))]
best_match = [scores.iloc[i].sort_values(ascending = False)[:1].index[0] for i in range(len(scores))] # match in acs

df_cosine_matches = pd.DataFrame(list(zip(countries['country'], best_match, best_score)), columns = ['query','match','score'])
df_cosine_matches = df_cosine_matches[df_cosine_matches['score'] > 0].sort_values('score', ascending = False)

second_score = [scores.iloc[i].sort_values(ascending = False)[:2][1] for i in range(len(scores))]
second_match = [scores.iloc[i].sort_values(ascending = False)[:2].index[1] for i in range(len(scores))]

df_cosine_matches2 = pd.DataFrame(list(zip(countries['country'], second_match, second_score)), columns = ['query','match','score'])
df_cosine_matches2 = df_cosine_matches2[df_cosine_matches2['score'] > 0].sort_values('score', ascending = False)

df_cosine_matches = pd.concat([df_cosine_matches, df_cosine_matches2]).drop_duplicates().sort_values('score', ascending = False).reset_index(drop = True)







