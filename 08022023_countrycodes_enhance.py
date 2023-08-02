#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 21:16:15 2023

@author: Tiangeng Lu

1. Scraped from https://www.census.gov/foreign-trade/schedules/c/country.txt
2. Manipulation of dictionary/reference data to enhance recordlinkage performance
3. Split rows with alternative country names and explode them to new rows
4. Then, remove the original rows whose country names contains alternative names in "()".
5. Keeping country names short has the following advantages: 
    (a) Avoids mistakenly reducing `tfidf` token weights, and 
    (b) Increases matching chances in `thefuzz`
"""

import pandas as pd
import re
codes_raw = pd.read_csv('country.txt', delimiter='\t', skiprows= [0, 1, 2, 3, 4, 245, 246, 247, 248], names = ['V'])

#### BASIC CLEANING ####
codes = codes_raw['V'].str.split('|', expand = True).rename(columns = {
    0:'code',
    1:'name',
    2:'iso'})
# applies to all columns
codes = codes.applymap(lambda x: x.strip() if isinstance(x, str) else x)
codes['name'] = codes['name'].str.upper()
del codes_raw

#### EXTRACTING USEFUL INFO ####

# DON'T ESCAPE
alt_names = [row.split('(')[1].replace(')','') for row in codes['name'] if '(' in row]
# DO ESCAPE
main_names = [re.sub(r'\([^()]*\)', '', row).strip() for row in codes['name'] if '(' in row]
# ISO and 4-digit code values
codes_value = codes['code'][codes['name'].str.contains('\(')].tolist()
iso_value = codes['iso'][codes['name'].str.contains('\(')].tolist()

#### Dataframe ####
# As an alternative, the above lists can also be zip into a dictionary. 
# I choose to use dataframe because there're two possible values: iso and four-digit numeric code
df1 = pd.DataFrame(data = {
    'code': codes_value,
    'name': main_names,
    'iso': iso_value})
df2 = pd.DataFrame(data = {
    'code': codes_value,
    'name': alt_names,
    'iso': iso_value})
# discretionary removal for not-so-useful info
df2 = df2[df2['name'] != 'IN THE INDIAN OCEAN']

#### Append certain rows based on political/historical facts ####
df3 = pd.DataFrame(data = {
    'code':['5660','5660','7950','4120','4120','2774','6820','5700','5700','9999'],
    'name':['MACAO S.A.R','MACAU','SWAZILAND','GREAT BRITAIN','NORTHERN IRELAND','SAINT MARTIN','MICRONESIA','CHINA-MAINLAND',"PRC",'WESTERN SAHARA'],
    'iso': ['MO','MO','SZ','GB','GB','SX','FM','CN','CN','UNKNOWN']})

# concatenate all dataframes
ScheduleC = pd.concat([codes,df1,df2,df3]).\
    sort_values(['code','name','iso'], ascending = True).\
        drop_duplicates().reset_index(drop = True)
ScheduleC = ScheduleC[~ScheduleC['name'].str.contains('\(')]

del alt_names,main_names,codes_value,iso_value,df1,df2
#### OUTPUT ####
ScheduleC.to_csv('schedulec.csv', index = False)