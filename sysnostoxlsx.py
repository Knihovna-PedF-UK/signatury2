# make xlsx file with mms_ids from Alma analytics XML

import almaxml
from sys import argv

script, almafile = argv

# load only record IDs
df = almaxml.load(almafile, {'C5':'id'})
# rename column to use the correct name in the xlsx file
df = df.rename(columns={'id':'MMS ID'})

# this doesn't seem to do anything, but the idea is to 
# remove duplicated IDs
df = df.drop_duplicates(ignore_index=True)

df.to_excel("mms_id.xlsx", index = False)
