# make xlsx file with mms_ids from Alma analytics XML

import almaxml
from sys import argv

script, almafile = argv

# load only record IDS
df = almaxml.load(almafile, {'C5':'id'})
# rename column to use the correct name in the xlsx file
df = df.rename(columns={'id':'MMS_ID'})

df.to_excel("mms_id.xlsx", index = False)
