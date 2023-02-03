# get 
from sys import argv, exit
import re
# from io import StringIO 
import pymarc.marcxml as marcxml
import pandas as pd


records = {}

class Marcpandas(marcxml.XmlHandler):
    """Get Marc name """

    def process_record(self, record):
        # save keywords and title from a marc record
        keywords = []
        title = []
        curr_id = record.get_fields('001')[0].value()
        
        for kw in record.get_fields("650"):
            aa = kw["a"]
            if aa:
                keywords.append(aa)
        
        # get specific fields from title
        title_fields = ["a", "b", "n", "p"]
        ttl = record.get_fields("245")[0]
        for f in title_fields:
            field = ttl[f]
            if field:
                title.append(field)

        title = " ".join(title).replace("/", "")
        keywords = ", ".join(keywords)

        # print(f"sysno: {curr_id} {title} {keywords}" )
        records[curr_id] = {"title": title, "keywords": keywords}
        # print(record['650'])
        # print(type(record['650']))

def load(filename):
    marcxml.parse_xml(filename, Marcpandas())
    return pd.DataFrame.from_dict(records,orient='index')
    # return records


if not len(argv) == 2:
    print("Usage: python marcpandas.py marcxmlfile.xml")
    exit()

script, filename = argv


# file = open(filename)

data = load(filename)


print(data)


