# get 
from sys import argv, exit
import pymarc.marcxml as marcxml
import pandas as pd



class Marcpandas(marcxml.XmlHandler):
    """Get Marc name """

    def get_records(self):
        # I cannot get class constructor to get properly with macxml,
        # so we initialize records object  here
        if not self.records: 
            self.records = {}
        return self.records

    def process_record(self, record):
        # save keywords and title from a marc record
        keywords = []
        title = []
        records = self.get_records()

        # get sys number
        curr_id = record.get_fields('001')[0].value()
        
        # there can be multiple 650 fields, so we need to process them all
        # get only subfield $a from keywords
        for kw in record.get_fields("650"):
            aa = kw["a"]
            if aa:
                keywords.append(aa)

        abstract = []
        for ab in record.get_fields("520"):
            aa = ab["a"]
            if aa:
                abstract.append(aa)

        mdt = []
        for kw in record.get_fields("080"):
            aa = kw["a"]
            if aa:
                mdt.append(aa)

        publisher = [] 
        year = []

        for pub in record.get_fields("260"):
            publisher.append(pub["b"])
            year.append(pub["c"])

        
        # get these specific fields from title
        title_fields = ["a", "b", "n", "p"]
        # there can be only one 245 field
        ttl = record.get_fields("245")[0]
        for f in title_fields:
            field = ttl[f]
            if field:
                title.append(field)

        # construct strings and clean them
        title = " ".join(title).replace("/", "")
        keywords = ", ".join(keywords)
        abstract = " ".join(abstract)
        mdt = ", ".join(mdt)
        publisher = ", ".join(publisher)
        year = ", ".join(year)

        # records are saved under sys number
        records[curr_id] = {"title": title, "keywords": keywords, "abstract": abstract, "mdt": mdt, "publisher": publisher, "year": year}

def load(filename):
    """Load Marcxml file and convert it to Pandas DataFrame"""
    marc = Marcpandas()
    marcxml.parse_xml(filename, marc)
    return pd.DataFrame.from_dict(marc.get_records(),orient='index')


# if not len(argv) == 2:
#     print("Usage: python marcpandas.py marcxmlfile.xml")
#     exit()

# script, filename = argv


# # file = open(filename)

# data = load(filename)


# print(data)


