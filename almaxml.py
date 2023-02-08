"""
Process XML files produced by ALMA analytics
"""

import xml.sax as sax
from sys import argv, exit
import pandas as pd

class AlmaXmlHandler(sax.ContentHandler):
    def __init__(self, mapping = {}):
        """
        Init with mapping between field names and object names
        At minimum, mapping must contain mapping to id: {'C5':'id'}
        """
        super(sax.ContentHandler, self).__init__()
        self.records = {}
        self.mapping = mapping
        self.current_field = None
        self.current_record = None
        
    def saveRecord(self):
        current = self.current_record
        if current:
            self.records[current['id']] = current 

    def newRecord(self):
        self.current_record = {}

    def setField(self, field, text):
        # sometimes, the characters method is called multiple times for the current tag.
        # the last call would overwrite the previous text, so we need to concat them
        self.current_record[field] = self.current_record.get(field, "") +  text


    def startElement(self, name, attributes):
        if name == "R":
            self.newRecord()
        else:
            newname = self.mapping.get(name)
            if newname:
                self.current_field = newname
            else:
                self.current_field = None
        
    def endElement(self, name):
        if name == "R":
            self.saveRecord()
        # don't forget to close current field
        self.current_field = None

    def characters(self,text):
        if self.current_field:
            self.setField(self.current_field, text)



def load(filename, mapping = { 'C5':'id', 'C1': 'barcode', 'C2': 'signatura_jednotky', 'C0': 'signatura', 'C3': 'type', 'C6': "author"}):
    # file = open(filename,"r")
    parser = sax.make_parser()
    handler = AlmaXmlHandler(mapping)
    parser.setContentHandler(handler)
    parser.parse(filename)
    return pd.DataFrame.from_dict(handler.records,orient='index')


# if not len(argv) == 2:
#     print("Usage: python almaxml.py amlaxml.xml")
#     exit()

# script, filename = argv


# # file = open(filename)

# data = load(filename)


# print(data)
