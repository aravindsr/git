# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 14:02:56 2019

@author: asesagiri
"""

import csv 
from sys import argv

inputfile = "tobeimported2.csv"
#inputfile = "lala.csv"
outputfile = "tobeimported2.ris"

items = []
#labels = ["AU", "TI", "VL", "IS", "DA", "SP", "EP", "PB", "T2", "N1", "ER"]
labels = ["TI", "AB", "PY", "T2", "VL", "M1", "SP", "AN", "DO", "ID", "AU"]


#with open(inputfile, 'r') as csvfile:
with open(inputfile, 'r', errors='ignore') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        
    for row in reader:
       	# in order from csv made from Google Sheet
        print("reading a row")
        # create a list of tuples where the first value is the two-letter label,
        # second value is a field in the row from the csv
        item = zip(labels, row)
        items.append(item)

with open(outputfile, 'w') as risfile:
    for citation in items:
        print("writing a row")
        # citation type is article
        risfile.write("TY  - JOUR \n")
        for field in citation:
            line = "{0}  - {1}\n".format(field[0], field[1])
            risfile.write(line)
        # add required end-of-record row
        risfile.write("ER  - \n")