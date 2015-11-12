__author__ = 'mcapizzi'


import StanfordDependencies
from nltk.parse import stanford

#stanford parser
parser = stanford.StanfordParser("/home/mcapizzi/Github/Semantics/stanford-parser-full-2014-08-27/stanford-parser.jar", "/home/mcapizzi/Github/Semantics/stanford-parser-full-2014-08-27/stanford-parser-3.4.1-models.jar")

#stanford dependency converter
#https://github.com/dmcc/PyStanfordDependencies/blob/master/README.rst
sd = StanfordDependencies.get_instance(jar_filename="/home/mcapizzi/Github/Semantics/stanford-parser-full-2014-08-27/stanford-parser.jar", backend="subprocess")

#to lemmatize
#http://textminingonline.com/dive-into-nltk-part-iv-stemming-and-lemmatization

#to parse raw text
# z = parser.raw_parse(TEXT)

#pretty print
# zPretty = z[0].pprint()

#convert to dependency
#sd.convert_tree(zPretty)
    #convert_trees for batch!

#TODO
#build method for extracting tuples