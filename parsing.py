
__author__ = 'mcapizzi'


import StanfordDependencies
from nltk.parse import stanford
from practnlptools import tools

############################################
#using Stanford parser and dependency wrapper

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
# sd.convert_tree(zPretty)
    #convert_trees for batch! ==> sd.convert_trees([list of pretty trees]





##################################################
import practnlptools

#using SENNA
annotator = practnlptools.tools.Annotator()

#get all annotations for ONE sentence
sent = annotator.getAnnotations("I am the walrus.", dep_parse=True)

#batch of sentences
    #returns list of annotations
sentBatch = annotator.getBatchAnnotations(["Dogs eat cats.", "Cats eat dogs."], dep_parse=True)

#access just dep_parse
dep = sent["dep_parse"]

#access semantic role labeling
srl = sent["srl"]



