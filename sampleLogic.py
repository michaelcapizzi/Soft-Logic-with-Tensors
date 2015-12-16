__author__ = 'mcapizzi'

#a sample model built to exhibit the first-order logic model

import LogicModel as m

#initialize domains
domainOfElements = ["john", "chris", "tom", "mary"]
domainOfUnPredicates = {"is_mathematician": ["john", "chris"], "is_ugly": ["chris"]}
domainOfBiPredicates = {"hates": [("tom", "chris"), ("tom", "john"), ("chris", "chris"), ("john", "mary")], "loves": [("john", "john"), ("tom", "tom"), ("mary", "mary"), ("mary", "john")]}

#build model
model = m.LogicModel(domainOfElements, domainOfUnPredicates, domainOfBiPredicates)

model.buildAll()

#update truth values
model.updateUnaryPredicate("chris", "is_mathematician", .75)
model.updateUnaryPredicate("tom", "is_mathematician", .25)
model.updateBinaryPredicate(("tom", "john"), "hates", .75)
model.updateBinaryPredicate(("chris", "chris"), "hates", .5)
model.updateBinaryPredicate(("tom", "tom"), "loves", .75)
model.updateBinaryPredicate(("mary", "mary"), "loves", .25)
model.updateBinaryPredicate(("mary", "john"), "loves", .5)
