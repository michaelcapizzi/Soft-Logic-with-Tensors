import LogicModel as m

##############domain 1####################

#elements in domain
domain1El = ["john", "chris", "tom"]

#unary predicate
domain1UnPreds = {"is_mathematician": ["john", "chris"]}

#binary predicates
domain1BiPreds = {"hates": [("tom", "chris"), ("tom", "john"), ("chris", "chris")]}

##############domain 2#####################

#elements in domain
# domain2El = ["mary", "john"]
domain2El = ["john", "mary", "bill"]

#binary predicates
domain2BiPreds = {"loves": [("mary", "john"), ("john", "john"), ("mary", "mary")]}

############################################

mod1 = m.LogicModel(domain1El, domain1UnPreds, domain1BiPreds)
mod1.buildAll()

#mod2 = m.LogicModel(domain2El, dictionaryOfBinaryPredicates=domain2BiPreds)

tom_mathematician = mod1.unaryOp("is_mathematician", "tom")

print("Tom is a mathematician:\n {}".format(tom_mathematician))