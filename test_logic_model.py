import pytest
import numpy as np
import LogicModel as m

class TestLogicModel:
    @pytest.fixture
    def basic_model(self):
        # elements in domain
        domainEl = ["john", "chris", "tom"]

        # unary predicate
        domainUnPreds = {"is_mathematician": ["john", "chris"]}

        # binary predicates
        domainBiPreds = {"hates": [("tom", "chris"), ("tom", "john"), ("chris", "chris")]}

        mod = m.LogicModel(domainEl, domainUnPreds, domainBiPreds)
        mod.buildAll()
        return mod

    def test_initialization(self, basic_model):
        assert basic_model.sizeOfDomain == 3
        assert "john" in basic_model.elementLookUp
        assert "chris" in basic_model.elementLookUp
        assert "tom" in basic_model.elementLookUp
        assert basic_model.elementLookUp["john"] == 0
        assert basic_model.elementLookUp["chris"] == 1
        assert basic_model.elementLookUp["tom"] == 2

    def test_unary_predicate_build(self, basic_model):
        # Check "is_mathematician"
        # John (index 0) is true -> [1, 0]
        # Chris (index 1) is true -> [1, 0]
        # Tom (index 2) is false -> [0, 1]

        mathematician_matrix = basic_model.unaryPredicateMatrices["is_mathematician"]

        # Check John
        np.testing.assert_array_equal(mathematician_matrix[:, 0], np.array([1., 0.]))
        # Check Chris
        np.testing.assert_array_equal(mathematician_matrix[:, 1], np.array([1., 0.]))
        # Check Tom
        np.testing.assert_array_equal(mathematician_matrix[:, 2], np.array([0., 1.]))

    def test_binary_predicate_build(self, basic_model):
        # Check "hates"
        # ("tom", "chris") -> True. Subject: tom (2), Object: chris (1)
        # tensor[0][obj][subj] = 1

        hates_tensor = basic_model.binaryPredicateTensors["hates"]

        john_idx = 0
        chris_idx = 1
        tom_idx = 2

        # Check hates(tom, chris) -> True
        assert hates_tensor[0][chris_idx][tom_idx] == 1.
        assert hates_tensor[1][chris_idx][tom_idx] == 0.

        # Check hates(tom, john) -> True
        assert hates_tensor[0][john_idx][tom_idx] == 1.

        # Check hates(chris, chris) -> True
        assert hates_tensor[0][chris_idx][chris_idx] == 1.

        # Check hates(john, tom) -> False (not in list)
        assert hates_tensor[0][tom_idx][john_idx] == 0.
        assert hates_tensor[1][tom_idx][john_idx] == 1.

    def test_add_to_domain_simple(self, basic_model):
        # Add "mary"
        basic_model.addToDomain(("mary", []))

        assert basic_model.sizeOfDomain == 4
        assert "mary" in basic_model.elementLookUp
        assert basic_model.elementLookUp["mary"] == 3

        # Check domain matrix expansion
        assert basic_model.domainMatrix.shape == (4, 4)
        assert basic_model.domainMatrix[3, 3] == 1.

        # Check unary predicate expansion
        mathematician_matrix = basic_model.unaryPredicateMatrices["is_mathematician"]
        assert mathematician_matrix.shape == (2, 4)
        # Default is False (0, 1) for new element if not specified
        np.testing.assert_array_equal(mathematician_matrix[:, 3], np.array([0., 1.]))

        # Check binary predicate expansion
        hates_tensor = basic_model.binaryPredicateTensors["hates"]
        assert hates_tensor.shape == (2, 4, 4)
        # Default should be false
        assert hates_tensor[1][3][0] == 1. # False that mary hates john

    def test_add_to_domain_with_predicates(self, basic_model):
        # Add "mary", who is a mathematician and hates tom (subject=mary, object=tom)
        # Tuple format: (predicate, otherElement, role)
        # role 'subject' means new element is subject: predicate(new, other) -> hates(mary, tom)

        basic_model.addToDomain(("mary", ["is_mathematician", ("hates", "tom", "subject")]))

        mary_idx = 3
        tom_idx = 2

        # Check Unary
        mathematician_matrix = basic_model.unaryPredicateMatrices["is_mathematician"]
        np.testing.assert_array_equal(mathematician_matrix[:, mary_idx], np.array([1., 0.]))

        # Check Binary: hates(mary, tom)
        hates_tensor = basic_model.binaryPredicateTensors["hates"]
        assert hates_tensor[0][tom_idx][mary_idx] == 1. # True

    def test_remove_from_domain(self, basic_model):
        # Remove "chris" (index 1)
        basic_model.removeFromDomain("chris")

        assert basic_model.sizeOfDomain == 2
        assert "chris" not in basic_model.elementLookUp
        assert "john" in basic_model.elementLookUp
        assert "tom" in basic_model.elementLookUp

        # Indices should shift. John was 0 (stays 0), Tom was 2 (becomes 1)
        assert basic_model.elementLookUp["john"] == 0
        assert basic_model.elementLookUp["tom"] == 1

        # Domain matrix check
        assert basic_model.domainMatrix.shape == (2, 2)

        # Unary predicate check
        mathematician_matrix = basic_model.unaryPredicateMatrices["is_mathematician"]
        assert mathematician_matrix.shape == (2, 2)
        # John is still mathematician
        np.testing.assert_array_equal(mathematician_matrix[:, 0], np.array([1., 0.]))
        # Tom is still NOT mathematician
        np.testing.assert_array_equal(mathematician_matrix[:, 1], np.array([0., 1.]))

    def test_add_unary_predicate(self, basic_model):
        basic_model.addUnaryPredicate("is_tall", ["tom"])

        assert "is_tall" in basic_model.unaryPredicateMatrices
        tall_matrix = basic_model.unaryPredicateMatrices["is_tall"]

        # Tom (index 2) should be tall
        np.testing.assert_array_equal(tall_matrix[:, 2], np.array([1., 0.]))
        # John (index 0) should not be tall - should be [0, 1]
        np.testing.assert_array_equal(tall_matrix[:, 0], np.array([0., 1.]))

    def test_add_unary_predicate_probabilistic(self, basic_model):
        # Test new probabilistic functionality
        # Chris is 80% happy
        basic_model.addUnaryPredicate("is_happy", [("chris", 0.8)])

        happy_matrix = basic_model.unaryPredicateMatrices["is_happy"]

        # Chris (index 1)
        np.testing.assert_array_almost_equal(happy_matrix[:, 1], np.array([0.8, 0.2]))

        # John (index 0) - not specified, should be False
        np.testing.assert_array_equal(happy_matrix[:, 0], np.array([0., 1.]))


    def test_add_binary_predicate(self, basic_model):
        basic_model.addBinaryPredicate("likes", [("john", "chris")])

        assert "likes" in basic_model.binaryPredicateTensors
        likes_tensor = basic_model.binaryPredicateTensors["likes"]

        # john likes chris -> True
        # subject: john (0), object: chris (1)
        assert likes_tensor[0][1][0] == 1.

        # others: checks `addBinaryPredicate` implementation.
        # It iterates over cartesian product, so it sets True or False correctly.
        # Unlike addUnaryPredicate, addBinaryPredicate seems to handle the False case correctly.

        # john likes tom -> False
        assert likes_tensor[1][2][0] == 1.

    def test_update_unary_predicate(self, basic_model):
        # Tom becomes a mathematician
        basic_model.updateUnaryPredicate("tom", "is_mathematician")

        mathematician_matrix = basic_model.unaryPredicateMatrices["is_mathematician"]
        np.testing.assert_array_equal(mathematician_matrix[:, 2], np.array([1., 0.]))

    def test_update_unary_predicate_probabilistic(self, basic_model):
        # John becomes 50% mathematician
        basic_model.updateUnaryPredicate("john", "is_mathematician", prob=0.5)

        mathematician_matrix = basic_model.unaryPredicateMatrices["is_mathematician"]
        np.testing.assert_array_almost_equal(mathematician_matrix[:, 0], np.array([0.5, 0.5]))

    def test_update_binary_predicate(self, basic_model):
        # Tom starts hating john (already does) -> let's make him hate himself
        basic_model.updateBinaryPredicate(("tom", "tom"), "hates")

        hates_tensor = basic_model.binaryPredicateTensors["hates"]
        assert hates_tensor[0][2][2] == 1.

    def test_remove_unary_predicate_element(self, basic_model):
        # John is no longer a mathematician
        basic_model.removeUnaryPredicate("john", "is_mathematician")

        mathematician_matrix = basic_model.unaryPredicateMatrices["is_mathematician"]
        np.testing.assert_array_equal(mathematician_matrix[:, 0], np.array([0., 1.]))
        assert "john" not in basic_model.unaryPredicateLookUp["is_mathematician"]

    def test_remove_binary_predicate_element(self, basic_model):
        # Tom stops hating chris
        basic_model.removeBinaryPredicate(("tom", "chris"), "hates")

        hates_tensor = basic_model.binaryPredicateTensors["hates"]
        # hates(tom, chris) -> False
        assert hates_tensor[0][1][2] == 0.
        assert hates_tensor[1][1][2] == 1.
        assert ("tom", "chris") not in basic_model.binaryPredicateLookUp["hates"]

    def test_logic_operations(self, basic_model):
        # Unary Op: is_mathematician(tom) -> False
        res = basic_model.unaryOp("is_mathematician", "tom")
        np.testing.assert_array_equal(res, np.array([0., 1.]).reshape(2, 1))

        # Binary Op: hates(tom, chris) -> True
        res = basic_model.binaryOp("hates", "tom", "chris")
        np.testing.assert_array_equal(res, np.array([1., 0.]).reshape(2, 1))

    def test_connectives(self, basic_model):
        true_val = np.array([1., 0.]).reshape(2, 1)
        false_val = np.array([0., 1.]).reshape(2, 1)

        # Negation
        np.testing.assert_array_equal(basic_model.negOp(true_val), false_val)
        np.testing.assert_array_equal(basic_model.negOp(false_val), true_val)

        # And
        np.testing.assert_array_equal(basic_model.andOp(true_val, true_val), true_val)
        np.testing.assert_array_equal(basic_model.andOp(true_val, false_val), false_val)
        np.testing.assert_array_equal(basic_model.andOp(false_val, true_val), false_val)
        np.testing.assert_array_equal(basic_model.andOp(false_val, false_val), false_val)

        # Or
        np.testing.assert_array_equal(basic_model.orOp(true_val, true_val), true_val)
        np.testing.assert_array_equal(basic_model.orOp(true_val, false_val), true_val)
        np.testing.assert_array_equal(basic_model.orOp(false_val, true_val), true_val)
        np.testing.assert_array_equal(basic_model.orOp(false_val, false_val), false_val)

        # Conditional (Implication)
        # T -> T = T
        np.testing.assert_array_equal(basic_model.conditionalOp(true_val, true_val), true_val)
        # T -> F = F
        np.testing.assert_array_equal(basic_model.conditionalOp(true_val, false_val), false_val)
        # F -> T = T
        np.testing.assert_array_equal(basic_model.conditionalOp(false_val, true_val), true_val)
        # F -> F = T
        np.testing.assert_array_equal(basic_model.conditionalOp(false_val, false_val), true_val)

    def test_get_methods(self, basic_model):
        one_hot_john = basic_model.getOneHot("john")
        assert one_hot_john.shape == (3, 1)
        assert one_hot_john[0] == 1

        mat_pred = basic_model.getUnaryPredicate("is_mathematician")
        assert mat_pred.shape == (2, 3)

        hates_pred = basic_model.getBinaryPredicate("hates")
        assert hates_pred.shape == (2, 3, 3)
    def test_remove_binary_predicate_maintains_truth(self, basic_model):
        # Initial state: hates(tom, chris) is True
        res = basic_model.binaryOp("hates", "tom", "chris")
        np.testing.assert_array_equal(res, np.array([1., 0.]).reshape(2, 1))

        # Initial state: hates(tom, john) is True
        res = basic_model.binaryOp("hates", "tom", "john")
        np.testing.assert_array_equal(res, np.array([1., 0.]).reshape(2, 1))

        # Remove hates(tom, chris)
        basic_model.removeBinaryPredicate(("tom", "chris"), "hates")

        # Check hates(tom, chris) is now False
        res = basic_model.binaryOp("hates", "tom", "chris")
        np.testing.assert_array_equal(res, np.array([0., 1.]).reshape(2, 1))

        # Check consistency: hates(tom, john) should still be True
        res = basic_model.binaryOp("hates", "tom", "john")
        np.testing.assert_array_equal(res, np.array([1., 0.]).reshape(2, 1))
