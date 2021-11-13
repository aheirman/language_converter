import unittest

from .late import *
from .read import *
from .tests import *

class PYTHON(unittest.TestCase, CheckUnit):
    #@unittest.skip("impl |")
    def test_py_1(self):
        uuids = [uuid.uuid4() for i in range(10)]
        ruleManagerA = Productiongenerator.createAllProductions([
            ([uuids[0]], 'calculation', '[0-9]{"pad": true}')])
        input = "1"
        outputExpect = " 1 "
        matched = match(ruleManagerA, tokenize(input))

        self.assertNotEqual(matched, None)
        vals = matched.fullStr()
        esr = matched.esrap(ruleManagerA, ruleManagerA)
        self.assertEqual(esr, outputExpect)