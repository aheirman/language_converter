import unittest

from .late import *
from .read import *
from .tests import *

class LANG_C(unittest.TestCase, CheckUnit):
    #@unittest.skip("lang_c")
    def test_c_1(self):
        uuids = [uuid.uuid4() for i in range(10)]
        ruleManagerA = getMetaIrProductions("./languages/c/c.late")
        input = "const int a = 5;"
        outputExpect = "constinta=5;"
        matched = match(ruleManagerA, tokenize(input))

        self.assertNotEqual(matched, None)
        vals = matched.fullStr()
        esr = matched.esrap(ruleManagerA, ruleManagerA)
        self.assertEqual(esr, outputExpect)