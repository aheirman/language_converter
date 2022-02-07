import unittest
import cProfile
from pstats import Stats

from .early.late import *
from .read import *
from .tests import *

ruleManagerA = getMetaIrProductions("./languages/c/c.late")

class LANG_C(unittest.TestCase, CheckUnit):
    
    
    #@unittest.skip("lang_c")
    def test_c_1(self):
        #profile = cProfile.Profile()
        
        input = """const int a = 1;
const int b = 2;
const int b = 3;
const int b = 4;
const int b = 5;
const int b = 6;
const int b = 7;
const int b = 8;
const int b = 9;
const int b = 10;"""
        outputExpect = """constinta=1;constintb=2;constintb=3;constintb=4;constintb=5;constintb=6;constintb=7;constintb=8;constintb=9;constintb=10;"""
        
        #profile.enable()
        matched = match(ruleManagerA, tokenize_c(input))

        self.assertNotEqual(matched, None)
        vals = matched.fullStr()
        esr = matched.esrap(ruleManagerA, ruleManagerA)
        self.assertEqual(esr, outputExpect)

        #p = Stats (profile)
        #p.strip_dirs()
        #p.sort_stats ('cumtime')
        #p.print_stats ()
        #assert False

    #@unittest.skip("lang_c")
    def test_c_empty_statement(self):
        #profile = cProfile.Profile()
        
        input = """
int foo () {
int a = 1;
}"""
        outputExpect = """constinta=1;constintb=2;constintb=3;constintb=4;constintb=5;constintb=6;constintb=7;constintb=8;constintb=9;constintb=10;"""
        
        #profile.enable()
        matched = match(ruleManagerA, tokenize_c(input))

        self.assertNotEqual(matched, None)
        vals = matched.fullStr()
        esr = matched.esrap(ruleManagerA, ruleManagerA)
        #self.assertEqual(esr, outputExpect)


    @unittest.skip("lang_c")
    def test_c_var2(self):
        #profile = cProfile.Profile()
        
        input = """int foo () {
            int a = 2;
            a = 3;
}"""
        outputExpect = """constinta=1;constintb=2;constintb=3;constintb=4;constintb=5;constintb=6;constintb=7;constintb=8;constintb=9;constintb=10;"""
        
        #profile.enable()
        matched = match(ruleManagerA, tokenize_c(input))

        self.assertNotEqual(matched, None)
        vals = matched.fullStr()
        esr = matched.esrap(ruleManagerA, ruleManagerA)
        #self.assertEqual(esr, outputExpect)

    #@unittest.skip("lang_c")
    def test_c_var(self):
        #profile = cProfile.Profile()
        
        input = """int foo () {
            int a = 2;
            int b = 3;
}"""
        outputExpect = """constinta=1;constintb=2;constintb=3;constintb=4;constintb=5;constintb=6;constintb=7;constintb=8;constintb=9;constintb=10;"""
        
        #profile.enable()
        matched = match(ruleManagerA, tokenize_c(input))

        self.assertNotEqual(matched, None)
        vals = matched.fullStr()
        esr = matched.esrap(ruleManagerA, ruleManagerA)
        #self.assertEqual(esr, outputExpect)





