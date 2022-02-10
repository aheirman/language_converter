import unittest

from eq.pika.pika import *
from eq.shared.tokenize import *
from eq.shared.grammar_reader import *
class PikaCheckUnit:
    """
        Match against the first and esrap with the second.
    """
    def check(self, ruleManagerA: RuleManager, ruleManagerB: RuleManager, input: str, output: str, begin = None):
        matched = parse(ruleManagerA, tokenize(input))
        #print(f'input: "{input}", matched: "{matched}", expected output: "{output}"')
        if matched == None:
            print(f'{bcolors.FAIL}ERROR: MATCH WAS NONE!{bcolors.ENDC}')
            self.assertEqual(output, None)
        else:
            self.assertNotEqual(output, None)
            vals = matched.fullStr()
            esr = matched.esrap(ruleManagerA, ruleManagerB)
            #print(f'input: "{input}", esr: "{esr}", expected output: "{output}"')
            self.assertEqual(esr, output)

    def runSubtests(self, ruleManagerA, ruleManagerB, inputs, outputs, begin = None):
        for input, output in zip(inputs, outputs):
            with self.subTest(input=input):
                self.check(ruleManagerA, ruleManagerB, input, output, begin)


class PIKA(unittest.TestCase, PikaCheckUnit):
    
    #@unittest.skip("")
    def test_pika_1(self):

        ruleManagerA = parseIR("""{"id": "calc"}
calculation → "abc"
""")
#
#        ruleManagerB = parseIR("""{"id": "calc"}
#calculation → term
#term → "abc"
#""")
#
#        ruleManagerC = parseIR("""{"id": "calc"}
#calculation → term "," term "," term
#term → "abc"
#""")
#
#        ruleManagerD = parseIR("""{"id": "calc"}
#calculation → term "," term "," term
#term → "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
#""")
#
#        ruleManagerE = parseIR("""{"id": "calc"}
#calculation → term "," term "," term
#term → [0-9]
#""")
#
#        ruleManagerF = parseIR("""{"id": "calc"}
#calculation → term "," term "," term
#term → lol | [0-9] | [0-9A-F]
#lol  → [abc]
#""")

        inputs = ["abc"]
        #inputs = ["0,2,6", "a,2,3"]

        projectManager = ProjectManager([ruleManagerA])
        projectManager.processProductions()
        self.runSubtests(ruleManagerA, ruleManagerA, inputs, inputs)

    #@unittest.skip("")
    def test_pika_2(self):

        ruleManagerA = parseIR("""{"id": "calc"}
calculation → term
term → "abc"
""")

        inputs = ["abc"]
        #inputs = ["0,2,6", "a,2,3"]

        projectManager = ProjectManager([ruleManagerA])
        projectManager.processProductions()
        self.runSubtests(ruleManagerA, ruleManagerA, inputs, inputs)

    #@unittest.skip("")
    def test_pika_3(self):

        ruleManagerA = parseIR("""{"id": "calc"}
calculation → "abc" "def"{"pad":true}
""")

        inputs = ["abc def "]

        projectManager = ProjectManager([ruleManagerA])
        projectManager.processProductions()
        self.runSubtests(ruleManagerA, ruleManagerA, inputs, inputs)
    
    #@unittest.skip("")
    def test_pika_4(self):

        ruleManagerA = parseIR("""{"id": "calc"}
calculation → abc "ghj"{"pad":true}
abc → "abcdef"
""")

        inputs = ["abcdef ghj "]

        projectManager = ProjectManager([ruleManagerA])
        projectManager.processProductions()
        self.runSubtests(ruleManagerA, ruleManagerA, inputs, inputs)

    #@unittest.skip("")
    def test_pika_5(self):

        ruleManagerA = parseIR("""{"id": "calc"}
calculation → abc "," digit
abc → "abcdef"
digit → "0" 
""")

        inputs = ["abcdef,0"]

        projectManager = ProjectManager([ruleManagerA])
        projectManager.processProductions()
        self.runSubtests(ruleManagerA, ruleManagerA, inputs, inputs)

    @unittest.skip("First rule is not singular(?)")
    def test_pika_6(self):

        ruleManagerA = parseIR("""{"id": "calc"}
digit → "0" | "1"
""")

        inputs = ["0","1"]

        projectManager = ProjectManager([ruleManagerA])
        projectManager.processProductions()
        self.runSubtests(ruleManagerA, ruleManagerA, inputs, inputs)

    #@unittest.skip("")
    def test_pika_7(self):

        ruleManagerA = parseIR("""{"id": "calc"}
digits → digit
digit → "0" | "1"
""")

        inputs = ["0","1"]

        projectManager = ProjectManager([ruleManagerA])
        projectManager.processProductions()
        self.runSubtests(ruleManagerA, ruleManagerA, inputs, inputs)

    #@unittest.skip("")
    def test_pika_8(self):

        ruleManagerA = parseIR("""{"id": "calc"}
calculation → abc "," digit
abc → "abcdef"
digit → "0" | "1"
""")

        inputs = ["abcdef,0","abcdef,1"]

        projectManager = ProjectManager([ruleManagerA])
        projectManager.processProductions()
        self.runSubtests(ruleManagerA, ruleManagerA, inputs, inputs)


    #@unittest.skip("")
    def test_pika_9(self):

        ruleManagerA = parseIR("""{"id": "calc"}
calculation → term "+" term
term → "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
""")


        inputs = ["1+5","9+6"]

        projectManager = ProjectManager([ruleManagerA])
        projectManager.processProductions()
        self.runSubtests(ruleManagerA, ruleManagerA, inputs, inputs)

    #@unittest.skip("")
    def test_pika_10(self):


        ruleManagerA = parseIR("""{"id": "calc"}
calculation → term opperator term
opperator → "+" | "-" 
term → "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
""")

        inputs = ["1+5","9+6"]

        projectManager = ProjectManager([ruleManagerA])
        projectManager.processProductions()
        self.runSubtests(ruleManagerA, ruleManagerA, inputs, inputs)