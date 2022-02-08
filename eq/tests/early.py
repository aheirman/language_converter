import uuid
import unittest

from eq.early.late import *
from .early_read import *

class CheckUnit:
    """
        Match against the first and esrap with the second.
    """
    def checkRegex(self, ruleManagerA: RuleManager, ruleManagerB: RuleManager, input: str, outputRegex: str, begin = None):
        matched = match(ruleManagerA, tokenize(input), begin)
        #print(f'input: "{input}", matched: "{matched}", expected output regex: "{outputRegex}"')
        if matched == None:
            self.assertEqual(outputRegex, None)
        else:
            self.assertNotEqual(outputRegex, None)
            vals = matched.fullStr()
            esr = matched.esrap(ruleManagerA, ruleManagerB)
            reg = re.compile(outputRegex)
            r = reg.match(esr)
            t = False if r==None else (r.start() == 0)
            #print(f'input: {input}, esr: "{esr}", expected output regex: "{outputRegex}"')
            self.assertTrue(t)

    """
        Match against the first and esrap with the second.
    """
    def check(self, ruleManagerA: RuleManager, ruleManagerB: RuleManager, input: str, output: str, begin = None):
        matched = match(ruleManagerA, tokenize(input), begin)
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

    def runSubtestsRegex(self, ruleManagerA, ruleManagerB, inputs, outputs, begin = None):
        for input, output in zip(inputs, outputs):
            with self.subTest(input=input):
                self.checkRegex(ruleManagerA, ruleManagerB, input, output, begin)

class ESRAP(unittest.TestCase, CheckUnit):
    #@unittest.skip("impl |")
    def test_regex(self):
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

    #@unittest.skip("impl |")
    def test_add(self):
        uuids = [uuid.uuid4() for i in range(10)]
        ruleManagerA = Productiongenerator.createAllProductions([
            ([uuids[0]], 'calculation', 'term'),
            ([uuids[1]], 'term', 'number "+" term'),
            ([uuids[2]], 'term', 'number'),
            ([uuids[3]], 'number', '[0-9]')])
        input = "1+2"
        outputExpect = input
        matched = match(ruleManagerA, tokenize(input))

        self.assertNotEqual(matched, None)
        vals = matched.fullStr()
        esr = matched.esrap(ruleManagerA, ruleManagerA)
        self.assertEqual(esr, outputExpect)

    #@unittest.skip("impl |")
    def test_add2(self):
        uuids = [uuid.uuid4() for i in range(10)]
        ruleManagerA = Productiongenerator.createAllProductions([
            ([uuids[0]], 'calculation', 'term'),
            ([uuids[1]], 'term', 'number "+" term'),
            ([uuids[2]], 'term', 'number'),
            ([uuids[3]], 'number', '[0-9]')])
        input = "1+2+3"
        outputExpect = input
        matched = match(ruleManagerA, tokenize(input))

        self.assertNotEqual(matched, None)
        vals = matched.fullStr()
        esr = matched.esrap(ruleManagerA, ruleManagerA)
        self.assertEqual(esr, outputExpect)
    
    #@unittest.skip("impl |")
    def test_add_rename(self):
        uuids = [uuid.uuid4() for i in range(10)]
        ruleManagerA = Productiongenerator.createAllProductions([
            ([uuids[0]], 'calculation', 'term'),
            ([uuids[1]], 'term', 'number "+" term'),
            ([uuids[2]], 'term', 'number'),
            ([uuids[3]], 'number', '[0-9]')])
        ruleManagerB = Productiongenerator.createAllProductions([
            ([uuids[0]], 'calculation', 'term'),
            ([uuids[1]], 'term', 'number "plus"{"pad": true} term'),
            ([uuids[2]], 'term', 'number'),
            ([uuids[3]], 'number', '[0-9]')])
        input = "1+2+3"
        outputExpect = '1 plus 2 plus 3'
        matched = match(ruleManagerA, tokenize(input))

        self.assertNotEqual(matched, None)
        vals = matched.fullStr()
        esr = matched.esrap(ruleManagerA, ruleManagerB)
        self.assertEqual(esr, outputExpect)

    #@unittest.skip("impl |")
    def test_mul_distributivity(self):
        uuids = [uuid.uuid4() for i in range(10)]

        prodD = Productiongenerator.createAllProductions([
            ([uuids[0]], 'calculation', 'term'),
            ([uuids[1]], 'term', 'number "*" "(" term "+" term ")"'),
            ([uuids[2]], 'term', 'number'),
            ([uuids[3]], 'number', '[0-9]')])
        prodF = Productiongenerator.createAllProductions([
            ([uuids[0]], 'calculation', 'term'),
            ([uuids[4]], 'term', '"(" number{"id": 0} "*" "(" term{"id": 1} ")" ")" "+" "(" number{"id": 0} "*" "(" term{"id": 2} ")" ")"', uuids[1]),
            ([uuids[2]], 'term', 'number'),
            ([uuids[3]], 'number', '[0-9]')])
        
        inputs = ["1*(2+3)"]
        outputs = ['(1*(2))+(1*(3))']

        for input, output in zip(inputs, outputs): 
            with self.subTest(input=input):
                self.check(prodD, prodF, input, output)

    def test_reorder(self):
        uuids = [uuid.uuid4() for i in range(10)]
        ruleManagerA = Productiongenerator.createAllProductions([
            ([uuids[0]], 'calculation', 'term{"id": 0} "," term{"id": 1} "," term{"id": 2}'),
            ([uuids[1]], 'term', '[0-9]')])
        
        ruleManagerB = Productiongenerator.createAllProductions([
            ([uuids[2]], 'calculation', 'term{"id": 2} "," term{"id": 1} "," term{"id": 0}', uuids[0]),
            ([uuids[1]], 'term', '[0-9]')])

        inputs = ["1,2,3"]
        outputs = ['3,2,1']

        for input, output in zip(inputs, outputs): 
            with self.subTest(input=input):
                self.check(ruleManagerA, ruleManagerB, input, output)

    def test_reorder2(self):
        uuids = [uuid.uuid4() for i in range(10)]
        ruleManagerA = Productiongenerator.createAllProductions([
            ([uuids[0]], 'calculation', 'term{"id": 0} "," term{"id": 1} "," term{"id": 2}'),
            ([uuids[1]], 'term', '[0-9]')])
        
        ruleManagerB = Productiongenerator.createAllProductions([
            ([uuids[2]], 'calculation', 'term{"id": 2, "pad":true} term{"id": 1, "pad":true} term{"id": 0, "pad":true}', uuids[0]),
            ([uuids[1]], 'term', '[0-9]')])

        inputs = ["1,2,3"]
        outputs = [' 3  2  1 ']

        for input, output in zip(inputs, outputs): 
            with self.subTest(input=input):
                self.check(ruleManagerA, ruleManagerB, input, output)

    def test_reorder3(self):
        uuids = [uuid.uuid4() for i in range(10)]
        ruleManagerA = Productiongenerator.createAllProductions([
            ([uuids[0]], 'calculation', 'term{"id": 0} "," term{"id": 1} "," term{"id": 2}'),
            ([uuids[1]], 'term', '[0-9]{"id": 0}')])
        
        ruleManagerB = Productiongenerator.createAllProductions([
            ([uuids[2]], 'calculation', 'term2{"id": 2} term2{"id": 1} term2{"id": 0}', uuids[0]),
            ([uuids[3]], 'term2', ' "(" [0-9]{"id": 0} ")" ', uuids[1])])

        inputs = ["1,2,3"]
        outputs = ['(3)(2)(1)']

        for input, output in zip(inputs, outputs):
            with self.subTest(input=input):
                self.check(ruleManagerA, ruleManagerB, input, output)

    #@unittest.skip("impl |")
    def test_or(self):
        uuids = [uuid.uuid4() for i in range(10)]
        begin = [uuids[0], uuids[1]]
        ruleManagerA = Productiongenerator.createAllProductions([(begin, 'number', '"a" | "b"')])
        inputs = ["a", "b"]
        for input in inputs:
            with self.subTest(input=input):
                self.check(ruleManagerA, ruleManagerA, input, input, begin)

    def test_zeroOrMore(self):
        pass

    #@unittest.skip("impl |")
    def test_or2(self):
        uuids = [uuid.uuid4() for i in range(10)]
        begin = [uuids[0], uuids[1]]
        beginB = [uuids[0], uuids[1], uuids[2]]
        ruleManagerA = Productiongenerator.createAllProductions([(begin, 'number', '"a" | "b"')])
        ruleManagerB = Productiongenerator.createAllProductions([(beginB, 'number', '"a" | "b" | "c"')])
        inputs = ["a", "b"]
        for input in inputs:
            with self.subTest(input=input):
                self.check(ruleManagerA, ruleManagerB, input, input, begin)

    #@unittest.skip("impl |")
    def test_or3(self):
        uuids = [uuid.uuid4() for i in range(10)]
        begin = [uuids[0], uuids[1]]
        ruleManagerA = Productiongenerator.createAllProductions([(begin, 'number', '"a" | [0-9]')])
        inputs = ["a", "0", "9"]
        for input in inputs:
            with self.subTest(input=input):
                self.check(ruleManagerA, ruleManagerA, input, input, begin)

    #@unittest.skip("impl |")
    def test_or4(self):
        uuids = [uuid.uuid4() for i in range(10)]
        begin = [uuids[0], uuids[1]]
        beginB = [uuids[0], uuids[1], uuids[2]]
        ruleManagerA = Productiongenerator.createAllProductions([(begin, 'number', '"a" | [0-9]')])
        ruleManagerB = Productiongenerator.createAllProductions([(begin, 'number', '"a" | [0-9] "BOOP"')])
        inputs = ["a", "0", "9"]
        outputs = ["a", "0BOOP", "9BOOP"]
        
        for input, output in zip(inputs, outputs): 
            with self.subTest(input=input):
                self.check(ruleManagerA, ruleManagerB, input, output, begin)

    #@unittest.skip("impl |")
    def test_or4(self):
        uuids = [uuid.uuid4() for i in range(10)]
        begin = [uuids[0], uuids[1]]
        ruleManagerA = Productiongenerator.createAllProductions([
            (begin, 'number', '"a" | [0-9] txt'),
            ([uuids[4]], 'txt', ' "BOOPly" ')])
        ruleManagerB = Productiongenerator.createAllProductions([
            ([uuids[0]], 'number', '"a"'),
            ([uuids[3]], 'number', '[0-9]{"id": 0} txt{"id": 1}', uuids[1]),
            ([uuids[4]], 'txt', ' "BOOPly" ')
            ])
        inputs = ["a", "0 BOOPly", "9 BOOPly"]
        outputs = ["a", "0BOOPly", "9BOOPly"]
        
        for input, output in zip(inputs, outputs): 
            with self.subTest(input=input):
                self.check(ruleManagerA, ruleManagerB, input, output, begin)

    def test_zeroOrMore(self):
        pass

    #@unittest.skip("impl |")
    def test_alo(self):
        uuids = [uuid.uuid4() for i in range(10)]
        ruleManagerA = Productiongenerator.createAllProductions([([uuids[0]], 'number', '"a"{"alo": true, "pad": true}')])
        inputs = [" a ", " a  a ", " a  a  a "]
        self.runSubtests(ruleManagerA, ruleManagerA, inputs, inputs)

    #@unittest.skip("impl |")
    def test_alo2(self):
        uuids = [uuid.uuid4() for i in range(10)]
        ruleManagerA = Productiongenerator.createAllProductions([
            ([uuids[0]], 'numbers', 'number{"alo": true}'),
            ([uuids[1]], 'number', '"a"{"pad": true}'),
            ])
        inputs = [" a ", " a  a ", " a  a  a "]
        self.runSubtests(ruleManagerA, ruleManagerA, inputs, inputs)

    #@unittest.skip("impl |")
    def test_alo3(self):
        uuids = [uuid.uuid4() for i in range(10)]
        ruleManagerA = Productiongenerator.createAllProductions([([uuids[0]], 'number', '"a"{"alo": true, "pad": true} "b"{"alo": true, "pad": true}')])
        inputs = [" a  b ", " a  b ", " a  a  b ", " a  a  a  b ", " a  b  b ", " a  b  b  b ", " a  a  b  b "]
        self.runSubtests(ruleManagerA, ruleManagerA, inputs, inputs)

    #@unittest.skip("impl |")
    def test_alo4(self):
        uuids = [uuid.uuid4() for i in range(10)]
        ruleManagerA = Productiongenerator.createAllProductions([
            ([uuids[0]], 'numbers', 'number{"alo": true}'),
            ([uuids[1]], 'number', '"a" ","'),
            ])
        inputs = ["a,", "a,a,", "a,a,a,"]
        self.runSubtests(ruleManagerA, ruleManagerA, inputs, inputs)

    #@unittest.skip("impl |")
    def test_alo5(self):
        uuids = [uuid.uuid4() for i in range(10)]
        ruleManagerA = Productiongenerator.createAllProductions([
            ([uuids[0]], 'numbers', 'number_sep{"alo": true} number'),
            ([uuids[1]], 'number_sep', 'number ","'),
            ([uuids[2]], 'number', '"a"'),
            ])
        inputs = ["a,a", "a,a,a", "a,a,a,a"]
        self.runSubtests(ruleManagerA, ruleManagerA, inputs, inputs)


    #@unittest.skip("impl |")
    def test_optional(self):
        uuids = [uuid.uuid4() for i in range(10)]
        ruleManagerA = Productiongenerator.createAllProductions([([uuids[0]], 'number', '"a" "b"{"opt": true, "pad": true}')])
        inputs = ["a", "a b "]
        self.runSubtests(ruleManagerA, ruleManagerA, inputs, inputs)

    #@unittest.skip("impl |")
    def test_optional2(self):
        uuids = [uuid.uuid4() for i in range(10)]
        ruleManagerA = Productiongenerator.createAllProductions([([uuids[0]], 'number', '"a" "b"{"opt": true, "pad": true} "a"{"pad": true}')])
        inputs = ["a a ", "a b  a "]
        self.runSubtests(ruleManagerA, ruleManagerA, inputs, inputs)

    @unittest.skip("BROKEN")
    def test_optional3(self):
        uuids = [uuid.uuid4() for i in range(10)]
        ruleManagerA = Productiongenerator.createAllProductions([([uuids[0]], 'number', '"a" "b"{"opt": true, "pad": true} "b"{"opt": true, "pad": true} "a"{"pad": true}')])
        inputs = ["a a ", "a b  a ", "a b  b  a "]
        outputs = copy.copy(inputs)
        outputs[1] = None
        for input, output in zip(inputs, outputs):
            with self.subTest(input=input):
                self.check(ruleManagerA, ruleManagerA, input, output)
        

    #@unittest.skip("impl |")
    def test_optional4(self):
        uuids = [uuid.uuid4() for i in range(10)]
        ruleManagerA = Productiongenerator.createAllProductions([([uuids[0]], 'number', '"a"{"opt": true, "pad": true} "b"')])
        inputs = [" a b", "b"]
        self.runSubtests(ruleManagerA, ruleManagerA, inputs, inputs)

    #@unittest.skip("impl |")
    def test_optional5(self):
        uuids = [uuid.uuid4() for i in range(10)]
        ruleManagerA = Productiongenerator.createAllProductions([
            ([uuids[0]], 'number', '"a"{"opt": true, "pad": true} ab'),
            ([uuids[1]], 'ab', '"ab"'),
            ])
        inputs = [" a ab", "ab"]
        self.runSubtests(ruleManagerA, ruleManagerA, inputs, inputs)

    #@unittest.skip("impl |")
    def test_optional6(self):
        uuids = [uuid.uuid4() for i in range(10)]
        ruleManagerA = Productiongenerator.createAllProductions([
            ([uuids[0]], 'number', '"a"{"opt": true, "pad": true} ab'),
            ([uuids[1]], 'ab', '"a"{"pad": true} "b"{"pad": true}'),
            ])
        inputs = [" a  b ", " a  a  b "]
        self.runSubtests(ruleManagerA, ruleManagerA, inputs, inputs)

    #@unittest.skip("impl |")
    def test_optional_alo(self):
        uuids = [uuid.uuid4() for i in range(10)]
        ruleManagerA = Productiongenerator.createAllProductions([([uuids[0]], 'number', '"a"{"alo": true, "pad": true} "b"{"opt": true, "pad": false}')])
        inputs = [" a ", " a b", " a  a b"]
        self.runSubtests(ruleManagerA, ruleManagerA, inputs, inputs)


    #@unittest.skip("impl |")
    def test_optional_alo2(self):
        uuids = [uuid.uuid4() for i in range(10)]
        ruleManagerA = Productiongenerator.createAllProductions([
            ([uuids[0]], 'numbers', 'number_sep{"alo": true} number{"opt": true}'),
            ([uuids[1]], 'number_sep', 'number ","'),
            ([uuids[2]], 'number', '"a"'),
            ])
        inputs = ["a,a", "a,a,a", "a,a,a,a", "a,", "a,a,", "a,a,a,"]
        #inputs = ['a,a']
        self.runSubtests(ruleManagerA, ruleManagerA, inputs, inputs)


    #@unittest.skip("impl |")
    def test_any(self):
        uuids = [uuid.uuid4() for i in range(10)]
        ruleManagerA = Productiongenerator.createAllProductions([
            ([uuids[0]], 'numbers', 'number_sep{"alo": true, "opt": true}'),
            ([uuids[1]], 'number_sep', 'number ","'),
            ([uuids[2]], 'number', '"a"'),
            ])
        inputs = ["a,", "a,a,", "a,a,a,"]
        self.runSubtests(ruleManagerA, ruleManagerA, inputs, inputs)

    #@unittest.skip("impl |")
    def test_any2(self):
        uuids = [uuid.uuid4() for i in range(10)]
        ruleManagerA = Productiongenerator.createAllProductions([
            ([uuids[0]], 'numbers', '"b"{"pad": true} number_sep{"alo": true, "opt": true}'),
            ([uuids[1]], 'number_sep', 'number ","'),
            ([uuids[2]], 'number', '"a"'),
            ])
        inputs = [" b ", " b a,", " b a,a,", " b a,a,a,"]
        self.runSubtests(ruleManagerA, ruleManagerA, inputs, inputs)

    #@unittest.skip("impl |")
    def test_any3(self):
        uuids = [uuid.uuid4() for i in range(10)]
        ruleManagerA = Productiongenerator.createAllProductions([
            ([uuids[0]], 'numbers', '"b"{"pad": true} "abc"{"alo": true, "opt": true, "pad": true}'),
            ])
        inputs = [" b ", " b  abc ", " b  abc  abc ", " b  abc  abc  abc "]
        self.runSubtests(ruleManagerA, ruleManagerA, inputs, inputs)

    #@unittest.skip("impl |")
    def test_any4(self):
        uuids = [uuid.uuid4() for i in range(10)]
        ruleManagerA = Productiongenerator.createAllProductions([
            ([uuids[0]], 'numbers', 'number_sep{"alo": true, "opt": true} "b"'),
            ([uuids[1]], 'number_sep', 'number ","'),
            ([uuids[2]], 'number', '"a"'),
            ])
        inputs = ["b", "a,b", "a,a,b", "a,a,a,b"]
        self.runSubtests(ruleManagerA, ruleManagerA, inputs, inputs)

    #@unittest.skip("impl |")
    def test_bnf(self):
        uuids = [uuid.uuid4() for i in range(50)]
        begin = [uuids[0],  uuids[1]]
        
        """
        ruleManagerBNF = Productiongenerator.createAllProductions([
            (begin,                             'syntax',         'rule | rule syntax'),
            ([uuids[2]],                        'rule',           'opt-whitespace rule-name opt-whitespace "::=" opt-whitespace expression line-end'),
            ([uuids[3]],                        'opt-whitespace', '[ ]*'),
            ([uuids[4],  uuids[5], uuids[6]],   'expression',     'list | list opt-whitespace "|" opt-whitespace expression'),
            ([uuids[7],  uuids[8]],             'line-end',       'opt-whitespace "|" opt-whitespace expression'),
            ([uuids[9],  uuids[10]],            'list',           'term | term opt-whitespace list'),
            ([uuids[11], uuids[12]],            'term',           'literal | rule-name'),
            ([uuids[13], uuids[14]],            'literal',        '"\\"" text1 "\\"" | "\'" text2 "\'"'),
            ([uuids[15], uuids[16]],            'text1',          '"" | character1 text1'),
            ([uuids[17], uuids[18]],            'text2',          '\'\' | character2 text2'),
            ([uuids[19], uuids[20], uuids[21]], 'character',      'letter | digit | symbol'),
            ([uuids[22]],                       'letter',         '[a-zA-Z]'),
            ([uuids[23]],                       'digit',          '[0-9]'),
            ([uuids[24]],                       'symbol',         '[\|!]'),
            #([uuids[12]], 'symbol',            '"|" | " " | "!" | "#" | "$" | "%" | "&" | "(" | ")" | "*" | "+" | "," | "-" | "." | "/" | ":" | ";" | ">" | "=" | "<" | "?" | "@" | "[" | "\" | "]" | "^" | "_" | "`" | "{" | "}" | "~"'),
            ([uuids[25], uuids[26]],            'character1',     'character | "\'"'),
            ([uuids[27], uuids[28]],            'character2',     'character | "\\""'),
            ([uuids[29], uuids[30]],            'rule-name',      'letter | rule-name rule-char'),
            ([uuids[31], uuids[32], uuids[33]], 'rule-char',      'letter | digit | "-"'),
            #Platform specific
            ([uuids[34]], 'EOL',            '"\r\n"'),
            ])  """
            
        
        
        ruleManagerBNF = Productiongenerator.createAllProductions([
            (begin,                             'syntax',         'rule | rule syntax'),
            ([uuids[2]],                        'rule',           'rule-name "::="{"pad": true} expression line-end'),
            ([uuids[4],  uuids[5], uuids[6]],   'expression',     'list | list "|" expression'),
            ([uuids[7],  uuids[8]],             'line-end',       'EOL | expression'),
            ([uuids[9],  uuids[10]],            'list',           'term | term list'),
            ([uuids[11], uuids[12]],            'term',           'literal | rule-name'),
            ([uuids[13], uuids[14]],            'literal',        '"\\"" [a-zA-Z0-9\\\']+ "\\"" | "\'" [a-zA-Z0-9\\"]+ "\'"'),
            #([uuids[12]], 'symbol',            '"|" | " " | "!" | "#" | "$" | "%" | "&" | "(" | ")" | "*" | "+" | "," | "-" | "." | "/" | ":" | ";" | ">" | "=" | "<" | "?" | "@" | "[" | "\" | "]" | "^" | "_" | "`" | "{" | "}" | "~"'),
            ([uuids[29], uuids[30]],            'rule-name',      '[a-zA-Z0-9]+'),
            #Platform specific
            ([uuids[34]], 'EOL',            '"\n"'),
            ])

        inputs = ["hello ::= a\n", "hello ::= a\na ::= b\n"]
        for input in inputs:
            with self.subTest(input=input):
                outputExpect = input
                interupts = ['+', '-', '*', '/', '(', ')', '\n', ',']
                matched = match(ruleManagerBNF, tokenize(input, interupts), begin)

                self.assertNotEqual(matched, None)
                vals = matched.fullStr()
                esr = matched.esrap(ruleManagerBNF, ruleManagerBNF)
                self.assertEqual(esr, outputExpect)


class READ2(unittest.TestCase, CheckUnit):
    
    @unittest.skip("escapecharacters are handeled differently")
    def test_read2_production(self):
        #BROKEN BECAUSE escapecharacters are not implemented in the new parser...
        input3 = """{"id": "calc"}
production →  name separator{"pad": false} "|"{"opt": true} to_match
to_match → sub | sub "|"{"pad": true} to_match
sub → token settings{"opt": true} | "(" to_match ")" settings{"opt": true}
settings → "{" token{"alo": true, "opt": true} "}"
name → [a-zA-Z0-9]+
token → [a-zA-Z0-9!<>#$\\\\"\\\\'\\\\+\\\\-\\\\*_\\\\.!:]+
separator → "→"
"""
        input2 = """{"id": "calc"}
calculation{"compatible": "a-1-0"} → term{"id": 2} "," term{"id": 1} "," term{"id": 0}
term{"compatible": "a-2-0"} → [0-9]{"id": 0}
"""

        input1 = """{"id": "calc"}
calculation → term "," term "," term
term → [0-9] | [0-9A-F]
"""
        inputs = [input1, input2, input3]
        for input in inputs:
            ruleManagerA = parseIR_handwritten(input.splitlines())
            ruleManagerB = parseIR(input)
            print(f'parseIR_handwritten: {str(ruleManagerA)}')
            print(f'parseIR: {str(ruleManagerB)}')
            assert str(ruleManagerA) == str(ruleManagerB)

    #@unittest.skip("read2")
    def test_read2_noitcudorp(self):
        input1 = """{"id": "a"}
term{"compatible": "b-1-0"} → new_name{"id": 0, "convert_only": true} [ab]+{"id": 1, "pad": true}
new_name ⇇ [a-zA-A][a-zA-A0-9_-]
"""

        input2 = """{"id": "b"}
term → new_name [ab]+{"pad": true} new_name
new_name → [a-zA-A][a-zA-A0-9_-]*
"""
        inputs = [input1] #[input1, input2]
        for input in inputs:
            ruleManagerA = parseIR_handwritten(input.splitlines())
            ruleManagerB = parseIR(input)
            print(f'parseIR_handwritten: {str(ruleManagerA)}')
            print(f'parseIR: {str(ruleManagerB)}')
            assert str(ruleManagerA) == str(ruleManagerB)

class READ(unittest.TestCase, CheckUnit):
    #@unittest.skip("read")
    def test_read(self):
        ruleManagerA = parseIR("""
production →  name separator{"pad": false} "|"{"opt": true} to_match
to_match → sub | sub "|"{"pad": true} to_match
sub → token settings{"opt": true} | "(" to_match ")" settings{"opt": true}
settings → "{" token{"alo": true, "opt": true} "}"
name → [a-zA-Z0-9]+
token → [a-zA-Z0-9!<>#$\\\\"\\\\'\\\\+\\\\-\\\\*_\\\\.!:]+
separator → ":"
""")
        inputs = [
            "abc:\"foo\"",
            "abc:\"foo\" | glue",
            "abc:\"foo\"{'pad':true}",
            "abc:(\"foo\" | glue)",
            "abc:(\"foo\" | glue){'pad':true}",
            "abc:(\"foo\" | glue){'pad':true}",
            "abc:(\"foo\" | glue){'pad':true} | (\"foo\" | glue){'pad':true}",
            ]
        projectManager = ProjectManager([ruleManagerA])
        projectManager.processProductions()
        self.runSubtests(ruleManagerA, ruleManagerA, inputs, inputs)

    #@unittest.skip("read")
    def test_read_reorder(self):
        ruleManagerA = parseIR("""{"id": "a"}
calculation → term "," term "," term
term → [0-9]
""")
        
        ruleManagerB = parseIR("""{"id": "b"}
calculation{"compatible": "a-1-0"} → term{"id": 2} "," term{"id": 1} "," term{"id": 0}
term{"compatible": "a-2-0"} → [0-9]{"id": 0}
""")

        inputs = ["1,2,3"]
        outputs = ['3,2,1']

        projectManager = ProjectManager([ruleManagerA, ruleManagerB])
        projectManager.processProductions()

        for input, output in zip(inputs, outputs): 
            with self.subTest(input=input):
                self.check(ruleManagerA, ruleManagerB, input, output)
        print('------------')
        for output, input in zip(inputs, outputs): 
            with self.subTest(input=input):
                self.check(ruleManagerB, ruleManagerA, input, output)

    #@unittest.skip("read")
    def test_read_reorder2(self):
        ruleManagerA = parseIR("""{"id": "a"}
calculation → term "," term "," term
term → [0-9]
""")
        
        ruleManagerB = parseIR("""{"id": "b"}
calculation{"compatible": "a-1-0"} → term{"id": 2} "," term{"id": 1} "," "strange"{"pad": true} "place"{"pad": true} term{"id": 0}
term{"compatible": "a-2-0"} → [0-9]{"id": 0}
""")

        inputs = ["1,2,3"]
        outputs = ['3,2, strange  place 1']

        projectManager = ProjectManager([ruleManagerA, ruleManagerB])
        projectManager.processProductions()

        for input, output in zip(inputs, outputs): 
            with self.subTest(input=input):
                self.check(ruleManagerA, ruleManagerB, input, output)
        print('------------')
        for output, input in zip(inputs, outputs): 
            with self.subTest(input=input):
                self.check(ruleManagerB, ruleManagerA, input, output)

    def test_read_id_requirements(self):
        ruleManagerA = parseIR("""{"id": "a"}
term → [ab]{"id": 0} [abc]
""")
        
        projectManager = ProjectManager([ruleManagerA, ruleManagerA])
        with self.assertRaises(RuntimeError):
            projectManager.processProductions()

    #@unittest.skip("noitcudorp")
    def test_read_noitcudorp(self):
        ruleManagerA = parseIR("""{"id": "a"}
term{"compatible": "b-1-0"} → new_name{"id": 0, "convert_only": true} [ab]+{"id": 1, "pad": true}
new_name ⇇ [a-zA-A][a-zA-A0-9_-]*
""")

        ruleManagerB = parseIR("""{"id": "b"}
term → new_name [ab]+{"pad": true}
new_name → [a-zA-A][a-zA-A0-9_-]*
""")

        inputs = ["ababba"]
        outputs = ["[a-zA-A][a-zA-A0-9_-]* ababba "]
        
        projectManager = ProjectManager([ruleManagerA, ruleManagerB])
        projectManager.processProductions()
        self.runSubtestsRegex(ruleManagerA, ruleManagerB, inputs, outputs)

    #@unittest.skip("noitcudorp")
    def test_read_noitcudorp2(self):
        ruleManagerA = parseIR("""{"id": "a"}
term{"compatible": "b-1-0"} → new_name{"id": 0, "convert_only": true} [ab]+{"id": 1, "pad": true} new_name{"id": 2, "convert_only": true}
new_name ⇇ [a-zA-A][a-zA-A0-9_-]*
""")

        ruleManagerB = parseIR("""{"id": "b"}
term → new_name [ab]+{"pad": true} new_name
new_name → [a-zA-A][a-zA-A0-9_-]*
""")

        inputs = ["ababba"]
        outputs = ["[a-zA-A][a-zA-A0-9_-]* ababba [a-zA-A][a-zA-A0-9_-]*"]
        
        projectManager = ProjectManager([ruleManagerA, ruleManagerB])
        projectManager.processProductions()
        
        self.runSubtestsRegex(ruleManagerA, ruleManagerB, inputs, outputs)

    #@unittest.skip("noitcudorp")
    def test_read_noitcudorp3_1(self):
        ruleManagerA = parseIR("""{"id": "a"}
term → new_name{"id": 0, "convert_only": true} [ab]+{"id": 1} new_name{"id": 2, "convert_only": true}
new_name ⇇ [a-zA-A][a-zA-A0-9_-]*
""")

        inputs = ["ababba"]
        
        projectManager = ProjectManager([ruleManagerA, ruleManagerA])
        projectManager.processProductions()
        self.runSubtestsRegex(ruleManagerA, ruleManagerA, inputs, inputs)

    #@unittest.skip("noitcudorp")
    def test_read_noitcudorp3_2(self):
        ruleManagerA = parseIR("""{"id": "a"}
term{"compatible": "b-1-0"} → new_name{"id": 0, "convert_only": true} [ab]+{"id": 1} new_name{"id": 2, "convert_only": true}
new_name ⇇ [a-zA-A][a-zA-A0-9_-]*
""")
        ruleManagerB = parseIR("""{"id": "b"}
term → new_name [ab]+{"pad": true} new_name
new_name → [a-zA-A][a-zA-A0-9_-]*
""")
        inputs = ["ababba"]
        outputs = ["[a-zA-A][a-zA-A0-9_-]* ababba [a-zA-A][a-zA-A0-9_-]*"]
        
        projectManager = ProjectManager([ruleManagerA, ruleManagerB])
        projectManager.processProductions()
        self.runSubtestsRegex(ruleManagerA, ruleManagerB, inputs, outputs)

    @unittest.skip("reader2: merge")
    def test_modify_upper(self):
        ruleManagerA = parseIR("""{"id": "a", "imports": ["b"] }
number → "(" [0-9] "," [0-9] ")"
number:"a-1-0" ⇈ numbers:"b-1-0" {
    append_into 0, 0;
    append_into 1, 0;
}
""")

        ruleManagerB = parseIR("""{"id": "b"}
numbers → number{"alo": true}
number  → [0-9]{"pad": true}
""")

        #inputs = ["0", "1 2", "2 3 (4,5)"]
        #outputs = [" 0 ", " 1  2 ", " 2  3  4  5 "]
        inputs = ["2 (4,5)"]
        outputs = [" 2  4  5 "]


        projectManager = ProjectManager([ruleManagerA, ruleManagerB])
        projectManager.processProductions()
        
        self.runSubtests(ruleManagerA, ruleManagerB, inputs, outputs, 'b-1-0')


'''
class INTERPRET(unittest.TestCase, CheckUnit):
    def get_parse(self, ruleManagerA: RuleManager, ruleManagerB: RuleManager, input: str, output: str, begin = None):
        matched = match(ruleManagerA, tokenize(input), begin)
        print(f'input: {input}, matched: {matched}, expected output: {output}')
        if matched == None:
            self.assertEqual(output, None)
        else:
            self.assertNotEqual(output, None)
            vals = matched.fullStr()
            esr = matched.esrap(ruleManagerA, ruleManagerB)
            print(f'input: {input}, esr: {esr}, expected output: {output}')

    def test_interpret(self):
        ruleManagerA = parseIR("""{"id": "a"}
calculation → term "," term "," term
term → [0-9]
""")
    
    projectManager = ProjectManager([ruleManagerA, ruleManagerB])
    projectManager.processProductions()

    input = ['a:b']
    for input, output in zip(inputs, outputs):
            with self.subTest(input=input):
'''


class PROJECT(unittest.TestCase, CheckUnit):

    #@unittest.skip("import")
    def test_import(self):
        ruleManagerA = parseIR("""{"id": "a", "imports": ["b"]}
start → number
""")

        ruleManagerB = parseIR("""{"id": "b"}
number → [0-9]
""")

        inputs = [
            "0",
            "1",
        ]

        projectManager = ProjectManager([ruleManagerA, ruleManagerB])
        projectManager.processProductions()
        self.runSubtests(ruleManagerA, ruleManagerA, inputs, inputs)

    #@unittest.skip("import")
    def test_import_add(self):
        ruleManagerA = parseIR("""{"id": "a", "imports": ["b"]}
start → number
number → [a-z]
""")

        ruleManagerB = parseIR("""{"id": "b"}
number → [0-9]
""")

        inputs = [
            "0",
            "1",
            "a",
            "b"
        ]

        projectManager = ProjectManager([ruleManagerA, ruleManagerB])
        projectManager.processProductions()
        self.runSubtests(ruleManagerA, ruleManagerA, inputs, inputs)


    #@unittest.skip("import")
    def test_import_import_start(self):
        #TODO: start production should be in the ir!
        ruleManagerA = parseIR("""{"id": "a", "imports": ["b"]}
number → [a-z]
""")

        ruleManagerB = parseIR("""{"id": "b"}
start → number
number → [0-9]
""")

        inputs = [
            "0",
            "1",
            "a",
            "b"
        ]

        projectManager = ProjectManager([ruleManagerA, ruleManagerB])
        projectManager.processProductions()
        self.runSubtests(ruleManagerA, ruleManagerA, inputs, inputs, "b-1-0")

    @unittest.skip("bracket_transform")
    def test_bracket_transform(self):
        ruleManagerA = parseIR("""{"id": "ir"}
start → starts
starts → glue1 | glue2 | abc1 | abc2 | abc4
glue1 → ( "glue1" "glue2" )
glue2 → ( "glue3" "glue4" ){"pad":true}
abc1 → ( "foo1" | "boo1" )
abc2 → ( "foo2" | "boo2" ){"pad":true}
abc4 → ( "foo4" | "boo4" ){"pad":true} | ( "Myfoo4" | "Myboo4" ){"pad":true}
""")
#
        inputs = [
            "glue1 glue2",
            "glue3 glue4",
            "boo1",
            "foo1",
            "boo1",
            "foo2",
            "boo2",
            "foo4",
            "boo4",
            "Myfoo4",
            "Myboo4",
            ]

        outputs = [
            "glue1glue2",
            " glue3glue4 ",
            "boo1",
            "foo1",
            "boo1",
            " foo2 ",
            " boo2 ",
            " foo4 ",
            " boo4 ",
            " Myfoo4 ",
            " Myboo4 ",
            ]

        projectManager = ProjectManager([ruleManagerA, ruleManagerA])
        projectManager.processProductions()

        self.runSubtests(ruleManagerA, ruleManagerA, inputs, outputs, "ir-1-0")


    @unittest.skip("bracket_transform")
    def test_bracket_transform_manual(self):
        #TODO: Implement the modification of a production above you...
        ruleManagerA = parseIR("""{"id": "mid", "imports": ["ir"]}
sub{"compatible": "b-1-0"} → "(" to_match ")" settings{"opt": true}
new_name ⇇ [a-zA-A0-9_-]
""")

        ruleManagerB = parseIR("""{"id": "ir"}
productions → production{"alo": true}
production →  name separator{"pad": false} "|"{"opt": true} to_match
to_match → sub | sub "|"{"pad": true} to_match
sub → token settings{"opt": true}
settings → "{" token{"alo": true, "opt": true} "}"
name → [a-zA-Z0-9]+
token → [a-zA-Z0-9!<>#$\\\\"\\\\'\\\\+\\\\-\\\\*_\\\\.!:]+
separator → ":"
""")

        inputs = [
            "abc:\"foo\"",
            "abc:\"foo\" | glue",
            "abc:\"foo\"{'pad':true}",
            "abc:(\"foo\" | glue)",
            "abc:(\"foo\" | glue){'pad':true}",
            "abc:(\"foo\" | glue){'pad':true} | (\"foo\" | glue){'pad':true}",
            ]
        outputs = [
            "abc:\"foo\"",
            "abc:\"foo\" | glue",
            "abc:\"foo\"{'pad':true}",
"""abc:[a-zA-A][a-zA-A0-9_-]*
[a-zA-A][a-zA-A0-9_-]*: "foo" | glue
""",
            "abc:(\"foo\" | glue){'pad':true}",
            "abc:(\"foo\" | glue){'pad':true} | (\"foo\" | glue){'pad':true}",
        ]

        projectManager = ProjectManager([ruleManagerA, ruleManagerB])
        projectManager.processProductions()

        self.runSubtests(ruleManagerA, ruleManagerA, inputs, inputs, "ir-1-0")
        self.runSubtests(ruleManagerA, ruleManagerB, inputs, inputs, "ir-1-0")
