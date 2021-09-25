import uuid
import unittest

from .late import *
from .read import *

class ESRAP(unittest.TestCase):
    #@unittest.skip("impl |")
    def test_add(self):
        uuids = [uuid.uuid4() for i in range(10)]
        prodA = Productiongenerator.createAllProductions([
            ([uuids[0]], 'calculation', 'term'),
            ([uuids[1]], 'term', 'number "+" term'),
            ([uuids[2]], 'term', 'number'),
            ([uuids[3]], 'number', '[0-9]')])
        input = "1+2"
        outputExpect = input
        matched = match(Productions(prodA), tokenize(input))

        self.assertNotEqual(matched, None)
        vals = matched.fullStr()
        esr = matched.esrap(Productions(prodA))
        self.assertEqual(esr, outputExpect)

    #@unittest.skip("impl |")
    def test_add2(self):
        uuids = [uuid.uuid4() for i in range(10)]
        prodA = Productiongenerator.createAllProductions([
            ([uuids[0]], 'calculation', 'term'),
            ([uuids[1]], 'term', 'number "+" term'),
            ([uuids[2]], 'term', 'number'),
            ([uuids[3]], 'number', '[0-9]')])
        input = "1+2+3"
        outputExpect = input
        matched = match(Productions(prodA), tokenize(input))

        self.assertNotEqual(matched, None)
        vals = matched.fullStr()
        esr = matched.esrap(Productions(prodA))
        self.assertEqual(esr, outputExpect)
    
    #@unittest.skip("impl |")
    def test_add_rename(self):
        uuids = [uuid.uuid4() for i in range(10)]
        prodA = Productiongenerator.createAllProductions([
            ([uuids[0]], 'calculation', 'term'),
            ([uuids[1]], 'term', 'number "+" term'),
            ([uuids[2]], 'term', 'number'),
            ([uuids[3]], 'number', '[0-9]')])
        prodB = Productiongenerator.createAllProductions([
            ([uuids[0]], 'calculation', 'term'),
            ([uuids[1]], 'term', 'number "plus"{"pad": true} term'),
            ([uuids[2]], 'term', 'number'),
            ([uuids[3]], 'number', '[0-9]')])
        input = "1+2+3"
        outputExpect = '1 plus 2 plus 3'
        matched = match(Productions(prodA), tokenize(input))

        self.assertNotEqual(matched, None)
        vals = matched.fullStr()
        esr = matched.esrap(Productions(prodB))
        self.assertEqual(esr, outputExpect)


    def __check(self, prodA, prodB, input: str, output: str, begin = None):
        matched = match(Productions(prodA), tokenize(input), begin)
        print(f'input: {input}, matched: {matched}, expected output: {output}')
        if matched == None:
            self.assertEqual(output, None)
        else:
            self.assertNotEqual(output, None)
            vals = matched.fullStr()
            esr = matched.esrap(Productions(prodB))
            print(f'input: {input}, esr: {esr}, expected output: {output}')
            self.assertEqual(esr, output)

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
            self.__check(prodD, prodF, input, output)

    def test_reorder(self):
        uuids = [uuid.uuid4() for i in range(10)]
        prodA = Productiongenerator.createAllProductions([
            ([uuids[0]], 'calculation', 'term{"id": 0} "," term{"id": 1} "," term{"id": 2}'),
            ([uuids[1]], 'term', '[0-9]')])
        
        prodB = Productiongenerator.createAllProductions([
            ([uuids[2]], 'calculation', 'term{"id": 2} "," term{"id": 1} "," term{"id": 0}', uuids[0]),
            ([uuids[1]], 'term', '[0-9]')])

        inputs = ["1,2,3"]
        outputs = ['3,2,1']

        for input, output in zip(inputs, outputs): 
            self.__check(prodA, prodB, input, output)

    def test_reorder2(self):
        uuids = [uuid.uuid4() for i in range(10)]
        prodA = Productiongenerator.createAllProductions([
            ([uuids[0]], 'calculation', 'term{"id": 0} "," term{"id": 1} "," term{"id": 2}'),
            ([uuids[1]], 'term', '[0-9]')])
        
        prodB = Productiongenerator.createAllProductions([
            ([uuids[2]], 'calculation', 'term{"id": 2, "pad":true} term{"id": 1, "pad":true} term{"id": 0, "pad":true}', uuids[0]),
            ([uuids[1]], 'term', '[0-9]')])

        inputs = ["1,2,3"]
        outputs = [' 3  2  1 ']

        for input, output in zip(inputs, outputs): 
            self.__check(prodA, prodB, input, output)

    def test_reorder3(self):
        uuids = [uuid.uuid4() for i in range(10)]
        prodA = Productiongenerator.createAllProductions([
            ([uuids[0]], 'calculation', 'term{"id": 0} "," term{"id": 1} "," term{"id": 2}'),
            ([uuids[1]], 'term', '[0-9]{"id": 0}')])
        
        prodB = Productiongenerator.createAllProductions([
            ([uuids[2]], 'calculation', 'term2{"id": 2} term2{"id": 1} term2{"id": 0}', uuids[0]),
            ([uuids[3]], 'term2', ' "(" [0-9]{"id": 0} ")" ', uuids[1])])

        inputs = ["1,2,3"]
        outputs = ['(3)(2)(1)']

        for input, output in zip(inputs, outputs): 
            self.__check(prodA, prodB, input, output)

    #@unittest.skip("impl |")
    def test_or(self):
        uuids = [uuid.uuid4() for i in range(10)]
        begin = [uuids[0], uuids[1]]
        prodA = Productiongenerator.createAllProductions([(begin, 'number', '"a" | "b"')])
        inputs = ["a", "b"]
        for input in inputs: 
            self.__check(prodA, prodA, input, input, begin)

    def test_zeroOrMore(self):
        pass

    #@unittest.skip("impl |")
    def test_or2(self):
        uuids = [uuid.uuid4() for i in range(10)]
        begin = [uuids[0], uuids[1]]
        beginB = [uuids[0], uuids[1], uuids[2]]
        prodA = Productiongenerator.createAllProductions([(begin, 'number', '"a" | "b"')])
        prodB = Productiongenerator.createAllProductions([(beginB, 'number', '"a" | "b" | "c"')])
        inputs = ["a", "b"]
        for input in inputs: 
            self.__check(prodA, prodB, input, input, begin)

    #@unittest.skip("impl |")
    def test_or3(self):
        uuids = [uuid.uuid4() for i in range(10)]
        begin = [uuids[0], uuids[1]]
        prodA = Productiongenerator.createAllProductions([(begin, 'number', '"a" | [0-9]')])
        inputs = ["a", "0", "9"]
        for input in inputs: 
            self.__check(prodA, prodA, input, input, begin)

    #@unittest.skip("impl |")
    def test_or4(self):
        uuids = [uuid.uuid4() for i in range(10)]
        begin = [uuids[0], uuids[1]]
        beginB = [uuids[0], uuids[1], uuids[2]]
        prodA = Productiongenerator.createAllProductions([(begin, 'number', '"a" | [0-9]')])
        prodB = Productiongenerator.createAllProductions([(begin, 'number', '"a" | [0-9] "BOOP"')])
        inputs = ["a", "0", "9"]
        outputs = ["a", "0BOOP", "9BOOP"]
        
        for input, output in zip(inputs, outputs): 
            self.__check(prodA, prodB, input, output, begin)

    #@unittest.skip("impl |")
    def test_or4(self):
        uuids = [uuid.uuid4() for i in range(10)]
        begin = [uuids[0], uuids[1]]
        prodA = Productiongenerator.createAllProductions([
            (begin, 'number', '"a" | [0-9] txt'),
            ([uuids[4]], 'txt', ' "BOOPly" ')])
        prodB = Productiongenerator.createAllProductions([
            ([uuids[0]], 'number', '"a"'),
            ([uuids[3]], 'number', '[0-9]{"id": 0} txt{"id": 1}', uuids[1]),
            ([uuids[4]], 'txt', ' "BOOPly" ')
            ])
        inputs = ["a", "0 BOOPly", "9 BOOPly"]
        outputs = ["a", "0BOOPly", "9BOOPly"]
        
        for input, output in zip(inputs, outputs): 
            self.__check(prodA, prodB, input, output, begin)

    def test_zeroOrMore(self):
        pass

    #@unittest.skip("impl |")
    def test_alo(self):
        uuids = [uuid.uuid4() for i in range(10)]
        prodA = Productiongenerator.createAllProductions([([uuids[0]], 'number', '"a"{"alo": true, "pad": true}')])
        inputs = [" a ", " a  a ", " a  a  a "]
        for input in inputs:
            self.__check(prodA, prodA, input, input)

    #@unittest.skip("impl |")
    def test_alo2(self):
        uuids = [uuid.uuid4() for i in range(10)]
        prodA = Productiongenerator.createAllProductions([
            ([uuids[0]], 'numbers', 'number{"alo": true}'),
            ([uuids[1]], 'number', '"a"{"pad": true}'),
            ])
        inputs = [" a ", " a  a ", " a  a  a "]
        for input in inputs:
            self.__check(prodA, prodA, input, input)

    #@unittest.skip("impl |")
    def test_alo3(self):
        uuids = [uuid.uuid4() for i in range(10)]
        prodA = Productiongenerator.createAllProductions([([uuids[0]], 'number', '"a"{"alo": true, "pad": true} "b"{"alo": true, "pad": true}')])
        inputs = [" a  b ", " a  b ", " a  a  b ", " a  a  a  b ", " a  b  b ", " a  b  b  b ", " a  a  b  b "]
        for input in inputs:
            self.__check(prodA, prodA, input, input)

    #@unittest.skip("impl |")
    def test_alo4(self):
        uuids = [uuid.uuid4() for i in range(10)]
        prodA = Productiongenerator.createAllProductions([
            ([uuids[0]], 'numbers', 'number{"alo": true}'),
            ([uuids[1]], 'number', '"a" ","'),
            ])
        inputs = ["a,", "a,a,", "a,a,a,"]
        for input in inputs:
            self.__check(prodA, prodA, input, input)

    #@unittest.skip("impl |")
    def test_alo5(self):
        uuids = [uuid.uuid4() for i in range(10)]
        prodA = Productiongenerator.createAllProductions([
            ([uuids[0]], 'numbers', 'number_sep{"alo": true} number'),
            ([uuids[1]], 'number_sep', 'number ","'),
            ([uuids[2]], 'number', '"a"'),
            ])
        inputs = ["a,a", "a,a,a", "a,a,a,a"]
        for input in inputs:
            self.__check(prodA, prodA, input, input)


    #@unittest.skip("impl |")
    def test_optional(self):
        uuids = [uuid.uuid4() for i in range(10)]
        prodA = Productiongenerator.createAllProductions([([uuids[0]], 'number', '"a" "b"{"opt": true, "pad": true}')])
        inputs = ["a", "a b "]
        for input in inputs:
            self.__check(prodA, prodA, input, input)

    #@unittest.skip("impl |")
    def test_optional2(self):
        uuids = [uuid.uuid4() for i in range(10)]
        prodA = Productiongenerator.createAllProductions([([uuids[0]], 'number', '"a" "b"{"opt": true, "pad": true} "a"{"pad": true}')])
        inputs = ["a a ", "a b  a "]
        for input in inputs:
            self.__check(prodA, prodA, input, input)

    #@unittest.skip("impl |")
    def test_optional3(self):
        uuids = [uuid.uuid4() for i in range(10)]
        prodA = Productiongenerator.createAllProductions([([uuids[0]], 'number', '"a" "b"{"opt": true, "pad": true} "b"{"opt": true, "pad": true} "a"{"pad": true}')])
        inputs = ["a a ", "a b  a ", "a b  b  a "]
        outputs = copy.copy(inputs)
        outputs[1] = None
        for input, output in zip(inputs, outputs):
            self.__check(prodA, prodA, input, output)
        

    #@unittest.skip("impl |")
    def test_optional4(self):
        uuids = [uuid.uuid4() for i in range(10)]
        prodA = Productiongenerator.createAllProductions([([uuids[0]], 'number', '"a"{"opt": true, "pad": true} "b"')])
        inputs = [" a b", "b"]
        for input in inputs:
            self.__check(prodA, prodA, input, input)

    #@unittest.skip("impl |")
    def test_optional5(self):
        uuids = [uuid.uuid4() for i in range(10)]
        prodA = Productiongenerator.createAllProductions([
            ([uuids[0]], 'number', '"a"{"opt": true, "pad": true} ab'),
            ([uuids[1]], 'ab', '"ab"'),
            ])
        inputs = [" a ab", "ab"]
        for input in inputs:
            self.__check(prodA, prodA, input, input)

    #@unittest.skip("impl |")
    def test_optional6(self):
        uuids = [uuid.uuid4() for i in range(10)]
        prodA = Productiongenerator.createAllProductions([
            ([uuids[0]], 'number', '"a"{"opt": true, "pad": true} ab'),
            ([uuids[1]], 'ab', '"a"{"pad": true} "b"{"pad": true}'),
            ])
        inputs = [" a  b ", " a  a  b "]
        for input in inputs:
            self.__check(prodA, prodA, input, input)

    #@unittest.skip("impl |")
    def test_optional_alo(self):
        uuids = [uuid.uuid4() for i in range(10)]
        prodA = Productiongenerator.createAllProductions([([uuids[0]], 'number', '"a"{"alo": true, "pad": true} "b"{"opt": true, "pad": false}')])
        inputs = [" a ", " a b", " a  a b"]
        for input in inputs:
            self.__check(prodA, prodA, input, input)


    #@unittest.skip("impl |")
    def test_optional_alo2(self):
        uuids = [uuid.uuid4() for i in range(10)]
        prodA = Productiongenerator.createAllProductions([
            ([uuids[0]], 'numbers', 'number_sep{"alo": true} number{"opt": true}'),
            ([uuids[1]], 'number_sep', 'number ","'),
            ([uuids[2]], 'number', '"a"'),
            ])
        inputs = ["a,a", "a,a,a", "a,a,a,a", "a,", "a,a,", "a,a,a,"]
        #inputs = ['a,a']
        for input in inputs:
            self.__check(prodA, prodA, input, input)


    #@unittest.skip("impl |")
    def test_any(self):
        uuids = [uuid.uuid4() for i in range(10)]
        prodA = Productiongenerator.createAllProductions([
            ([uuids[0]], 'numbers', 'number_sep{"alo": true, "opt": true}'),
            ([uuids[1]], 'number_sep', 'number ","'),
            ([uuids[2]], 'number', '"a"'),
            ])
        inputs = ["a,", "a,a,", "a,a,a,"]
        for input in inputs:
            self.__check(prodA, prodA, input, input)

    #@unittest.skip("impl |")
    def test_any2(self):
        uuids = [uuid.uuid4() for i in range(10)]
        prodA = Productiongenerator.createAllProductions([
            ([uuids[0]], 'numbers', '"b"{"pad": true} number_sep{"alo": true, "opt": true}'),
            ([uuids[1]], 'number_sep', 'number ","'),
            ([uuids[2]], 'number', '"a"'),
            ])
        inputs = [" b ", " b a,", " b a,a,", " b a,a,a,"]
        for input in inputs:
            self.__check(prodA, prodA, input, input)

    #@unittest.skip("impl |")
    def test_any3(self):
        uuids = [uuid.uuid4() for i in range(10)]
        prodA = Productiongenerator.createAllProductions([
            ([uuids[0]], 'numbers', '"b"{"pad": true} "abc"{"alo": true, "opt": true, "pad": true}'),
            ])
        inputs = [" b ", " b  abc ", " b  abc  abc ", " b  abc  abc  abc "]
        for input in inputs:
            self.__check(prodA, prodA, input, input)

    #@unittest.skip("impl |")
    def test_any4(self):
        uuids = [uuid.uuid4() for i in range(10)]
        prodA = Productiongenerator.createAllProductions([
            ([uuids[0]], 'numbers', 'number_sep{"alo": true, "opt": true} "b"'),
            ([uuids[1]], 'number_sep', 'number ","'),
            ([uuids[2]], 'number', '"a"'),
            ])
        inputs = ["b", "a,b", "a,a,b", "a,a,a,b"]
        for input in inputs:
            self.__check(prodA, prodA, input, input)

    @unittest.skip("grouping")
    def test_grouping(self):
        uuids = [uuid.uuid4() for i in range(10)]
        prodA = Productiongenerator.createAllProductions([
            ([uuids[0]], 'letter', '( "a" )'),
            ])
        inputs = ["a"]
        for input in inputs:
            self.__check(prodA, prodA, input, input)

    
    def test_read(self):
        uuids = [uuid.uuid4() for i in range(10)]
        prodA = parseIR("""
production →  name separator{"pad": false} "|"{"opt": true} to_match
to_match → sub | sub "|"{"pad": true} to_match
sub → token settings{"opt": true} | "(" to_match ")" settings{"opt": true}
settings → "{" tokens{"alo": true, "opt": true} "}"
token → name | '\\'' tokens '\\''
name → [a-zA-Z0-9]*
tokens → [a-zA-Z0-9!<>#$\\\\+\\\\-\\\\*_\\\\.!]*
separator → ":"
""".splitlines())
        inputs = [
            "abc:\"foo\"",
            "abc:\"foo\" | boo",
            "abc:\"foo\"{'pad':true}",
            "abc:(\"foo\" | boo)",
            "abc:(\"foo\" | boo){'pad':true}",
            "abc:(\"foo\" | boo){'pad':true}",
            #"abc:(\"foo\" | boo){'pad':true} | (\"foo\" | boo){'pad':true}",
            ]
        for input in inputs:
            self.__check(prodA, prodA, input, input)

    #@unittest.skip("impl |")
    def test_bnf(self):
        uuids = [uuid.uuid4() for i in range(50)]
        begin = [uuids[0],  uuids[1]]
        
        """
        prodBNF = Productiongenerator.createAllProductions([
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
            
        
        
        prodBNF = Productiongenerator.createAllProductions([
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
            outputExpect = input
            interupts = ['+', '-', '*', '/', '(', ')', '\n', ',']
            matched = match(Productions(prodBNF), tokenize(input, interupts), begin)

            self.assertNotEqual(matched, None)
            vals = matched.fullStr()
            esr = matched.esrap(Productions(prodBNF))
            self.assertEqual(esr, outputExpect)

