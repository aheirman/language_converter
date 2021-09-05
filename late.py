import re
import copy
import uuid
from enum import Enum
import json
import unittest

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Token:
    def __init__(self, tok, settings):
        self.tok = tok
        self.settings = settings

    def __str__(self):
        return f'{{token: {self.tok}, settings: {self.settings}}}' if self.settings != {} else f'{{token: {self.tok}}}'

class NonTerminal:
    def __init__(self, tok):
        self.token = tok
        self.token.settings.pop('regex', None)

    def __str__(self):
        return f'{{NonTerminal: {self.token}}}'

    def name(self):
        return self.token.tok

class Terminal:
    def __init__(self, rule: Token):
        self.token = rule
        #print(f'{self.token}, {self.token.settings}')
        if (self.token.settings['regex']):
            self.rule = rule
            print(f'Terminal with regex rule: {self.token.tok}')
            self.reg = re.compile(self.token.tok)
        else:
            self.rule = rule
            #print(f"Terminal with rule: {self.token.tok}")

    
    def match(self, input):
        if (self.token.settings['regex']):
            print(f'Terminal regex match: "{input}", "{self.rule}"')
            r = self.reg.match(input)
            t = False if r==None else (r.start() == 0)
            print(f'Terminal regex match: "{input}", "{self.rule}, res: {r}, bool: {t}"')
            return t
        else:
            print(f'Terminal match: "{input}", "{self.rule}", "{input == self.rule}"')
            return input == self.rule.tok

    def name(self):
        return self.rule.tok

    def __str__(self):
        return f'{{Terminal: "{self.rule}}}'

class Production:
    def process(self, productions):
        steps = []
        for tok in self.tokens:
            #print(f'production name: {self.name}, process token: {tok}')
            prod = productions.productionWithNameExists(tok.tok)
            if (prod == False):
                steps.append(Terminal(tok))
            else:
                steps.append(NonTerminal(tok))
        self.steps = steps
    
    def __init__(self, uuid, name: str, tokens: list[Token], uuid_compat = None):
        self.name = name
        self.tokens = tokens
        self.uuid = uuid
        self.uuid_compat = uuid_compat

    def __str__(self):

        #tokensStr = ' '.join([str(elem) for elem in self.tokens])
        stepStr   = '\n\t' + '\n\t'.join([str(elem) for elem in self.steps])
        #return f'Production {{ name: {self.name}  tokens: [ {tokensStr} ], {stepStr} }}'
        uuidCompatStr = f', uuid_compat: {self.uuid_compat}' if self.uuid_compat != None else ''
        return f'Production {{ name: {self.name}, {stepStr}{uuidCompatStr}'
        #return f'Production {{ name: {self.name}, {tokensStr} }}'

class TokenizeSettings(Enum):
    PRE = 0
    NORMAL = 1
    QUOTE = 2
    REGEX = 6
    REGEXPOSEND = 7
    VAL_FINISHED = 3
    SETTINGS_BLOCK = 4
    END = 5

class Productiongenerator():
    @staticmethod
    def tokenize(input):
        print(f'Production tokenize {input}')
        tokens = [[]]
        currProduction = 0
        curr = ''
        strSettings = ''
        status = TokenizeSettings.PRE
        defSettings = {'regex': True}
        settings = copy.copy(defSettings)
        escaped = False
        nextEscaped = False

        def tokenEnd(status, c):
            if (status in [
                TokenizeSettings.NORMAL,
                TokenizeSettings.END,
                TokenizeSettings.VAL_FINISHED,
                TokenizeSettings.REGEXPOSEND]):
                return True
            return False

        for c in input:
            escaped = nextEscaped
            nextEscaped = False
            print(f'currProduction: {currProduction}, escaped: {escaped}, status: {status}, tokens: ' + ' '.join(map(str, tokens[currProduction])) + ', adding char: ' + c)
            
            if (tokenEnd(status, c) and (c == ' ' or c == '\n')):
                #Start new token
                #print(f'settings: {settings} strSettings: {strSettings}')
                if (strSettings != ''):
                    settings.update(json.loads(strSettings))
                    #print(f'---settings: {settings}')
                tokens[currProduction].append(Token(curr, settings))
                status = TokenizeSettings.PRE

                settings = copy.copy(defSettings)
                curr = ''
                strSettings = ''
                if c == '\n':
                    #A real token!
                    tokens[currProduction].append(Token('\n', {}))

            elif (status == TokenizeSettings.PRE):
                if (c == ' '):
                    pass
                elif (c == '|'):
                    currProduction += 1
                    tokens.append([])
                elif ( c == '"'):
                    status = TokenizeSettings.QUOTE
                    settings['regex'] = False
                elif ( c == '['):
                    status = TokenizeSettings.REGEX
                    settings['regex'] = True
                    curr += c
                else:
                    status = TokenizeSettings.NORMAL
                    curr += c
            elif (status == TokenizeSettings.REGEX):
                if (c == '\\' and not escaped):
                    nextEscaped = True
                elif (c == ']' and not escaped):
                    curr += c
                    status = TokenizeSettings.REGEXPOSEND
                else:
                    curr += c
            elif (status == TokenizeSettings.QUOTE and not escaped and c == '"'):
                    status = TokenizeSettings.VAL_FINISHED
            elif ((status in [TokenizeSettings.VAL_FINISHED, TokenizeSettings.NORMAL, TokenizeSettings.REGEXPOSEND]) and c == '{'):
                status = TokenizeSettings.SETTINGS_BLOCK
                strSettings += c
            elif (status == TokenizeSettings.SETTINGS_BLOCK and c == '}'):
                status = TokenizeSettings.END
                strSettings += c
            elif (status in [TokenizeSettings.NORMAL, TokenizeSettings.QUOTE, TokenizeSettings.REGEXPOSEND]):
                if (c == '\\' and not escaped):
                    nextEscaped = True
                else:
                    curr += c
            elif (status == TokenizeSettings.SETTINGS_BLOCK):
                strSettings += c
            else:
                print(f'ERROR: status: {status}')
                assert False

        if(curr != ''):
            #print(f'strSettings: {strSettings}')
            if (strSettings != ''):
                settings.update(json.loads(strSettings))
            tokens[currProduction].append(Token(curr, settings))
        
        #print(f'Production tokenize tokens: {str([str(tok) for tok in tokens])}')
        return tokens

    @staticmethod
    def __createProductions(uuids, name, input: str, uuid_compat = None) -> list[Production]:
        tokenList = Productiongenerator.tokenize(input)
        prods = [Production(uuids[i], name, tokens, uuid_compat) for i, tokens in enumerate(tokenList)]
        return prods

    @staticmethod
    def createAllProductions(list):
        prods = []
        for rule in list:
            prods.extend(Productiongenerator.__createProductions(*rule))
        return prods

class Productions:

    def generateMap(self):
        self.RuleProdMap = dict(zip([prod.stdandard for prod in self.productions], [prod.stdandard for prod in self.productions]))

    def __init__(self, productions: list[Production]):
        self.productions = productions
        for prod in self.productions:
            prod.process(self)

    def __str__(self):
        return '\n'.join([str(elem) for elem in self.productions])

    def getProduction(self, uuid):
        for prod in self.productions:
            if (prod.uuid == uuid):
                return prod
        return None

    def getCompatableProduction(self, uuid):
        for prod in self.productions:
            if (prod.uuid_compat == uuid):
                return prod
        return None

    def productionWithNameExists(self, name: str):
        for prod in self.productions:
            if (prod.name == name):
                return True
        return False

    def getEquivalentProduction(self, standardese) -> Production:
        pass


def containsAndTrue(dict, key):
    return (key in dict) and (dict[key] == True)

class State:
    def __init__(self, production, originPosition):
        self.production = production
        self.position = 0
        self.originPosition = originPosition
        self.values = []
    
    def __str__(self):
        str = f'{self.production.name.ljust(15)} → {{'
        for index, step in enumerate(self.production.steps):
            if (index == self.position):
                str += ' ȣ '
            
            str += step.name() + ' '
        if (len(self.production.steps) == self.position):
            str += ' ȣ '

        str += '},'
        str = str.ljust(40)
        str += f'from {self.originPosition}'
        return str
    
    def name(self) -> str:
        return self.production.name

    def nextIsTerminal(self):
        term = isinstance(self.production.steps[self.position], Terminal)
        #print(f'name: {self.name()}, pos: {self.position}, Terminal: {term}')
        return term

    def match(self, input):
        assert self.nextIsTerminal()
        return self.production.steps[self.position].match(input)

    def advance(self, state):
        
        self.position += 1
        self.values.append(state)
        #print(f'state: {self.production.name}, at {self.position} added {str(state)}, now contains {len(self.values)} values')

    def isCompleted(self):
        return self.position == len(self.production.steps)

    def isinstance(self, NonTerminalName: str):
        #print(f'state: isinstance NonTerminalName: {NonTerminalName}, self.position: {self.position}')
        if not self.isCompleted():
            next = self.production.steps[self.position]
            if not isinstance(next, NonTerminal):
                return False
            return next.name() == NonTerminalName
        else:
            return False

    def getNextName(self):
        return self.production.steps[self.position].name()

    def fullStr(self):
        tostr = lambda inVal: inVal.fullStr() if isinstance(inVal, State) else inVal
        return f'{self.production.name} {{' + (', '.join(map(tostr, self.values))) + '}'

    def esrapSelf(self):
        req = lambda x: x.esrap() if isinstance(x, State) else x
        return ''.join(map(req, self.values))

    """
    NOTE:   This method works for both implicit and explicit compatibility 
            compat may have more non terminals
            That's why it is on the lhs
    """
    def __genIndexToIndices(self, compat, explicit):
        indexToIndex = {}
        print(f'self production:       {self.production}')
        print(f'compatible production: {compat}')

        count = 0
        for compatIndex, step in enumerate(compat.steps):
            if (isinstance(step, NonTerminal) or (step.token.settings['regex'] == True)):
                selfIndex = step.token.settings['id'] if explicit else count
                if (not selfIndex in indexToIndex):
                    indexToIndex[selfIndex] = []
                    #print(f'empty dictionary array at : {selfIndex}')
                
                indexToIndex[selfIndex].append(compatIndex)
                count += 1
        return indexToIndex

    def esrap(self, productions: Productions):
        
        #print(f'-------BEGIN ESRAP OF {self.name()}-------')
        #print(self.fullStr())
        otherProd = productions.getProduction(self.production.uuid)
        equal = True
        if otherProd == None:
            #print(f'=====CONVERSIONS ARE NEEDED=====')
            # conversions are needed
            otherProd = productions.getCompatableProduction(self.production.uuid)
            equal = False

        stepsA = self.production.steps
        stepsB = otherProd.steps

        indexToIndices = self.__genIndexToIndices(otherProd, not equal)
        #print(f'indexToIndices: {indexToIndices}')

        #Replace all
        strs = [None]*len(stepsB)

        # Set strings
        for i, step in enumerate(otherProd.steps):
            if (isinstance(step, Terminal) and not step.rule.settings['regex']):
                string = step.rule.tok
                if containsAndTrue(step.rule.settings, 'pad'):
                    string = ' ' + string + ' '
                strs[i] = string
        # Set (Non)Terminals

        selfCountExludingStr = 0
        for i, step in enumerate(stepsA):
            typeName = type(step).__name__
            #print(f'{bcolors.OKGREEN}status: name {self.name()}, i: {i}, step: {step}, typeName: {typeName}{bcolors.ENDC}')

            val = self.values[i]
            typeName = type(val).__name__
            #print(f'{bcolors.OKGREEN}status: name {self.name()}, i: {i}, val: {val}, typeName: {typeName}{bcolors.ENDC}')
            if isinstance(val, State):
                compatIndices = indexToIndices[selfCountExludingStr]
                
                for compatIndex in compatIndices:
                    strs[compatIndex] = val.esrap(productions)
                selfCountExludingStr += 1
            else:
                if i in indexToIndices:
                    compatIndex = indexToIndices[i][0]
                    assert(len(indexToIndices[i]) == 1)
                    if (isinstance(val, Terminal)):
                        isRegex = stepsA[i].rule.settings['regex']
                        isRegexB = stepsB[compatIndex].rule.settings['regex']
                        #print(f'isRegex: {isRegex}, isRegexB: {isRegexB}')
                        assert(isRegex == isRegexB)

                        strs[compatIndex] = val
                        selfCountExludingStr += 1
                    elif (isinstance(val, NonTerminal)):
                        rule = stepsB[compatIndex].rule
                        string = rule.tok
                        if containsAndTrue(rule.settings, 'pad'):
                            string = ' ' + string + ' '

                        strs[compatIndex] = string
                        selfCountExludingStr += 1
                    elif (isinstance(val, str)):
                        strs[compatIndex] = val
                    else:
                        typeName = type(val).__name__
                        print(f'ERROR: {typeName}')
                        assert False
        
        #print(f'-------END ESRAP OF {self.name()}-------')
        #print(strs)
        strs = [str if str else 'BOOO' for str in strs ]
        return ''.join(strs)


class Column():
    def __init__(self, productions, states):
        self.states = states
        self.productions = productions

    def containsState(self, name: str, index: int):
        for state in self.states:
            if (state.production.name == name and state.position == index):
                #print(f'{state.production.name} == {name}, {state.position} == {index}')
                #print(f'containsState true')
                return True
        #print(f'containsState false')
        return False

    def predict(self, productionName: str, currentChart):
        new = []
        #print(f'predict: productionName: {productionName}, currentChart {currentChart}')
        for prod in self.productions.productions:
            #print(f'predict: prod.name {prod.name}')
            if (prod.name == productionName):
                #print('matching name')
                if (not self.containsState(productionName, 0)):
                    #print(f'predicted: {prod.name}')
                    new.append(State(prod, currentChart))
        #print(f'predict: {new}')
        return new
    
    def append(self, state):
        self.states.append(state)
    
    def extend(self, states):
        self.states.extend(states)


def complete(table, state: State):
    print(f'complete {state.production.name}, {state.originPosition}')
    colJ = table[state.originPosition].states
    newStates = []
    for stateJ in colJ:
        if stateJ.isinstance(state.production.name):
            newState = copy.deepcopy(stateJ)
            newState.advance(state)
            newStates.append(newState)
    return newStates

def tokenize(input: str):
        tokens = []
        interupts = ['+', '-', '*', '/', '(', ')', '\n']
        curr = ''
        status = TokenizeSettings.NORMAL
        for c in input:
            #print('status: ' + ' '.join(map(str, ret)) + ' adding char: ' + c)
            if ((status == TokenizeSettings.NORMAL) and c == ' '): 
                tokens.append(curr)
                curr = ''
            elif ((status == TokenizeSettings.NORMAL) and (c in interupts)):
                tokens.append(curr)
                tokens.append(c)
                curr = ''
            elif (status == TokenizeSettings.NORMAL):
                curr += c
            else:
                print(f'ERROR: status: {status}')
                assert False

        if(curr != ''):
            tokens.append(curr)
        
        tokens = [tok for tok in tokens if (len(tok) != 0)]
        return tokens

def match(productions: Productions, inTokens: list[str], beginRules: list[uuid.UUID] = None):
    print(str(productions))
    tokenStr = '\n\t'.join([str(tok) for tok in inTokens])
    print(f'tokenized: \n\t{tokenStr}, len: {len(inTokens)}')
    #table = [Column(productions, [State(prod, 0) for prod in productions.productions])]
    topLevelName = productions.productions[0].name
    table = [Column(productions, []) for i in range(len(inTokens)+1)]

    if beginRules == None:
        table[0].extend([State(productions.productions[0], 0)])
    else:
        table[0].extend([State(productions.productions[i], 0) for i in range(len(productions.productions)) if productions.productions[i].uuid in beginRules])
    
    for currentChart, col in enumerate(table):
        #pre
        tok = inTokens[currentChart] if currentChart<len(inTokens) else None
        print(f'------{currentChart}, {tok}------')

        #real work
        for state in col.states:
            print(f'sate name: {state.production.name}')
            if (state.isCompleted()):
                col.states.extend(complete(table, state))
            else:
                if (tok != None and state.nextIsTerminal()):
                    print('Scanning')
                    if (state.match(tok)):
                        print('Scanning matched!')
                        #Scanning
                        newState = copy.copy(state)
                        newState.advance(tok)
                        table[currentChart+1].append(newState)
                else:
                    print('Predicting')
                    #Prediction
                    productionName = state.getNextName()
                    col.extend(col.predict(productionName, currentChart))

        #post
        print('\n'.join(map(str, col.states)))
    
    # Find result
    matches = []
    for status in table[-1].states:
        if (status.originPosition == 0 and status.isCompleted() and status.production.name == topLevelName):
            matches.append(status)
    
    print(f'MATCHES: {matches}')
    if (len(matches) > 1):
        print(f'ERROR: MULTIPLE MATCHES')
        assert False
    return matches[0] if len(matches) == 1 else None


class ESRAP(unittest.TestCase):

    @unittest.skip("impl |")
    def test_add(self):
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
    
    @unittest.skip("impl |")
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

    @unittest.skip("impl |")
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
        
        input = "1*(2+3)"
        outputExpect = '(1*(2))+(1*(3))'
        matched = match(Productions(prodD), tokenize(input))

        self.assertNotEqual(matched, None)
        vals = matched.fullStr()
        esr = matched.esrap(Productions(prodF))
        self.assertEqual(esr, outputExpect)

    @unittest.skip("impl |")
    def test_or(self):
        uuids = [uuid.uuid4() for i in range(10)]
        begin = [uuids[0], uuids[1]]
        prodA = Productiongenerator.createAllProductions([(begin, 'number', '"a" | "b"')])
        inputs = ["a", "b"]
        for input in inputs: 
            outputExpect = input
            matched = match(Productions(prodA), tokenize(input), begin)

            self.assertNotEqual(matched, None)
            vals = matched.fullStr()
            esr = matched.esrap(Productions(prodA))
            self.assertEqual(esr, outputExpect)

    
    def test_func(self):
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
            matched = match(Productions(prodBNF), tokenize(input), begin)

            self.assertNotEqual(matched, None)
            vals = matched.fullStr()
            esr = matched.esrap(Productions(prodBNF))
            self.assertEqual(esr, outputExpect)



def foo():
    #number = Terminal('[0-9]')
    #number2 = Terminal('\+')
    #print(number.rule)
    #print(number2.match('+'))

    uuids = [uuid.uuid4() for i in range(10)]
    
    prodB = [Production(uuids[0], 'number', 'a | b')]
    #prodB = [Production(uuids[0], 'calculation', 'term'), Production(uuids[1], 'term', '"hi" number "hi" "plus"{"pad": true} "hi" term "hi" '), Production(uuids[2], 'term', 'number'), Production(uuids[3], 'number', '[0-9]')]
    #prodC = [Production(uuids[0], 'calculation', 'term'), Production(uuids[1], 'term', 'term " plus " number'), Production(uuids[2], 'term', 'number'), Production(uuids[3], 'number', '[0-9]')]

    #prodD = [Production(uuids[0], 'calculation', 'term'), Production(uuids[1], 'term', 'number "*" "(" term "+" term ")"'), Production(uuids[2], 'term', 'number'), Production(uuids[3], 'number', '[0-9]')]
    #prodE = [Production(uuids[0], 'calculation', 'term'), Production(uuids[1], 'term', 'number "*" "(" term "+" term ")"'), Production(uuids[2], 'term', 'number'), Production(uuids[3], 'number', '[0-9]')]
    #prodF = [Production(uuids[0], 'calculation', 'term'), Production(uuids[4], 'term', '"(" number{"id": 0} "*" "(" term{"id": 1} ")" ")" "+" "(" number{"id": 0} "*" "(" term{"id": 2} ")" ")"', uuids[1]), Production(uuids[2], 'term', 'number'), Production(uuids[3], 'number', '[0-9]')]

    #
    #input = "1+1"
    #input = "1 plus 2 plus 3"
    input = '4*(1*(2+3)+5*(6+7))'


    print('-------------INPUT--------------')
    print(input)
    print('-------------PARSE--------------')
    matched = match(Productions(prodD), input)

    if (matched != None):
        print('----------PARSE RESULT----------')
        vals = matched.fullStr()
        print(vals)
        #esrap(Productions(prodA), matched)
        print('-------------ESRAP-------------')
        esr = matched.esrap(Productions(prodF))
        print('----------ESRAP RESULT----------')
        print(esr)
    else:
        print('No match!')


unittest.main()
