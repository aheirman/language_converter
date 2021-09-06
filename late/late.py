import re
import copy
import uuid
from enum import Enum
import json
#from late.helper import print

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

        settings = self.token.settings
        if (settings['regex']):
            self.rule = rule
            #print(f'Terminal with regex rule: {self.token.tok}')
            self.reg = re.compile(self.token.tok)
        else:
            self.rule = rule
            #print(f"Terminal with rule: {self.token.tok}")

    
    def match(self, input: str):
        if (self.token.settings['regex']):
            #print(f'Terminal regex match: "{input}", "{self.rule}"')
            r = self.reg.match(input)
            t = False if r==None else (r.start() == 0)
            #print(f'Terminal regex match: "{input}", "{self.rule}, res: {r}, bool: {t}"')
            return t
        else:
            #print(f'Terminal match: "{input}", "{self.rule.tok}", "{input == self.rule.tok}"')
            return input == self.rule.tok

    def name(self):
        return self.rule.tok

    def __str__(self):
        return f'{{Terminal: "{self.rule}}}'

class Production:
    def process(self, productions):
        steps = []
        for tok in self.tokens:
            print(f'production name: {self.name}, process token: {tok}')
            
            if containsAndTrue(tok.settings, "regex") or containsAndTrue(tok.settings, "quote"):
                steps.append(Terminal(tok))
            else:
                prod = productions.productionWithNameExists(tok.tok)
                assert prod
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
        #print(f'Production tokenize {input}')
        tokens = [[]]
        currProduction = 0
        curr = ''
        strSettings = ''
        status = TokenizeSettings.PRE
        defSettings = {}
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
            #print(f'currProduction: {currProduction}, escaped: {escaped}, status: {status}, tokens: ' + ' '.join(map(str, tokens[currProduction])) + ', adding char: ' + c)
            
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
                    settings['quote'] = True
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

def containsAndTrueAny(dict, keys):
    for key in keys:
        if (key in dict) and (dict[key] == True):
            return True
    return False


class State:
    def __init__(self, production, originPosition):
        self.production = production
        self.originPosition = originPosition
        self.values = []
        self.positions = [0]
        self.__updatePosition(0)
    


    def __updatePosition(self, pos):
        while not pos == len(self.production.steps):
            set = self.production.steps[pos].token.settings
            if containsAndTrue(set, 'opt'):
                self.positions.append(pos + 1)
            else:
                break
            pos += 1


    def __str__(self):
        str = f'{self.production.name.ljust(15)} → {{'
        for index, step in enumerate(self.production.steps):
            if (index in self.positions):
                str += ' ȣ '
            
            str += step.name() + ' '
        if (len(self.production.steps) in self.positions):
            str += ' ȣ '

        str += '},'
        str = str.ljust(40)
        str += f'from {self.originPosition}'
        return str
    
    def name(self) -> str:
        return self.production.name

    def containsNextTerminal(self):
        #print(f'name: {self.name()}, pos: {self.position}, Terminal: {term}')
        for pos in self.positions:
            if pos < len(self.production.steps):
                if isinstance(self.production.steps[pos], Terminal):
                    return True
        return False

    """
    def match(self, input):
        assert self.containsNextTerminal()
        
        for pos in self.positions:
            if self.production.steps[pos].match(input):
                return True

        return False
     """

    def advance(self, state, index) -> list[str]:
        print(f'advance index: {index}, self.positions: {self.positions}')
        assert index < len(self.positions)
        myPos = self.positions[index]
        nextPos = myPos + 1
        

        # We avanced to the alo (at least one)
        mySettings = self.production.steps[myPos].token.settings
        if containsAndTrue(mySettings, 'alo'):
            if not nextPos in self.positions:
                self.positions.append(nextPos)

            if len(self.values)>myPos:
                self.values[myPos].append(state)
            else:
                self.values.append([state])
        else:
            self.values.append(state)
            self.positions[index] = nextPos

        #For the advanced ȣ
        self.__updatePosition(nextPos)

    def __isCompleted(self):
        if len(self.positions) == 1 and self.positions[0] == len(self.production.steps):
            return True
        return False

    def collapseFinished(self):
        assert self.containsCompleted()
        #self.positions = [len(self.production.steps)]
        self.__collapse(len(self.production.steps))
        assert self.__isCompleted()


    def __collapse(self, position):
        print(f'COLLAPSE position: {position}, len positions: {len(self.positions)}')
        assert position in self.positions

        #mySettings = self.production.steps[position].token.settings
        for pos in range(min(self.positions), position):
            # Add None for optionals
            if containsAndTrue(self.production.tokens[pos].settings, 'opt'):
                self.values.append(None)

        self.positions = [position]
        return 0

    def MatchThenAdvanceStateCopies(self, tok: Token):
        #print(f'MatchThenAdvanceStateCopies {str(tok)}')
        retStates = []
        #for index, pos in enumerate(self.positions):

        for index in range(len(self.positions)):
            if index >= len(self.positions):
                break
            
            pos = self.positions[index]
            print(f'MatchThenAdvanceStateCopies index: {index}, pos: {pos}, tok: {tok}, len pos: {len(self.positions)}')
            if pos < len(self.production.steps):
                print(f'MatchThenAdvanceStateCopies Match?')
                TernOrNonTerm = self.production.steps[pos]
                if isinstance(TernOrNonTerm, Terminal):
                    if TernOrNonTerm.match(tok):
                        print(f'MatchThenAdvanceStateCopies Matched')
                        newState = copy.deepcopy(self)
                        indexUpdated = newState.__collapse(pos)
                        newState.advance(tok, indexUpdated)
                        retStates.append(newState)

        #print(f'retStates: {str([str(stat) for stat in retStates])}')
        return retStates

    def containsCompleted(self):
        return len(self.production.steps) in self.positions

    def containsUncompleted(self):
        for pos in self.positions:
            if pos != len(self.production.steps):
                return True
        return False 

    def indexesOfNextInstance(self, NonTerminalName: str):
        #print(f'state {self.name()}: \tisinstance NonTerminalName: {NonTerminalName}, self.position: {self.positions}')
        ret = []
        for index, pos in enumerate(self.positions):
            if not (pos == len(self.production.steps)):
                next = self.production.steps[pos]
                if isinstance(next, NonTerminal) and next.name() == NonTerminalName:
                    ret.append(index)
        return ret

    def getNextNames(self) -> list[str]:
        ret = []
        for pos in self.positions:
            if pos < len(self.production.steps):
                ret.append(self.production.steps[pos].name())

        return ret

    def fullStr(self):
        def toStr(inVal):
            if isinstance(inVal, State):
                return inVal.fullStr()
            elif isinstance(inVal, list):
                return ''.join([toStr(i) for i in inVal])
            elif isinstance(inVal, str):
                return inVal
            elif inVal == None:
                # Optionals
                return ''
            else:
                assert False

        return f'{self.production.name} {{' + (', '.join(map(toStr, self.values))) + '}'

    def esrapSelf(self):
        req = lambda x: x.esrap() if isinstance(x, State) else x
        return ''.join(map(req, self.values))

    @staticmethod
    def __isStored(step, settings):
        return isinstance(step, NonTerminal) or containsAndTrueAny(settings, ['regex', 'opt', 'alo'])

    """
    NOTE:   This method works for both implicit and explicit compatibility 
            compat may have more non terminals
            That's why it is on the lhs
    """
    def __genIndexToIndices(self, compat, explicit):
        indexToIndex = {}
        #print(f'self production:       {self.production}')
        #print(f'compatible production: {compat}')

        count = 0
        for compatIndex, step in enumerate(compat.steps):
            settings = step.token.settings
            if State.__isStored(step, settings):
                selfIndex = step.token.settings['id'] if explicit else count
                if (not selfIndex in indexToIndex):
                    indexToIndex[selfIndex] = []
                    #print(f'empty dictionary array at : {selfIndex}')
                
                indexToIndex[selfIndex].append(compatIndex)
                count += 1
        return indexToIndex

    def esrap(self, productions: Productions):
        assert self.__isCompleted()
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
        print(f'indexToIndices: {indexToIndices}')

        #Replace all
        strs = [None]*len(stepsB)

        def optPad(val: str, settings):
            string = val
            if containsAndTrue(settings, 'pad'):
                string = ' ' + string + ' '
            return string

        def handleConversion(productions, settings, val) -> str:
            if isinstance(val, State):
                return val.esrap(productions)
            elif isinstance(val, Terminal):
                return optPad(val, settings)
            elif isinstance(val, NonTerminal):
                return optPad(val, settings)
            elif isinstance(val, str):
                return optPad(val, settings)
            elif isinstance(val, list):
                # DO NOT PAD
                return ''.join([handleConversion(productions, settings, v) for v in val])
            else:
                assert False

            

        # Set strings
        for i, step in enumerate(otherProd.steps):
            if (isinstance(step, Terminal) and not step.rule.settings['regex'] and not containsAndTrue(step.rule.settings, 'opt') and not containsAndTrue(step.rule.settings, 'alo')):
                string = step.rule.tok
                if containsAndTrue(step.rule.settings, 'pad'):
                    string = ' ' + string + ' '
                strs[i] = string
        # Set (Non)Terminals

        selfCountExludingStr = 0

        print(f'stepsA: {stepsA}, self.values: {self.values}')
        assert len(stepsA) == len(self.values)
        for i, step in enumerate(stepsA):
            typeNameStep = type(step).__name__
            val = self.values[i]
            typeNameVal = type(val).__name__
            print(f'{bcolors.OKGREEN}status: name {self.name()}, i: {i}, selfCountExludingStr: {selfCountExludingStr}, step: {step}, typeNameStep: {typeNameStep}, val: {val}, typeNameVal: {typeNameVal}{bcolors.ENDC}')

            if isinstance(val, str) and not containsAndTrueAny(step.token.settings, ['regex', 'opt']):
                assert not State.__isStored(step, step.token.settings)
                pass
            elif isinstance(val, State):
                compatIndices = indexToIndices[selfCountExludingStr]
                
                for compatIndex in compatIndices:
                    strs[compatIndex] = handleConversion(productions, None, val)
                selfCountExludingStr += 1

            #Check key    
            elif selfCountExludingStr in indexToIndices:

                assert(len(indexToIndices[selfCountExludingStr]) == 1)
                compatIndex = indexToIndices[selfCountExludingStr][0]
                selfCountExludingStr += 1

                if (isinstance(val, Terminal)):
                    isRegex = stepsA[i].rule.settings['regex']
                    isRegexB = stepsB[compatIndex].rule.settings['regex']
                    #print(f'isRegex: {isRegex}, isRegexB: {isRegexB}')
                    assert(isRegex == isRegexB)

                    strs[compatIndex] = handleConversion(productions, val.token.settings, val)
                    
                elif (isinstance(val, NonTerminal)):
                    rule = stepsB[compatIndex].rule
                    strs[compatIndex] = handleConversion(productions, rule.settings, rule.tok)
                elif (isinstance(val, str)):
                    settings = stepsB[compatIndex].token.settings
                    strs[compatIndex] = handleConversion(productions, settings, val)
                elif (isinstance(val, list)):
                    settings = stepsB[compatIndex].token.settings
                    strs[compatIndex] = handleConversion(productions, settings, val)
                elif val == None:
                    strs[compatIndex] = ''
                else:
                    typeName = type(val).__name__
                    print(f'ERROR: type {typeName}')
                    assert False
                """
            elif containsAndTrue(step.rule.settings, 'opt'):
                rule = stepsB[i].rule 
                strs[i] = optPad(val, rule.settings) if val != None else ''
            elif containsAndTrue(step.rule.settings, 'alo'):
                rule = stepsB[i].rule 
                strs[i] = ''.join([optPad(v, rule.settings) for v in val])
                """
            else:
                typeName = type(val).__name__
                #assert False
        
        #print(f'-------END ESRAP OF {self.name()}-------')
        print(strs)
        strs = [str if str != None else 'BOOO' for str in strs ]
        return ''.join(strs)


class Column():
    def __init__(self, productions, states):
        self.states = states
        self.productions = productions

    def containsState(self, name: str, index: int):
        for state in self.states:
            if len(state.positions) == 1:
                #We only have one index
                if (state.production.name == name and state.positions[0] == index):
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
    # complete non terminals
    #print(f'complete {state.production.name}, from: {state.originPosition}')
    colJ = table[state.originPosition].states
    newStates = []
    for stateJ in colJ:
        indexes = stateJ.indexesOfNextInstance(state.production.name)
        assert(len(indexes) in [0,1])
        if indexes != []:
            #print(f' completing: {stateJ.name()}')
            newState = copy.deepcopy(stateJ)
            newState.advance(state, indexes[0])
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
    #print(f'tokenized: \n\t{tokenStr}, len: {len(inTokens)}')
    #table = [Column(productions, [State(prod, 0) for prod in productions.productions])]
    topLevelName = productions.productions[0].name
    table = [Column(productions, []) for i in range(len(inTokens)+1)]

    if beginRules == None:
        table[0].extend([State(productions.productions[0], 0)])
    else:
        table[0].extend([State(productions.productions[i], 0) for i in range(len(productions.productions)) if productions.productions[i].uuid in beginRules])
    

    # Init 

    def predict(col, state, currentChart):
        #Prediction
        print('Predicting')
                    
        productionNames = state.getNextNames()
        for name in productionNames:
            col.extend(col.predict(name, currentChart))
    
    for currentChart, col in enumerate(table):
        #pre
        tok = inTokens[currentChart] if currentChart<len(inTokens) else None
        print(f'------{currentChart}, {tok}------')
        

        #real work
        for state in col.states:
            print(f'sate name: {state.production.name}, contains completed: {state.containsCompleted()}, uncompleted: {state.containsUncompleted()}')
            if (state.containsCompleted()):
                print('Is completed!')
                col.states.extend(complete(table, state))
            
            if (state.containsUncompleted()):

                if (tok != None and state.containsNextTerminal()):
                    print('Scanning')
                    newStates = state.MatchThenAdvanceStateCopies(tok)
                    table[currentChart+1].extend(newStates)

                #New NonTerminals may be found
                predict(col, state, currentChart)

        #post
        print('\n'.join(map(str, col.states)))
    
    # Find result
    matches = []
    for status in table[-1].states:
        if (status.originPosition == 0 and status.containsCompleted() and status.production.name == topLevelName):
            status.collapseFinished()
            matches.append(status)
    
    print(f'MATCHES: {matches}')
    if (len(matches) > 1):
        print(f'{bcolors.FAIL}ERROR: MULTIPLE MATCHES{bcolors.ENDC}')
        return None
    return matches[0] if len(matches) == 1 else None


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


