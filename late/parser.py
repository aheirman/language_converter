from late.helper import *

import copy
import json
import re
import uuid

from enum import Enum

class Token:
    def __init__(self, tok, settings):
        self.tok = tok
        self.settings = settings

    def __str__(self):
        return f'{{token: "{self.tok}", settings: {self.settings}}}' if self.settings != {} else f'{{token: {self.tok}}}'

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
            print(f'Terminal with regex rule: {self.token.tok}')
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
        #print(f'Production tokens: {[str(t) for t in self.tokens]}')
        for tok in self.tokens:
            #print(f'production process name: {self.name}, token: "{tok}"')
            
            if containsAndTrue(tok.settings, "regex") or containsAndTrue(tok.settings, "quote"):
                steps.append(Terminal(tok))
            else:
                prod = productions.productionWithNameExists(tok.tok)
                if prod == False:
                    print(f'{bcolors.FAIL}Production with name {tok.tok} missing!{bcolors.ENDC}')
                    assert False

                steps.append(NonTerminal(tok))
        self.steps = steps
    
    def __init__(self, uuid, name: str, tokens: list[Token], uuid_compat = None):
        self.name = name
        self.tokens = tokens
        self.uuid = uuid
        self.uuid_compat = uuid_compat

    def __str__(self):
        if hasattr(self, 'steps'):
            #tokensStr = ' '.join([str(elem) for elem in self.tokens])
            stepStr   = '\n\t' + '\n\t'.join([str(elem) for elem in self.steps])
            #return f'Production {{ name: {self.name}  tokens: [ {tokensStr} ], {stepStr} }}'
            uuidCompatStr = f', uuid_compat: {self.uuid_compat}' if self.uuid_compat != None else ''
            return f'Production {{ uuid: {self.uuid}, name: {self.name}, {stepStr}{uuidCompatStr}}}'
            #return f'Production {{ name: {self.name}, {stepStr}{uuidCompatStr}}}'
        else:
            tokensStr = ' '.join([str(elem) for elem in self.tokens])
            #return f'Production {{ uuid: {self.uuid}, name: {self.name}, {tokensStr} }}'
            return f'Production {{ name: {self.name}, {tokensStr} }}'

    def len(self):
        return len(self.steps)

class TokenizeSettings(Enum):
    PRE = 0
    NORMAL = 1
    QUOTE_DOUBLE = 2
    QUOTE_SINGLE = 8
    REGEX = 6
    REGEXPOSEND = 7
    VAL_FINISHED = 3
    SETTINGS_BLOCK = 4
    END = 5
    GROUPING = 9

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

        allowableEnds = [
                TokenizeSettings.NORMAL,
                TokenizeSettings.END,
                TokenizeSettings.VAL_FINISHED,
                TokenizeSettings.REGEXPOSEND]

        for index in range(len(input)):
            c = input[index]
            escaped = nextEscaped
            nextEscaped = False
            #print(f'currProduction: {currProduction}, escaped: {escaped}, status: {status}, tokens: ' + ' '.join(map(str, tokens[currProduction])) + ', adding char: ' + c)
            
            if ((status in allowableEnds) and (c == ' ' or c == '\n')):
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
                    status = TokenizeSettings.QUOTE_DOUBLE
                    settings['quote'] = True
                    settings['regex'] = False
                elif ( c == '\''):
                    status = TokenizeSettings.QUOTE_SINGLE
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
            elif (status == TokenizeSettings.QUOTE_DOUBLE and not escaped and c == '"'):
                status = TokenizeSettings.VAL_FINISHED
            elif (status == TokenizeSettings.QUOTE_SINGLE and not escaped and c == '\''):
                status = TokenizeSettings.VAL_FINISHED
            elif ((status in [TokenizeSettings.VAL_FINISHED, TokenizeSettings.NORMAL, TokenizeSettings.REGEXPOSEND]) and c == '{'):
                status = TokenizeSettings.SETTINGS_BLOCK
                strSettings += c
            elif (status == TokenizeSettings.SETTINGS_BLOCK and c == '}'):
                status = TokenizeSettings.END
                strSettings += c
            elif (status in [TokenizeSettings.NORMAL, TokenizeSettings.QUOTE_DOUBLE, TokenizeSettings.QUOTE_SINGLE, TokenizeSettings.REGEXPOSEND]):
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
    def UUIDgen(line: int) -> str:
        i = 0
        while True:
            yield f'{line}-{i}'
            i += 1
    """
        list of:
            0: list of used uuids
            1: name
            2: rule
            3: compatible with
    """
    @staticmethod
    def createAllProductions(list: list):
        prods = []
        for rule in list:
            prods.extend(Productiongenerator.__createProductions(*rule))
        return prods

    @staticmethod
    def createAllProductionsGenUUID(list: list):
        prods = []
        for index, rule in enumerate(list):
            uuids = self.__genUUID(index, rule[3])
            prods.extend(Productiongenerator.__createProductions(uuids, *rule))
        return prods

class Productions:

    def generateMap(self):
        self.RuleProdMap = dict(zip([prod.stdandard for prod in self.productions], [prod.stdandard for prod in self.productions]))

    def __init__(self, productions: list[Production]):
        self.productions = productions
        for prod in self.productions:
            prod.process(self)

        self.__checkForErrors()

    def __checkForErrors(self):
        uuids = []
        for prod in self.productions:
            if prod.uuid in uuids:
                assert False
            else:
                uuids.append(prod.uuid)

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
