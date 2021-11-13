from late.helper import *

import copy
import json
import re
import uuid
import rstr
import random

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

class Noitcudorp:
    def process(self):
        pass
    
    def generate(self):
        return rstr.xeger(self.tokens[0].tok)

    def __init__(self, uuid, name: str, tokens: list[Token], settings):
        self.name = name
        self.uuid = uuid
        self.settings = settings
        if not len(tokens) == 1:
            print(f'ERROR: len(tokens): {len(tokens)}, tokens: {str([str(tok) for tok in tokens])}')
            assert False 
        self.tokens = tokens


    def __str__(self):
        tokensStr = ' '.join([str(elem) for elem in self.tokens])
        return f'Noitcudorp {{ uuid: {self.uuid}, name: {self.name}, tokens: {tokensStr}}}'

def is_trivial_step(step):
    return isinstance(step, Terminal) and not step.rule.settings['regex'] and not containsAndTrue(step.rule.settings, 'opt') and not containsAndTrue(step.rule.settings, 'alo')

def is_tok_trivial(is_terminal, rule_settings):
    return is_terminal and not (rule_settings['regex'] or containsAndTrue(rule_settings, 'opt') or containsAndTrue(rule_settings, 'opt'))

class Production:
    def process(self, ruleManager):
        input_steps = []
        #print(f'Production tokens: {[str(t) for t in self.tokens]}')
        noitcudorps = []
        inputstep_to_compat_index = {}
        contains_id = None

        def check(is_trivial, _settings):
            nonlocal contains_id
            if not is_trivial:
                #print(f'nontrivial & pre: _contains_id: {contains_id}, _settings: {_settings}')
                if contains_id == None:
                    contains_id = "id" in _settings
                else:
                    if contains_id != ("id" in _settings):
                        error_txt = "ERROR: if any non trivial step in a production has an ID all of them need one!"
                        print(f'{bcolors.FAIL}{error_txt}{bcolors.ENDC}')
                        raise RuntimeError(error_txt)


        input_index  = 0
        compat_index = 0
        for step_index, tok in enumerate(self.tokens):
            #print(f'production process name: {self.name}, token: "{tok.tok}", {tok.settings}')
            is_terminal = containsAndTrue(tok.settings, "regex") or containsAndTrue(tok.settings, "quote")
            trivial = is_tok_trivial(is_terminal, tok.settings)
            check(trivial, tok.settings)

            if is_terminal:
                input_steps.append(Terminal(tok))
                if not trivial:
                    input_index  += 1
            elif not containsAndTrue(tok.settings, "convert_only"):
                exists = ruleManager.productionWithNameExists(tok.tok)
                if exists:
                    input_steps.append(NonTerminal(tok))
                    inputstep_to_compat_index[step_index] = compat_index
                    input_index  += 1
                    compat_index += 1
                else:
                    print(f'{bcolors.FAIL}Production with name "{tok.tok}" missing!{bcolors.ENDC}')
                    assert False
            else:
                exists = ruleManager.noitcudorpWithNameExists(tok.tok)
                if exists:
                    noitcudorps.append(NonTerminal(tok))
                    compat_index += 1
                else:
                    print(f'{bcolors.FAIL}Noitcudorp with name "{tok.tok}" missing!{bcolors.ENDC}')
                    assert False

        self.noitcudorps = noitcudorps   
        self.input_steps = input_steps
        self.inputstep_to_compat_index = inputstep_to_compat_index
        #self.output_steps = 
    
    def __init__(self, uuid, name: str, tokens: list[Token], uuid_compat = None):
        self.name = name
        self.tokens = tokens
        self.uuid = uuid
        self.uuid_compat = uuid_compat
        self.noitcudorps = []

    def __str__(self):
        noitcudorpsStr = '\n\t\t' + '\n\t\t'.join([str(elem) for elem in self.noitcudorps])
        if hasattr(self, 'steps'):
            #tokensStr = ' '.join([str(elem) for elem in self.tokens])
            stepStr   = '\n\t' + '\n\t'.join([str(elem) for elem in self.input_steps])
            #return f'Production {{ name: {self.name}  tokens: [ {tokensStr} ], {stepStr} }}'
            uuidCompatStr = f', uuid_compat: {self.uuid_compat}' if self.uuid_compat != None else ''
            return f'Production {{ uuid: {self.uuid}, name: {self.name}, {stepStr}{uuidCompatStr}, \n\tnoitcudorps:{noitcudorpsStr}}}'
            #return f'Production {{ name: {self.name}, {stepStr}{uuidCompatStr}}}'
        else:
            tokensStr = ' '.join([str(elem) for elem in self.tokens])
            #return f'Production {{ uuid: {self.uuid}, name: {self.name}, {tokensStr} }}'
            return f'Production {{ name: {self.name}, \n\t{tokensStr}, \n\tnoitcudorps:{noitcudorpsStr}}}'

    def len(self):
        return len(self.input_steps)

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
            elif (status == TokenizeSettings.SETTINGS_BLOCK):
                if c == '}':
                    status = TokenizeSettings.END
                    strSettings += c
                else:
                    strSettings += c
            elif (status in [TokenizeSettings.NORMAL, TokenizeSettings.QUOTE_DOUBLE, TokenizeSettings.QUOTE_SINGLE, TokenizeSettings.REGEXPOSEND]):
                if (c == '\\' and not escaped):
                    nextEscaped = True
                else:
                    curr += c
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
    def createAllProductions(list: list, name: str = f'generated-{random.randint(0,65536)}'):
        prods = []
        for rule in list:
            prods.extend(Productiongenerator.__createProductions(*rule))
        manager = RuleManager(name, prods, [], [], [])
        manager.process(([], []))
        return manager

class Merge:
    def __init__(self, from_id, into_id, steps: list):
        self.from_id = from_id
        self.into_id = into_id
        self.steps   = steps

    def __str__(self):
        return f'{{Merge: {self.from_id} into {self.into_id} }}'

class RuleManager:

    def generateMap(self):
        self.RuleProdMap = dict(zip([prod.stdandard for prod in self.my_productions], [prod.stdandard for prod in self.my_productions]))

    def __init__(self, name, productions: list[Production], noitcudorps: list[Noitcudorp], merges: list[Merge], imports: list[str]):
        self.name        = name
        self.productions = productions
        self.noitcudorps = noitcudorps
        self.merges      = merges
        self.imports     = imports
        self.__checkForErrors()

    def process(self, importsState: [list, list]):
        imported_productions = importsState[0]
        imported_noitcudorp  = importsState[1]

        print('------------')
        print([str(p) for p in imported_productions])
        print([str(p) for p in imported_noitcudorp])

        #TODO: handle delete statements!
        
        self.productions += imported_productions
        self.noitcudorps += imported_noitcudorp
        
        for prod in self.productions:
            prod.process(self)

    def __checkForErrors(self):
        uuids = []
        for prod in self.productions:
            if prod.uuid in uuids:
                print(f'{bcolors.FAIL}ERROR: trying to add production with uuid: "{prod.uuid}" is already in uuids: {uuids}{bcolors.ENDC}')
                assert False
            else:
                uuids.append(prod.uuid)

    def __str__(self):
        #return '\n'.join([str(elem) for elem in (self.my_productions + self.noitcudorps + ['--------', 'IMPORTED', '--------'] + self.imported_productions + self.imported_noitcudorp )])
        return 'RuleManager:\n' + '\t\n'.join([str(elem) for elem in (self.productions + self.merges + self.noitcudorps)])

    def getProduction(self, uuid):
        for prod in self.productions:
            if (prod.uuid == uuid):
                return prod
        return None

    def getNoitcudorp(self, name: str):
        for noitcudorp in self.noitcudorps:
            if (noitcudorp.name == name):
                return noitcudorp
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

    def noitcudorpWithNameExists(self, name: str):
        for noitcudorp in self.noitcudorps:
            if (noitcudorp.name == name):
                return True
        return False

    def getEquivalentProduction(self, standardese) -> Production:
        pass
