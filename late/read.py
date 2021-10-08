from typing import Optional
import asyncio

from .late import *
from .project_manager import *

def __readFile(url: str) -> str:
    with open(url) as f:
         return f.readlines()

def __readFileLines(url: str) -> list[str]:
    with open(url) as f:
         return f.read().splitlines()


class UUID_GEN():
    def __init__(self, pageNumber, lineNumber):
        self.lineNumber = lineNumber
        self.index = 0
        self.pageNumber = pageNumber
    
    def __getitem__(self, item: int):
        return f'{self.pageNumber}-{self.lineNumber}-{item}'

    def next(self):
        ret = f'{self.pageNumber}-{self.lineNumber}-{self.index}'
        self.index += 1
        return ret

class ParseState(Enum):
    INITIAL              = 0,
    SETTINGS_BLOCK_PAGE  = 1,
    SETTINGS_BLOCK       = 2,
    NAME                 = 3,
    SEP                  = 4,
    RULE                 = 5

class RuleType(Enum):
    UNKNOWN = -1,
    PRODUCTION = 0,
    NOITCUDORP = 1

def parseIR(lines):
    productions = []
    noitcudorps = []

    page_settings_txt = ''
    page_settings = None
    for lineNumber, line in enumerate(lines):
        name = ''
        rule_settings_txt = ''
        rule = ''
        ruleType = RuleType.UNKNOWN
        state = ParseState.INITIAL
        print(f'{lineNumber}, {state}, line: {line}')
        isPageSettings = False
        if len(line) == 0 or line[0] == '#':
            continue

        for index, c in enumerate(line):
            #print(f'{lineNumber}, {state}, char: {c}')
            match state:
                case ParseState.INITIAL if c == '{':
                    #print(f'SETT')
                    page_settings_txt += c
                    state = ParseState.SETTINGS_BLOCK_PAGE
                    isPageSettings = True
                case ParseState.SETTINGS_BLOCK_PAGE:
                    #print(f'SETTINGS')
                    page_settings_txt += c
                    if c == '}':
                        #print(f'closing settings block: {index}, {len(line)-1}')
                        assert index == len(line)-1
                case ParseState.SETTINGS_BLOCK:
                    #print(f'SETTINGS')
                    rule_settings_txt += c
                    if c == '}':
                        #print(f'closing settings block: {index}, {len(line)-1}')
                        state = ParseState.SEP
                case ParseState.INITIAL | ParseState.NAME if not c in ['→', '⇇']:
                    state = ParseState.NAME
                    #print(f'NAME')
                    if c.isalpha() or c in "_-":
                        name += c 
                    elif c == ' ':
                        pass
                    elif c == '→':
                        state = ParseState.RULE
                    elif c == '{':
                        rule_settings_txt += c
                        state = ParseState.SETTINGS_BLOCK
                    else:
                        assert False
                case ParseState.SEP | ParseState.NAME if (state == ParseState.SEP) or (c in ['→', '⇇']):
                    #print(f'SEP')
                    if c == '→':
                        state = ParseState.RULE
                        ruleType = RuleType.PRODUCTION
                    elif c == '⇇':
                        state = ParseState.RULE
                        ruleType = RuleType.NOITCUDORP
                    else:
                        pass
                case ParseState.RULE:
                    #print(f'RULE')
                    rule += c
                case _:
                    print(f'ERROR: character not matched: {c}, state: {state}')
                    assert False

        if isPageSettings:
            page_settings = json.loads(page_settings_txt)
        else:
            print(f'parseIR line: {lineNumber}, {ruleType}, name: {name}, rule: \n\t"{rule}"')
            assert len(rule) != 0

            tokensList = Productiongenerator.tokenize(rule)
            print(rule_settings_txt)
            rule_settings = json.loads(rule_settings_txt) if rule_settings_txt != '' else None

            pageNumber = uuid.uuid4() if page_settings_txt == ''  else page_settings['id']
            gen = UUID_GEN(pageNumber, lineNumber)
            match ruleType:
                case RuleType.PRODUCTION:
                    for tokens in tokensList:
                        productions.append(Production(gen.next(), name, tokens, rule_settings['compatible'] if rule_settings else None))
                case RuleType.NOITCUDORP:
                    assert len(tokensList) == 1
                    noitcudorps.append(Noitcudorp(gen.next(), name, tokensList[0], rule_settings))
                case _:
                    assert False

    imports = [] if page_settings == None or (not 'imports' in page_settings) else page_settings['imports']
    name = f'generated-{random.randint(0,65536)}' if page_settings == None or (not 'id' in page_settings) else page_settings['id']
    return RuleManager(name, productions, noitcudorps, imports)

def __parseSrc(lines: list[str], rManager: Optional[RuleManager] = None):
    if productions == None:
        uuids = [uuid.uuid4() for i in range(10)]
        prodA = Productiongenerator.createAllProductions([
            ([uuids[0]], 'calculation', 'term'),
            ([uuids[1]], 'term', 'number "+" term'),
            ([uuids[2]], 'term', 'number'),
            ([uuids[3]], 'number', '[0-9]')])
        matched = match(Productions(prodA), tokenize(lines))
        
        vals = matched.fullStr()
        return vals
        #esr = matched.esrap(Productions(prodA))
        #self.assertEqual(esr, outputExpect)
    #return 

def __parseFile(lines: list[str], rManager: Optional[RuleManager] = None):
    if productions == None:
        uuids = [uuid.uuid4() for i in range(10)]
        prodA = Productiongenerator.createAllProductions([
            ([uuids[0]], 'calculation', 'term'),
            ([uuids[1]], 'term', 'number "+" term'),
            ([uuids[2]], 'term', 'number'),
            ([uuids[3]], 'number', '[0-9]')])
        matched = match(Productions(prodA), tokenize(lines))
        
        vals = matched.fullStr()
        return vals
        #esr = matched.esrap(Productions(prodA))
        #self.assertEqual(esr, outputExpect)
    #return 


def getMetaIrProductions(url: str) -> list[Production]:
    ruleManager = parseIR(__readFileLines(url))
    return ruleManager
    
def parse(url_grammer2: str, url_grammer: str, tokens: list) -> str:
    prods = __parseIR(__readFileLines(url_grammer2))
    rManager = Productions(prods)
    #print(rManager)
    return match(rManager, tokens)
    #productions_lang = __parseFile(__readFile(url_grammer), rManager)
    #match(productions_lang, tokens)


