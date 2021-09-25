
from .late import *
from typing import Optional

import asyncio

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

def parseIR(lines):
    productions = []

    page_settings_txt = ''
    for lineNumber, line in enumerate(lines):
        name = ''
        production_settings_txt = ''
        rule = ''
        state = ParseState.INITIAL
        #print(f'{lineNumber}, {state}, line: {line}')
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
                    production_settings_txt += c
                    if c == '}':
                        #print(f'closing settings block: {index}, {len(line)-1}')
                        state = ParseState.SEP
                case ParseState.INITIAL | ParseState.NAME:
                    state = ParseState.NAME
                    #print(f'NAME')
                    if c.isalpha() or c in "_-":
                        name += c 
                    elif c == ' ':
                        pass
                    elif c == '→':
                        state = ParseState.RULE
                    elif c == '{':
                        production_settings_txt += c
                        state = ParseState.SETTINGS_BLOCK
                    else:
                        assert False
                case ParseState.SEP:
                    #print(f'SEP')
                    if c == ' ':
                        pass
                    elif c == '→':
                        state = ParseState.RULE
                case ParseState.RULE:
                    #print(f'RULE')
                    rule += c

        if isPageSettings:
            page_settings = json.loads(page_settings_txt)
        else:
            #print(f'parseIR line: {lineNumber}, name: {name}, rule: \n\t"{rule}"')
            assert len(rule) != 0

            tokensList = Productiongenerator.tokenize(rule)
            #print(production_settings_txt)
            production_settings = json.loads(production_settings_txt) if production_settings_txt != '' else None

            pageNumber = uuid.uuid4() if page_settings_txt == ''  else page_settings['id']
            gen = UUID_GEN(pageNumber, lineNumber)
            for tokens in tokensList:
                productions.append(Production(gen.next(), name, tokens, production_settings['compatible'] if production_settings else None))
    return productions

def __parseSrc(lines: list[str], productions: Optional[Productions] = None):
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

def __parseFile(lines: list[str], productions: Optional[Productions] = None):
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
    prods = parseIR(__readFileLines(url))
    return Productions(prods)
    
def parse(url_grammer2: str, url_grammer: str, tokens: list) -> str:
    prods = __parseIR(__readFileLines(url_grammer2))
    productions = Productions(prods)
    #print(productions)
    return match(productions, tokens)
    #productions_lang = __parseFile(__readFile(url_grammer), productions)
    #match(productions_lang, tokens)


