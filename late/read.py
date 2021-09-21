
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
    def __init__(self, lineNumber):
        self.lineNumber = lineNumber
        self.index = 0
    
    def __getitem__(self, item: int):
        return f'{self.lineNumber}-{item}'

    def next(self):
        ret = f'{self.lineNumber}-{self.index}'
        self.index += 1
        return ret

class ParseState(Enum):
    NAME = 1,
    SEP = 2,
    RULE = 3

def __parseIR(lines):
    productions = []

    for lineNumber, line in enumerate(lines):
        name = ''
        rule = ''
        state = ParseState.NAME
        
        if line[0] == '#':
            continue

        for c in line:
            if state == ParseState.NAME:
                if not (c == ' ' or c == '→'):
                    name += c 
                elif c == ' ':
                    pass
                elif c == '→':
                    state = ParseState.RULE
            elif state == ParseState.SEP:
                if c == ' ':
                    pass
                elif c == '→':
                    state = ParseState.RULE
            elif state == ParseState.RULE:
                rule += c

        assert len(rule) != 0

        print(f'__parseIR line: {lineNumber}, name: {name}, \n\t rule: "{rule}"')
        tokensList = Productiongenerator.tokenize(rule)
        gen = UUID_GEN(lineNumber)
        for tokens in tokensList:
            productions.append(Production(gen.next(), name, tokens))
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


def getMetaIrProductions(url: str) -> Productions:
    prods = __parseIR(__readFileLines(url))
    return Productions(prods)
    
def parse(url_grammer2: str, url_grammer: str, tokens: list) -> str:
    prods = __parseIR(__readFileLines(url_grammer2))
    productions = Productions(prods)
    #print(productions)
    return match(productions, tokens)
    #productions_lang = __parseFile(__readFile(url_grammer), productions)
    #match(productions_lang, tokens)


