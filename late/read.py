
from .late import *
from typing import Optional

import asyncio

def __readFile(url: str) -> str:
    with open(url) as f:
         return f.readlines()

def __readFileLines(url: str) -> list[str]:
    with open(url) as f:
         return f.readlines()


class UUID_GEN():
    def __init__(self, lineNumber):
        self.lineNumber = lineNumber
    
    def __getitem__(self, item: int):
        return f'{self.lineNumber}-{item}'

class ParseState(Enum):
    NAME = 1,
    SEP = 2,
    RULE = 3

def __parseIR(lines):
    productions = []

    for lineNumber, line in enumerate(lines):
        name = ''
        rule = ''
        sep = ''
        state = ParseState.NAME

        for c in line:
            if state == ParseState.NAME:
                if not c == ' ':
                    name += c
                else:
                    state == ParseState.SEP
            elif state == ParseState.SEP:
                if len(sep) == 0 and c == '-': 
                    sep += c
                elif len(sep) == 1 and c == '>': 
                    sep += c
                elif len(sep) == 2 and c == ' ': 
                    state = ParseState.RULE
                else:
                    assert False
            elif state == ParseState.RULE:
                rule += c

        productions.append(Production(UUID_GEN(lineNumber), name, rule))
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

def parse(url_grammer2: str, url_grammer: str, input: str) -> str:
    productions = __parseEBNF(__readFileLines(url_grammer2))
    #print(productions)
    #productions_lang = __parseFile(__readFile(url_grammer), productions)
    #match(productions_lang, tokenize(input))


