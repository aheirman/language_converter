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
    RULE                 = 5,
    MERGE                = 6,
    NAME_ID              = 7,
    END_OF_EXP           = 8

class RuleType(Enum):
    UNKNOWN       = -1,
    PRODUCTION    = 0,
    NOITCUDORP    = 1,
    MERGE         = 2,
    PAGE_SETTINGS = 3

class DSL_handle:
    def __init__(self, lines):
        self.lines = lines
        self.line_index = 0
        self.char_index = 0
        self.line = self.lines[self.line_index] if self.line_index < len(self.lines) else None

    def getRestOfLine(self):
        txt = self.line[self.char_index:]
        self.line_index += 1
        self.line = self.lines[self.line_index] if self.line_index < len(self.lines) else None
        self.char_index = 0
        return txt
    
    def getIndexCharAdvance(self):
        ret = (self.char_index, self.line[self.char_index])
        self.char_index += 1
        return ret
    
    def more_lines_left(self):
        return self.line_index < len(self.lines)

    def more_chars_in_line(self):
        return self.char_index < len(self.line)

    def end_line(self):
        self.line_index += 1
        self.line = self.lines[self.line_index] if self.line_index < len(self.lines) else None
        self.char_index = 0

    def continue_line_or_next(self, ruleType):
        ret = self.more_chars_in_line()
        if (not ret and __rule_is_multi_line__(ruleType)):
            if self.line_index + 1 < len(self.lines):
                ret = True
                self.end_line()
        return ret

    def __str__(self):
        return f'DSL_handle: {self.line_index}, {self.char_index}'

def __rule_is_multi_line__(rule):
    return rule in [RuleType.MERGE]

def parseIR2(input):

    ruleManagerA = Productiongenerator.createAllProductions([
            ('ESC-0-0', 'NL', '"\n"')], 'ESC')
    IRruleManager = parseIR("""{"id": "ir", "imports": ["ESC"]}
page         → settings{"opt": true} NL expression{"alo": true}
expression   → production | noitcudorp | merge
production   → name settings{"opt": true} "→"{"pad": false} multi_prod NL
noitcudorp   → name settings{"opt": true} "⇇"{"pad": false} token_regex NL
merge        → name ":" token_str "⇈" name ":" token_str "{" NL merge_ops{"alo": true, "opt": true} "}" NL
merge_ops    → merge_append
merge_append → "append_into" [0-9] "," [0-9] ";" NL{"opt": true}
multi_prod   → single_prod | single_prod "|"{"pad": true} multi_prod
single_prod  → bit{"alo": true, "pad": true}
bit          → name settings{"opt": true} | token_str settings{"opt": true} | token_regex
settings     → "{" token{"alo": true, "opt": true} "}"
name         → [a-zA-Z0-9]+
token        → [a-zA-Z0-9!<>#$,\\\\"\\\\'\\\\+\\\\-\\\\*_\\\\.\\\\[\\\\]!:→]+
token_str    → '"[a-zA-Z0-9,\-|(){}→]+"'{"regex": true, "quote": false}
token_regex  → '([\\\\[[a-zA-Z0-9!<>#$,\\\\"\\\\\\'\\\\-\\\\+\\\\*_\\\\.!:→\\\\\\\\]+\\\\][\\\\+\\\\*]?)+'{"regex": true, "quote": false} settings{"opt": true}
""".splitlines())
    
    projectManager = ProjectManager([ruleManagerA, IRruleManager])
    projectManager.processProductions()
    tokens = tokenize(input)
    print(f'tokens: {tokens}')
    matched = match(IRruleManager, tokens)
    print(f'matched: {type(matched)}')
    assert matched != None
    
    #move data from AST to data structure
    productions = []
    noitcudorps = []
    merges      = []

    page_settings_txt = matched.values[0].esrap(IRruleManager, IRruleManager)
    page_settings = page_settings = json.loads(page_settings_txt)

    for index, expression in enumerate(matched.values[2]):
        
        # prodorp is a either a production or a noitcudorp
        prodorp = expression.values[0]
        exp_name     = prodorp.values[0].values[0]
        

        start_line = index+1
        pageNumber = uuid.uuid4() if page_settings_txt == ''  else page_settings['id']
        gen = UUID_GEN(pageNumber, start_line)

        match prodorp.production.uuid:
            case 'ir-3-0':
                ruleType = RuleType.PRODUCTION
            case 'ir-4-0':
                ruleType = RuleType.NOITCUDORP
            case 'ir-5-0':
                ruleType = RuleType.MERGE
            case _:
                print(f'{bcolors.FAIL}ERROR: exression uuid {prodorp.uuid} not handeled!"{bcolors.ENDC}')
                assert False

        match ruleType:
            case RuleType.PRODUCTION:
                exp_settings = json.loads(prodorp.values[1].esrap(IRruleManager, IRruleManager)) if prodorp.values[1] != None else None
                multi_rule = prodorp.values[3]
                while multi_rule != None:
                    single_rule = multi_rule.values[0]
                    bits = single_rule.values[0]
                    rule = single_rule.esrap(IRruleManager, IRruleManager)
                    
                    tokensList = Productiongenerator.tokenize(rule)

                    print(f'{ruleType}: {exp_name}:{exp_settings} → {rule}')
                    for tokens in tokensList:
                        productions.append(Production(gen.next(), exp_name, tokens, exp_settings['compatible'] if exp_settings else None))
                    
                    multi_rule = multi_rule.values[2] if len(multi_rule.values)>1 else None
            case RuleType.NOITCUDORP:
                exp_settings = json.loads(prodorp.values[1].esrap(IRruleManager, IRruleManager)) if prodorp.values[1] != None else None
                token_regex = prodorp.values[3]
                rule = token_regex.esrap(IRruleManager, IRruleManager)
                tokensList = Productiongenerator.tokenize(rule)
                assert len(tokensList) == 1
                noitcudorps.append(Noitcudorp(gen.next(), exp_name, tokensList[0], exp_settings))
            case RuleType.MERGE:
                pass
            case RuleType.UNKNOWN|_:
                print(f'{bcolors.FAIL}ERROR: RuleType: {ruleType} not handeled!"{bcolors.ENDC}')
                assert False

            
        
        #print(f'{type(prod)}: {}')

    imports = [] if page_settings == None or (not 'imports' in page_settings) else page_settings['imports']
    name = f'generated-{random.randint(0,65536)}' if page_settings == None or (not 'id' in page_settings) else page_settings['id']
    return RuleManager(name, productions, noitcudorps, merges, imports)



def parseIR(lines):
    productions = []
    noitcudorps = []
    merges      = []

    page_settings_txt = ''
    page_settings = None

    handle = DSL_handle(lines)
    while handle.more_lines_left():
        exp_name         = ''
        exp_settings_txt = ''
        exp_id           = None

        merge_into_id    = None
        merge_subexp     = []

        rule = ''
        ruleType = RuleType.UNKNOWN
        state = ParseState.INITIAL
        #print(f'{lineNumber}, {state}, line: {line}')
        #print(handle)
        if len(handle.line) == 0 or handle.line[0] == '#':
            handle.end_line()
            continue

        start_line = handle.line_index

        while handle.continue_line_or_next(ruleType):
            index, c = handle.getIndexCharAdvance()
            #print(f'{lineNumber}, {state}, char: {c}')
            
            match state:
                case ParseState.INITIAL if c == '{':
                    #print(f'SETT')
                    page_settings_txt += c
                    state = ParseState.SETTINGS_BLOCK_PAGE
                    ruleType = RuleType.PAGE_SETTINGS
                case ParseState.SETTINGS_BLOCK_PAGE:
                    #print(f'SETTINGS')
                    page_settings_txt += c
                    if c == '}':
                        #print(f'closing settings block: {index}, {len(line)-1}')
                        assert index == len(handle.line)-1
                case ParseState.SETTINGS_BLOCK:
                    #print(f'SETTINGS')
                    exp_settings_txt += c
                    if c == '}':
                        #print(f'closing settings block: {index}, {len(line)-1}')
                        state = ParseState.SEP
                case ParseState.INITIAL | ParseState.NAME if not c in ['→', '⇇', '⇈']:
                    state = ParseState.NAME
                    #print(f'NAME')
                    if c.isalpha() or c in "_-":
                        exp_name += c 
                    elif c == ' ':
                        pass
                    elif c == '{':
                        exp_settings_txt += c
                        state = ParseState.SETTINGS_BLOCK
                    elif c == ':':
                        state = ParseState.NAME_ID
                        exp_id = ''
                    else:
                        print(f'ERROR: incorrect character found "{c}"')
                        assert False
                case ParseState.SEP | ParseState.NAME if (state == ParseState.SEP) or (c in ['→', '⇇', '⇈']):
                    #print(f'SEP')
                    if c == '→':
                        state = ParseState.RULE
                        ruleType = RuleType.PRODUCTION
                    elif c == '⇇':
                        state = ParseState.RULE
                        ruleType = RuleType.NOITCUDORP
                    elif c == '⇈':
                        state = ParseState.MERGE
                        ruleType = RuleType.MERGE
                    elif c == ' ':
                        pass
                    else:
                        print(f'ERROR: incorrect character found "{c}"')
                        assert False
                case ParseState.RULE:
                    #print(f'RULE')
                    rule += c
                case ParseState.NAME_ID:
                    if c == ' ':
                        state = ParseState.SEP
                    else:
                        exp_id += c
                case ParseState.MERGE:
                    if c == '}':
                        state = ParseState.END_OF_EXP
                    else:
                        exp_id += c
                case _:
                    print(f'ERROR: character not matched: "{c}", state: {state}')
                    assert False

        match ruleType:
            case RuleType.PAGE_SETTINGS:
                page_settings = json.loads(page_settings_txt)
            case RuleType.PRODUCTION | RuleType.NOITCUDORP:
                #print(f'parseIR line: {start_line}, {ruleType}, exp_name: {exp_name}, rule: \n\t"{rule}"')
                assert len(rule) != 0

                tokensList = Productiongenerator.tokenize(rule)
                print(exp_settings_txt)
                exp_settings = json.loads(exp_settings_txt) if exp_settings_txt != '' else None

                pageNumber = uuid.uuid4() if page_settings_txt == ''  else page_settings['id']
                gen = UUID_GEN(pageNumber, start_line)
                #print(f'gen: {pageNumber}, {start_line}')
                match ruleType:
                    case RuleType.PRODUCTION:
                        for tokens in tokensList:
                            productions.append(Production(gen.next(), exp_name, tokens, exp_settings['compatible'] if exp_settings else None))
                    case RuleType.NOITCUDORP:
                        assert len(tokensList) == 1
                        noitcudorps.append(Noitcudorp(gen.next(), exp_name, tokensList[0], exp_settings))
                    case _:
                        assert False
            case RuleType.MERGE:
                pass
            case _:
                print(f'{bcolors.FAIL}ERROR: rule type "{ruleType}" is unhandeled!, ParseState: {state}{bcolors.ENDC}')
                assert False

        handle.end_line()

    imports = [] if page_settings == None or (not 'imports' in page_settings) else page_settings['imports']
    name = f'generated-{random.randint(0,65536)}' if page_settings == None or (not 'id' in page_settings) else page_settings['id']
    return RuleManager(name, productions, noitcudorps, merges, imports)

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


