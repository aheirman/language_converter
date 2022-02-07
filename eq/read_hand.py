import enum

from .early.late import *

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

def parseIR_handwritten(lines):
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