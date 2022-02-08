from typing import Optional
import asyncio

from eq.early.late import *
from eq.read_hand import *
from eq.project_manager import *
from eq.shared.tokenize import *

def __readFile(url: str) -> str:
    with open(url) as f:
         return f.read()

def __readFileLines(url: str) -> list[str]:
    with open(url) as f:
         return f.read().splitlines()

ruleManagerA = Productiongenerator.createAllProductions([
        ('ESC-0-0', 'NL', '"\n"')], 'ESC')
IRruleManager = parseIR_handwritten("""{"id": "ir", "imports": ["ESC"]}
page         → settings{"opt": true} NL{"alo": true, "opt": true} expression{"alo": true}
expression   → production | noitcudorp | merge
production   → name settings{"opt": true} "→"{"pad": false} multi_prod NL{"alo": true}
noitcudorp   → name settings{"opt": true} "⇇"{"pad": false} token_regex NL{"alo": true}
merge        → name ":" token_str "⇈" name ":" token_str "{" NL merge_ops{"alo": true, "opt": true} "}" NL
merge_ops    → merge_append
merge_append → "append_into" [0-9] "," [0-9] ";" NL{"opt": true}
multi_prod   → single_prod | single_prod "|"{"pad": true} multi_prod
single_prod  → bit{"alo": true, "pad": true}
bit          → name settings{"opt": true} | token_str settings{"opt": true} | token_regex settings{"opt": true} | "(" multi_prod ")" settings{"opt": true}
settings     → "{" token{"alo": true, "opt": true} "}"
name         → [a-zA-Z0-9]+
token        → [a-zA-Z0-9!<>#$,\\\\"\\\\'\\\\+\\\\-\\\\*_\\\\.\\\\[\\\\]!:→]+
token_str    → '"[a-zA-Z0-9,\+\-|(){}→\\\\"\\'<>#&%!=\\\\*~]+"'{"regex": true, "quote": false}
token_regex  → '\\\\[[a-zA-Z0-9!<>#$,\\\\"\\\\\\'\\\\-\\\\+\\\\*_\\\\.!:→\\\\\\\\ /]+\\\\]([\\\\+\\\\*])?'{"regex": true, "quote": false}
""".splitlines())
    
projectManager = ProjectManager([ruleManagerA, IRruleManager])
projectManager.processProductions()


class Parser():

    def __init__(self):
        self.productions = []
        self.noitcudorps = []
        self.merges      = []

    def __gen_prod(self, exp_name, exp_settings, multi_prod, real_uuid = False):
        while multi_prod != None:
            single_prod = multi_prod.values[0]
            
            
            bits = single_prod.values[0]

            # rule = single_prod.esrap(IRruleManager, IRruleManager)
            tokenList = [] #Productiongenerator.tokenize(rule)
            for bit in bits:
                match bit.production.uuid:
                    case 'ir-10-0':
                        #name
                        bit_txt = bit.values[0].values[0]
                        bit_settings = {}

                        if bit.values[1] != None:
                            bit_settings.update(json.loads(bit.values[1].esrap(IRruleManager, IRruleManager)))

                        tokenList.append(Token(bit_txt, bit_settings))
                    case 'ir-10-1':
                        #token string
                        bit_txt = bit.values[0].values[0]
                        bit_txt = bit_txt[1:-1]
                        bit_settings = {}

                        bit_settings['quote'] = True
                        bit_settings['regex'] = False

                        if bit.values[1] != None:
                            unwrapped = bit.values[1].esrap(IRruleManager, IRruleManager)
                            #print(f'__gen_prod: {unwrapped}')
                            bit_settings.update(json.loads(unwrapped))

                        tokenList.append(Token(bit_txt, bit_settings))
                    case 'ir-10-2':
                        #token regex
                        bit_txt = bit.values[0].values[0]
                        bit_settings = {}

                        #bit_settings['quote'] = False
                        bit_settings['regex'] = True

                        if bit.values[1] != None:
                            bit_settings.update(json.loads(bit.values[1].esrap(IRruleManager, IRruleManager)))

                        tokenList.append(Token(bit_txt, bit_settings))
                    case 'ir-10-3':
                        bracket_exp_settings = json.loads(bit.values[3].esrap(IRruleManager, IRruleManager)) if bit.values[3] != None else {}
                        bracket_multi_prod = bit.values[1]
                        bracket_name = str(uuid.uuid4())
                        self.__gen_prod(bracket_name, None, bracket_multi_prod, True)
                        tokenList.append(Token(bracket_name, bracket_exp_settings))
                    case _:
                        print(f'{bcolors.FAIL}ERROR: bit uuid {bit.production.uuid} not handeled!"{bcolors.ENDC}')
                        assert False
            
            
            
            id = str(uuid.uuid4()) if real_uuid else self.gen.next()
            comp = containsNotNoneAndPresent(exp_settings, 'compatible')
            #print(f'{RuleType.PRODUCTION}: {id}:{exp_name} sett: {exp_settings} → {str([str(tok) for tok in tokenList])}')
            self.productions.append(Production(id, exp_name, tokenList, comp))
            #---
            multi_prod = multi_prod.values[2] if len(multi_prod.values)>1 else None


    def parseIR(self, input : str):
        tokens = tokenize(input)
        #print(f'tokens: {tokens}')
        matched = match(IRruleManager, tokens)
        if matched == None:
            print(f"{bcolors.FAIL}ERROR: THE LATE SOURCE FILE IS NON COMPLIANT{bcolors.ENDC}")
            print(f'ERROR: input: {input}')
            assert False

        #print(f'matched: {type(matched)}, {matched.production.uuid}')
        assert matched != None
        
        #move data from AST to data structure


        settings_block = matched.values[0]
        page_settings_txt = None
        page_settings     = None
        if settings_block != None:
            page_settings_txt = matched.values[0].esrap(IRruleManager, IRruleManager)
            page_settings = page_settings = json.loads(page_settings_txt)

        for index, expression in enumerate(matched.values[2]):
            
            # prodorp is a either a production or a noitcudorp
            prodorp  = expression.values[0]
            exp_name = prodorp.values[0].values[0]
            

            start_line = index+1
            pageNumber = "PIPE" if page_settings == None  else page_settings['id']
            #uuid.uuid4() if page_settings == None  else page_settings['id']
            self.gen = UUID_GEN(pageNumber, start_line)

            match prodorp.production.uuid:
                case 'ir-3-0':
                    ruleType = RuleType.PRODUCTION
                case 'ir-4-0':
                    ruleType = RuleType.NOITCUDORP
                case 'ir-5-0':
                    ruleType = RuleType.MERGE
                case _:
                    print(f'{bcolors.FAIL}ERROR: exression uuid {prodorp.production.uuid} not handeled!"{bcolors.ENDC}')
                    assert False

            match ruleType:
                case RuleType.PRODUCTION:
                    exp_settings = json.loads(prodorp.values[1].esrap(IRruleManager, IRruleManager)) if prodorp.values[1] != None else None
                    multi_prod = prodorp.values[3]
                    self.__gen_prod(exp_name, exp_settings, multi_prod)
                case RuleType.NOITCUDORP:
                    exp_settings = json.loads(prodorp.values[1].esrap(IRruleManager, IRruleManager)) if prodorp.values[1] != None else None
                    token_regex = prodorp.values[3]
                    rule = token_regex.esrap(IRruleManager, IRruleManager)
                    tokensList = Productiongenerator.tokenize(rule)
                    assert len(tokensList) == 1
                    self.noitcudorps.append(Noitcudorp(self.gen.next(), exp_name, tokensList[0], exp_settings))
                case RuleType.MERGE:
                    pass
                case RuleType.UNKNOWN|_:
                    print(f'{bcolors.FAIL}ERROR: RuleType: {ruleType} not handeled!"{bcolors.ENDC}')
                    assert False

                
            
            #print(f'{type(prod)}: {}')

        imports = [] if page_settings == None or (not 'imports' in page_settings) else page_settings['imports']
        #print(f'imports: {imports}')
        name = f'generated-{random.randint(0,65536)}' if page_settings == None or (not 'id' in page_settings) else page_settings['id']
        return RuleManager(name, self.productions, self.noitcudorps, self.merges, imports)


def parseIR(input : str):
    parsy = Parser()
    return parsy.parseIR(input)

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
    ruleManager = parseIR(__readFile(url))
    projectManager = ProjectManager([ruleManager, ruleManager])
    projectManager.processProductions()
    return ruleManager
    
def parse_file_to_file(url_grammer_src: str, url_grammer_dest: str, tokens: list) -> str:
    prods = __parseIR(__readFileLines(url_grammer_src))
    rManager = Productions(prods)
    #print(rManager)
    return match(rManager, tokens)
    #productions_lang = __parseFile(__readFile(url_grammer), rManager)
    #match(productions_lang, tokens)


