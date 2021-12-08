from __future__ import annotations

import copy
import uuid
import itertools
import graphviz
import html


from late.helper import *
from late.expression import *

class UnwrapMethod(Enum):
    ESRAP = 1,
    DOT = 2

class Compatibility(Enum):
    EQUAL     = 0,
    OTHER_EXPLICIT  = 1,
    SELF_EXPORT  = 2

class State:
    def __init__(self, production, originPosition):
        self.production = production
        self.originPosition = originPosition
        self.values = []
        self.position = 0
        self.uuid = uuid.uuid4()
    
    def __skipToPosPad(self, pos):
        assert self.position <= pos
        while self.position < pos:
            self.values.append(None)
            self.position += 1

    #NOTE: The position shift of us occurs after this is run
    def createNewStates(self) -> List[State]:
        retStates = []
        
        myCurrentPos = self.position
        mySettings = self.production.input_steps[myCurrentPos].token.settings

        if containsAndTrue(mySettings, 'alo'):
                newState = copy.deepcopy(self)
                newState.position = myCurrentPos
                retStates.append(newState)

        pos = myCurrentPos + 1
        while not pos == len(self.production.input_steps):
            set = self.production.input_steps[pos].token.settings
            if containsAndTrue(set, 'opt'):
                newState = copy.deepcopy(self)
                newState.position += 1 #Passed my state
                newState.__skipToPosPad(pos+1)
                retStates.append(newState)
            else:
                break
            pos += 1
        return retStates

    """
        create extra states needed for optionals
    """
    def createInitial(self) -> List[State]:
        retStates = []
        
        myCurrentPos = self.position
        mySettings = self.production.input_steps[myCurrentPos].token.settings
        
        pos = myCurrentPos # NO plus 1
        while not pos == len(self.production.input_steps):
            set = self.production.input_steps[pos].token.settings
            if containsAndTrue(set, 'opt'):
                newState = copy.deepcopy(self)
                newState.position += 1 #Passed my state
                newState.__skipToPosPad(pos+1)
                retStates.append(newState)
            else:
                break
            pos += 1
        return retStates

    def __str__(self):
        str = f'{self.production.name.ljust(15)} → {{'
        for index, step in enumerate(self.production.input_steps):
            if (index == self.position):
                str += ' ȣ '
            
            str += step.name() + ' '
        if (len(self.production.input_steps) == self.position):
            str += ' ȣ '

        str += '},'
        str = str.ljust(40)
        str += f'from {self.originPosition}'
        return str
    
    def name(self) -> str:
        return self.production.name

    def nextIsTerminal(self):
        #print(f'name: {self.name()}, pos: {self.position}, Terminal: {term}')
        if self.position < len(self.production.input_steps):
            if isinstance(self.production.input_steps[self.position], Terminal):
                return True
        return False

    """
    def match(self, input):
        assert self.containsNextTerminal()
        
        for pos in self.positions:
            if self.production.input_steps[pos].match(input):
                return True

        return False
     """

    def __setValue2(self, val):
        settings = self.production.input_steps[self.position].token.settings
        if containsAndTrue(settings, 'alo'):
            self.values.append([val])
        else:
            self.values.append(val)

    def __setValue(self, val):
        if self.position ==  len(self.values):
            self.__setValue2(val)
            
        elif self.position ==  len(self.values)-1:
            self.values[self.position].append(val)
        elif self.position >  len(self.values):
            while len(self.values) < self.position:
                self.values.append(None)
            self.__setValue2(val)
        else:
            assert False

    def advance(self, val) -> list[State]:
        #print(f'advance self.position: {self.position}')
        assert self.position < self.production.len()
        myPos = self.position

        #If the current position is
        self.__setValue(val)

        retStates = self.createNewStates()
        self.position = myPos + 1

        return retStates


    def MatchThenAdvanceStateCopies(self, tok: Token):
        #print(f'MatchThenAdvanceStateCopies {str(tok)}')
        retStates = []
        #for index, pos in enumerate(self.positions):

        pos = self.position
        #print(f'MatchThenAdvanceStateCopies index: pos: {pos}, tok: {tok}')
        if pos < self.production.len():
            #print(f'MatchThenAdvanceStateCopies Match?')
            TernOrNonTerm = self.production.input_steps[pos]
            if isinstance(TernOrNonTerm, Terminal):
                if TernOrNonTerm.match(tok):
                    #print(f'MatchThenAdvanceStateCopies Matched')
                    newState = copy.deepcopy(self)
                    retStates.extend(newState.advance(tok))
                    retStates.append(newState)

        #print(f'retStates: {str([str(stat) for stat in retStates])}')
        return retStates

    def isCompleted(self):
        return self.production.len() == self.position

    def getNextName(self) -> str:
        pos = self.position
        if pos < self.production.len():
            return self.production.input_steps[pos].name()
        else:
            assert False

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


    @staticmethod
    def __isStored(step, settings):
        return isinstance(step, NonTerminal) or containsAndTrueAny(settings, ['regex', 'opt', 'alo'])



    """
    NOTE:   This method works for equal, implicit and explicit compatibility 
            prodB may have more non terminals
            That's why it is on the lhs of the assignment in valIndexAToStepIndicesB
            This is problematic for implicit compatibility...
            TODO: BROKEN FOR ADDED variables. FUCK
                split id into external id and internal id
    """
    def __genIndexAToIndicesB(self, prodB: Production, compatibility: Compatibility):
        valIndexAToStepIndicesB = {}
        #print(f'__genIndexAToIndicesB: self production:       {self.production}')
        #print(f'__genIndexAToIndicesB: compatible production: {prodB}')
        #print(f'__genIndexAToIndicesB: {compatibility}')

        
        
        # Loop over A
        val_index_implicit_TO_explicit_sep_index = {}
        val_index_implicit_TO_explicit_val_index = {}
        if compatibility in [Compatibility.SELF_EXPORT, Compatibility.OTHER_EXPLICIT]:
            if compatibility == Compatibility.SELF_EXPORT:
                name        = self.name()
                input_steps = self.production.input_steps
                inputstep_to_compat_index = self.production.inputstep_to_compat_index
            else:
                name        = prodB.name
                input_steps = prodB.input_steps
                inputstep_to_compat_index = prodB.inputstep_to_compat_index
            #print(f'inputstep_to_compat_index: {inputstep_to_compat_index}')
            explicit_prod_input_val_index = 0
            for explicit_prod_input_step_index, step in enumerate(input_steps):
                
                settings = step.token.settings
                if State.__isStored(step, settings):
                    if not 'id' in step.token.settings:
                        print(f'{bcolors.FAIL}Production with name {name} is missing an id setting for input step: {explicit_prod_input_step_index}!{bcolors.ENDC}')
                        assert False
                    
                    val_index_implicit = step.token.settings['id']
                    #print(f'QQQQQQQQ: explicit step index: {explicit_prod_input_step_index}, implicit value index: {val_index_implicit}')
                    
                    #compat_index_a = inputstep_to_compat_index[explicit_prod_input_step_index]
                    if not val_index_implicit in val_index_implicit_TO_explicit_sep_index:
                        val_index_implicit_TO_explicit_sep_index[val_index_implicit] = []
                        val_index_implicit_TO_explicit_val_index[val_index_implicit] = []

                    val_index_implicit_TO_explicit_sep_index[val_index_implicit].append(explicit_prod_input_step_index)
                    val_index_implicit_TO_explicit_val_index[val_index_implicit].append(explicit_prod_input_val_index)
                    explicit_prod_input_val_index += 1

            #print(f'val_index_implicit_TO_explicit_sep_index: {val_index_implicit_TO_explicit_sep_index}')




        val_index_b = 0
        match compatibility:
            case Compatibility.OTHER_EXPLICIT:
                #B knows which values needs to be placed where
                valIndexAToStepIndicesB = val_index_implicit_TO_explicit_sep_index
            case Compatibility.EQUAL:
                for prodB_input_step_index, step in enumerate(prodB.input_steps):
                    settings = step.token.settings
                    if State.__isStored(step, settings):
                        valIndexAToStepIndicesB[val_index_b] = [prodB_input_step_index]
                        val_index_b += 1
            case Compatibility.SELF_EXPORT:
                for prodB_input_step_index, step in enumerate(prodB.input_steps):
                    settings = step.token.settings
                    if State.__isStored(step, settings):

                        if not val_index_b in val_index_implicit_TO_explicit_val_index:
                            # This case occurs when a new noitcudorp is placed there
                            pass
                            #print(f'ERROR val_index_b: {val_index_b} no in {val_index_implicit_TO_explicit_sep_index}')
                            #assert False
                        else:
                            val_index_a = val_index_implicit_TO_explicit_val_index[val_index_b]

                            #print(f'val_index_a: {val_index_a}')
                            assert len(val_index_a) == 1
                            valIndexAToStepIndicesB[val_index_a[0]] = [prodB_input_step_index]
                        val_index_b += 1
        
        #print(f'valIndexAToStepIndicesB: {valIndexAToStepIndicesB}')
        return valIndexAToStepIndicesB

    def esrap(self, rManagerA: RuleManager, rManagerB: RuleManager) -> str:
        return self.__unwarp(rManagerA, rManagerB, UnwrapMethod.ESRAP)

    def getDot(self, rManagerA: RuleManager, rManagerB: RuleManager, fileName: str):
        #graph = pydot.Dot('my_graph', graph_type='graph', bgcolor='white')
        graph = graphviz.Digraph(comment='The Round Table', node_attr={'shape': 'plaintext'})
        
        self.__unwarp(rManagerA, rManagerB, UnwrapMethod.DOT, 0, graph)
        graph.render(f'test-output/{fileName}', view=True)
        #graph.write_raw('output_raw.dot')

        #graph.write_png('output.png')

    @staticmethod
    def __createNode(string: str, settings, graph):
        name = uuid.uuid4().hex
        graph.node(name, label=string, shape='box')
        return name

    def __createNodeWithDeps(self, deps: list[str], settings, graph, name: Optional[str] = None):
        if name == None:
            name = uuid.uuid4().hex

        #print(f'self.production.input_steps[0]: {self.production.input_steps[0]}')

        def present(index):
            #print(f'self.values: {self.values}, index: {index}')
            return self.values[index] != None

        def infoToCol(index, step):
            escaped = html.escape(step.name())
            return f'<TD PORT="f{index}" BGCOLOR="{"white" if present(index) else "grey"}">{escaped}</TD>'
        
        cols = [infoToCol(index, step) for index, step in enumerate(self.production.input_steps)]
        code = ''.join(cols)
        graph.node(name, f'''<
<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
  <TR>
    <TD COLSPAN="{len(cols)}">{self.name()}</TD>
  </TR>
  <TR>
    {code}
  </TR>
</TABLE>>''')
        for i, dep in enumerate(deps):
            if present(i):
                outPort = name+':f'+str(i)
                my_edge = graph.edge(outPort, dep, color='black')
        return name

    @staticmethod
    def __createNodeFromList(string: str, strings: list[str], settings, graph, name: Optional[str] = None):
        if name == None:
            name = uuid.uuid4().hex

        graph.node(name, label=string, shape='box')
        for string in strings:
            id = string#State.__createNode(string, settings, graph)
            my_edge = graph.edge(name, id, color='orange')
        return name

    @staticmethod
    def __optPad(val: str, settings):
        string = val
        if containsAndTrue(settings, 'pad'):
            string = ' ' + string + ' '
        return string

    @staticmethod
    def __handleConversion(rManagerA: RuleManager, rManagerB: RuleManager, settings, val, method: UnwrapMethod, recursion_index: int, graph: Optional[str] = None) -> str:
        match method:
            case UnwrapMethod.ESRAP:
                func = State.__optPad
            case UnwrapMethod.DOT:
                func = lambda *args: State.__createNode(*args, graph)


        if isinstance(val, State):
            state = val.__unwarp(rManagerA, rManagerB, method, recursion_index+1, graph)
            match method:
                case UnwrapMethod.ESRAP:
                    ret = State.__optPad(state, settings)
                case UnwrapMethod.DOT:
                    ret = state
            return ret
        elif isinstance(val, Terminal):
            return func(val, settings)
        elif isinstance(val, NonTerminal):
            return func(val, settings)
        elif isinstance(val, str):
            return func(val, settings)
        elif isinstance(val, list):
            # DO NOT PAD
            strings = [State.__handleConversion(rManagerA, rManagerB, settings, v, method, recursion_index+1, graph) for v in val]
            
            match method:
                case UnwrapMethod.ESRAP:
                    ret = ''.join(strings)
                case UnwrapMethod.DOT:
                   ret = State.__createNodeFromList('list', strings, settings, graph)
            return ret
        else:
            assert False
        

    """
        NOTE: This uses the noiducorps of rManagerA and the productions of rManagerB
    """
    def __unwarp(self, rManagerA: RuleManager, rManagerB: RuleManager, method: UnwrapMethod, recursion_index = 0, graph: Optional[str] = None) -> str:
        assert self.isCompleted()
        tab_string = '\t' * recursion_index
        #print(f'-------BEGIN unwrap OF {self.name()}-------')
        #print(str(rManagerB))
        #print(f'self.fullStr(): {self.fullStr()}')


        # Algo:
        #   Do I exist in the _new_ productions?
        #   Is any of their productions compatible to me?
        #   Am I compatible to any of their productions?
        prodB = rManagerB.getProduction(self.production.uuid)
        if prodB != None:
            compatibility = Compatibility.EQUAL
        else:
            #print(f'=====CONVERSIONS ARE NEEDED=====')
            # conversions are needed
            prodB = rManagerB.getCompatableProduction(self.production.uuid)
            if prodB != None:
                compatibility = Compatibility.OTHER_EXPLICIT
            else:
                if self.production.uuid_compat == None:
                    print(f'{bcolors.FAIL}ERROR: PRODUCTION "{self.production.name}:{self.production.uuid}" NEEDS EXPLICIT COMPATIBILITY{bcolors.ENDC}')
                    assert False

                prodB = rManagerB.getProduction(self.production.uuid_compat)
                if prodB != None:
                    compatibility = Compatibility.SELF_EXPORT
                else:
                    print(f'{bcolors.FAIL}ERROR: COMPATIBLE PRODUCTION NOT FOUND{bcolors.ENDC}')
                    assert prodB != None
                    
                


        stepsA = self.production.input_steps
        stepsB = prodB.input_steps

        valIndexAToStepIndicesB = self.__genIndexAToIndicesB(prodB, compatibility)
        #print(f'{tab_string}valIndexAToStepIndicesB: {valIndexAToStepIndicesB}')


        strings = [None]*len(stepsB) 

        # Set trivial(strings)
        for i, step in enumerate(stepsB):
            if (is_trivial_step(step)):
                string = step.rule.tok
                settings = step.rule.settings
                if containsAndTrue(settings, 'pad'):
                    string = ' ' + string + ' '
                
                if containsAndTrue(settings, 'id'):
                    index = step.token.settings["id"]
                else:
                    index = i
                
                match method:
                    case UnwrapMethod.ESRAP:
                        strings[index] = string
                    case UnwrapMethod.DOT:
                        strings[index] = State.__createNode(string, settings, graph)
                
        # Set (Non)Terminals
        #print(f'{tab_string}stepsA: {stepsA}, self.values: {self.values}')
        assert len(stepsA) == len(self.values)
        value_index_a = 0
        for i, step in enumerate(stepsA):
            typeNameStep = type(step).__name__
            val = self.values[i]
            typeNameVal = type(val).__name__
            #print(f'{tab_string}{bcolors.OKGREEN}status: name {self.name()}, i: {i}, step: {step}, typeNameStep: {typeNameStep}, val: {val}, typeNameVal: {typeNameVal}{bcolors.ENDC}')
            
            if not State.__isStored(step, step.token.settings):
                #print(f'{tab_string}Not stored')
                pass
            elif isinstance(val, State):
                
                #print(f'{tab_string}State')
                compatIndices = valIndexAToStepIndicesB[value_index_a]
                
                for compatIndex in compatIndices:
                    #print(f'compatIndex: {compatIndex}')
                    settings = stepsB[compatIndex].token.settings
                    strings[compatIndex] = State.__handleConversion(rManagerA, rManagerB, settings, val, method, recursion_index+1, graph)
                value_index_a += 1
            #Check key    
            elif value_index_a in valIndexAToStepIndicesB:
                #print(f'{tab_string}i in valIndexAToStepIndicesB')
                assert(len(valIndexAToStepIndicesB[value_index_a]) == 1)
                compatIndex = valIndexAToStepIndicesB[value_index_a][0]

                if (isinstance(val, Terminal)):
                    isRegex = stepsA[i].rule.settings['regex']
                    isRegexB = stepsB[compatIndex].rule.settings['regex']
                    #print(f'isRegex: {isRegex}, isRegexB: {isRegexB}')
                    assert(isRegex == isRegexB)

                    strings[compatIndex] = State.__handleConversion(rManagerA, rManagerB, val.token.settings, val, method, recursion_index+1, graph)
                    
                elif (isinstance(val, NonTerminal)):
                    rule = stepsB[compatIndex].rule
                    strings[compatIndex] = State.__handleConversion(rManagerA, rManagerB, rule.settings, rule.tok, method, recursion_index+1, graph)
                elif (isinstance(val, str)):
                    settings = stepsB[compatIndex].token.settings
                    strings[compatIndex] = State.__handleConversion(rManagerA, rManagerB, settings, val, method, recursion_index+1, graph)
                elif (isinstance(val, list)):
                    settings = stepsB[compatIndex].token.settings
                    strings[compatIndex] = State.__handleConversion(rManagerA, rManagerB, settings, val, method, recursion_index+1, graph)
                elif val == None:
                    strings[compatIndex] = ''
                else:
                    typeName = type(val).__name__
                    print(f'ERROR: type {typeName}')
                    assert False
                value_index_a += 1
            else:
                typeName = type(val).__name__
                #assert False
        
        # Set noitcudorps
        #print(f'__unwrap strings: {strings}')
        if compatibility != Compatibility.EQUAL:
            for noitcudorpToken in self.production.noitcudorps:
                index = noitcudorpToken.token.settings["id"]
                noitcudorp = rManagerA.getNoitcudorp(noitcudorpToken.token.tok)
                #print(f'noitcudorpToken index: {index}')
                match method:
                        case UnwrapMethod.ESRAP:
                            generated = noitcudorp.generate()
                            #print(f'generated: {generated}')
                            strings[index] = generated
                        case UnwrapMethod.DOT:
                            txt = str([str(t.tok) for t in noitcudorp.tokens])
                            #print(f'noitcudorp txt: {txt}')
                            strings[index] = State.__createNode(txt, None, graph)

        #print(f'-------END ESRAP OF {self.name()}-------')
        #print(strings)
        strings = [str if str != None else 'BOOO' for str in strings ]

        match method:
            case UnwrapMethod.ESRAP:
                ret = ''.join(strings)
            case UnwrapMethod.DOT:
                State.__createNodeWithDeps(self, strings, None, graph, self.uuid.hex)
                ret = self.uuid.hex
                
        #print(f'__unwrap ret: {ret}')
        return ret


class Column():
    def __init__(self, ruleManager, states):
        self.states = states
        self.ruleManager = ruleManager

    def __containsState(self, name: str, index: int):
        for state in self.states:
            if (state.production.name == name and state.position == index):
                #print(f'{state.production.name} == {name}, {state.position} == {index}')
                #print(f'containsState true')
                return True
        #print(f'containsState false')
        return False

    def predict(self, productionName: str, currentChart) -> List[State]:
        new = []
        #print(f'predict: productionName: {productionName}, currentChart {currentChart}')
        FoundMatch = False
        for prod in self.ruleManager.productions:
            #print(f'predict: prod.name "{prod.name}"')
            if (prod.name == productionName):
                #print('matching name')
                FoundMatch = True
                if (not (self.__containsState(productionName, 0))):
                    #print(f'predicted: {prod.name}')
                    # So it does not contain the zero state
                    newState = State(prod, currentChart)
                    new.append(newState)
                    new.extend(newState.createInitial())
                
        if not FoundMatch:
            print(f'{bcolors.FAIL}ERROR: production with name "{productionName}" not found!{bcolors.ENDC}')
            assert False
        #print(f'predict new state: {str([str(state) for state in new])}')
        return new
        
    
    def append(self, state):
        self.states.append(state)
    
    def extend(self, states):

        self.states.extend(states)


def complete(table, state: State):
    # complete non terminals
    #print(f'complete name: {state.production.name}, from: {state.originPosition}')
    colJ = table[state.originPosition].states
    newStates = []
    for stateJ in colJ:
        if (not stateJ.isCompleted()) and (stateJ.getNextName() == state.production.name) and not stateJ.nextIsTerminal():
            #print(f' completing name: {stateJ.name()} from: {stateJ.originPosition}')
            newState = copy.deepcopy(stateJ)
            newStates.extend(newState.advance(state))
            newStates.append(newState)
    return newStates

class TokenizeSettings(Enum):
    UNKNOWN              = -1,
    NORMAL               = 0,
    NORMAL_SETTINGS_INT  = 1,
    QUOTE_SINGLE         = 2,
    QUOTE_DOUBLE         = 3,
    SQUARE_BRACKET       = 4


def tokenize(input: str, interupts = ['+', '-', '*', ':', '/', '(', ')', '\n', ',', '{', '}', '\'', '→', '⇇', ';']):
    tokens = []
    curr = ''
    status = TokenizeSettings.NORMAL
    old_char = ''
    escaped = False
    for c in input:
        if c == '\\' and not escaped:
            escaped = True
            #print('NOW ESCAPED')
            continue
        #tokens_txt = ' '.join(map(str, tokens))
        #print(f'status: {status}, escaped: {escaped}, curr: "{curr}" adding char: ' + c)
        
        if status == TokenizeSettings.NORMAL_SETTINGS_INT and c == '{':
            tokens.append(curr)
            status = TokenizeSettings.NORMAL
            curr = '{'
        elif status == TokenizeSettings.NORMAL_SETTINGS_INT and c == '}':
            tokens.append(curr)
            tokens.append('}')
            status = TokenizeSettings.NORMAL
            curr = ''
        elif (status in [TokenizeSettings.NORMAL, TokenizeSettings.NORMAL_SETTINGS_INT] and c in [' ', '\n']): 
            tokens.append(curr)
            if c == '\n':
                tokens.append(c)
            status = TokenizeSettings.NORMAL
            curr = ''
        elif (status == TokenizeSettings.NORMAL and c == '"' and not escaped):
            tokens.append(curr)
            curr = c
            status = TokenizeSettings.QUOTE_DOUBLE
        elif (status == TokenizeSettings.NORMAL and c == '['):
            tokens.append(curr)
            curr = c
            status = TokenizeSettings.SQUARE_BRACKET
        elif (status == TokenizeSettings.NORMAL and c == '-' and (not old_char in "0123456789")):
            curr += c
        elif (status == TokenizeSettings.QUOTE_DOUBLE):
            curr += c
            if c == '"' and not escaped:
                status = TokenizeSettings.NORMAL
        elif (status == TokenizeSettings.SQUARE_BRACKET):
            curr += c
            if c == ']':
                status = TokenizeSettings.NORMAL_SETTINGS_INT
        elif ((status == TokenizeSettings.NORMAL) and (c in interupts)):
            tokens.append(curr)
            tokens.append(c)
            curr = ''
        elif (status == TokenizeSettings.NORMAL and c not in ['"', '[']) or status == TokenizeSettings.NORMAL_SETTINGS_INT:
            curr += c
        else:
            print(f'ERROR: status: {status}')
            assert False
        old_char = c
        escaped = False

    if(curr != ''):
        tokens.append(curr)
    
    tokens = [tok for tok in tokens if (len(tok) != 0)]
    return tokens

def tokenizeFromJson(code: list):
    tokens = []
    #print(f'code: {code}')
    for line in code:
        #print(f'line: {line}')
        for obj in line['words']:
            #print(f'obj: {obj}')
            #If type is 1 it;s modified
            if (obj['style'] == 1):
                newTokens = tokenize(obj['word'])
                tokens.extend(newTokens)
            else:
                tokens.append(obj['word'])
    
    tokens = [tok for tok in tokens if (len(tok) != 0)]
    return tokens

def match(ruleManager: RuleManager, inTokens: list[str], beginRules: list[uuid.UUID] = None):
    #print(str(ruleManager))
    tokenStr = '\n\t'.join([str(tok) for tok in inTokens])
    #print(f'tokenized: \n\t{tokenStr}, len: {len(inTokens)}')
    #table = [Column(ruleManager, [State(prod, 0) for prod in ruleManager.productions])]
    table = [Column(ruleManager, []) for i in range(len(inTokens)+1)]

    if beginRules == None:
        table[0].extend([State(ruleManager.productions[0], 0)])
        #TODO: This only handles the first rule in a line not all the rules in that line...
        beginRules = [ruleManager.productions[0].uuid]
    else:
        table[0].extend([State(ruleManager.productions[i], 0) for i in range(len(ruleManager.productions)) if ruleManager.productions[i].uuid in beginRules])
    

    # Init 
    newStates = []
    for state in table[0].states:
        newStates.extend(state.createInitial())
    table[0].extend(newStates)

    def predict(col, state, currentChart):
        #Prediction
        #assert not state.nextIsTerminal()
        name = state.getNextName()
        #print(f'Predicting {currentChart}, adding: {name}')
        
        col.extend(col.predict(name, currentChart))
    
    for currentChart, col in enumerate(table):
        #pre
        tok = inTokens[currentChart] if currentChart<len(inTokens) else None
        #print(f'------{currentChart}, {tok}: PRE------')
        #print('\n'.join(map(str, col.states)))

        #real work
        for state in col.states:
            #print(f'sate name: {state.production.name}, checking completion')
            if (state.isCompleted()):
                #print(f'sate name: {state.production.name}, is completed: {state.isCompleted()}!')
                col.states.extend(complete(table, state))

            elif tok != None:
                if state.nextIsTerminal():
                    #print(f'{state.production.name} is scanning')
                    newStates = state.MatchThenAdvanceStateCopies(tok)
                    table[currentChart+1].extend(newStates)
                else:
                    #New NonTerminals may be found
                    predict(col, state, currentChart)

        #post
        #print(f'------{currentChart}, {bcolors.OKBLUE}{repr(tok)}{bcolors.ENDC}: POST------')
        #print('\n'.join(map(str, col.states)))
    
    # Find result
    matches = []
    for status in table[-1].states:
        if (status.originPosition == 0 and status.isCompleted() and status.production.uuid in beginRules):
            matches.append(status)
    
    #print(f'MATCHES: {matches}')

    #for index, match in enumerate(matches):
    #    match.getDot(ruleManager, ruleManager, f'index-{index}.gv')
    if (len(matches) > 1):
        print(f'{bcolors.FAIL}ERROR: MULTIPLE MATCHES{bcolors.ENDC}')
        for index, match in enumerate(matches):
            match.getDot(ruleManager, ruleManager, f'index-{index}.gv')
            #print(f'{bcolors.FAIL}{esr}{bcolors.ENDC}')
        
        return None
    
    return matches[0] if len(matches) == 1 else None
