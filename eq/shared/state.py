import copy

from enum   import Enum
from typing import Optional

from .helper import *
from .expression import *



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
        self.uuid = uuid.uuid4()

    def __skipToPosPad(self, pos):
        assert self.position <= pos
        while self.position < pos:
            self.values.append(None)
            self.position += 1
    
    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        result.values = copy.deepcopy(self.values, memodict)
        return result
    
    def name(self) -> str:
        return self.production.name

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


