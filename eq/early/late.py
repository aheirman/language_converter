from __future__ import annotations

import copy
import uuid
import itertools
import graphviz
import html


from eq.helper import *
from eq.shared.expression import *
from eq.shared.state import *

class EarlyState(State):
    def __init__(self, production, originPosition):
        self.originPosition = originPosition
        self.position = 0
        super().__init__(production, originPosition)

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
    


    #NOTE: The position shift of us occurs after this is run
    def createNewStates(self) -> List[EarlyState]:
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
    def createInitial(self) -> List[EarlyState]:
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

    def advance(self, val) -> list[EarlyState]:
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

    def predict(self, productionName: str, currentChart) -> List[EarlyState]:
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
                    newState = EarlyState(prod, currentChart)
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


def complete(table, state: EarlyState):
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


def match(ruleManager: RuleManager, inTokens: list[str], beginRules: list[uuid.UUID] = None):
    #print(str(ruleManager))
    #tokenStr = '\n\t'.join([str(tok) for tok in inTokens])
    #print(f'tokenized: \n\t{tokenStr}, len: {len(inTokens)}')
    #table = [Column(ruleManager, [State(prod, 0) for prod in ruleManager.productions])]
    table = [Column(ruleManager, []) for i in range(len(inTokens)+1)]

    if beginRules == None:
        table[0].extend([EarlyState(ruleManager.productions[0], 0)])
        #TODO: This only handles the first rule in a line not all the rules in that line...
        beginRules = [ruleManager.productions[0].uuid]
    else:
        table[0].extend([EarlyState(ruleManager.productions[i], 0) for i in range(len(ruleManager.productions)) if ruleManager.productions[i].uuid in beginRules])
    

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
        print(f'------{currentChart}, {bcolors.OKBLUE}{repr(tok)}{bcolors.ENDC}: POST------')
        print('\n'.join(map(str, col.states)))
    
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
