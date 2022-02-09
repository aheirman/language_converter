from queue import PriorityQueue

from eq.shared.expression import *
from eq.shared.state import *


class MemoKey:
    def __init__(self, clause, startPos):
        self.clause = clause
        self.startPos = startPos

class MemoTable:
    def __init__(self, tokens: list[str]):
        self.tokens = tokens
    
    def add_match(self):
        pass

"""
    What is a clause?
        It is either a terminal or a non terminal.
        So, one array that converts NonTerminal to their IDs and one for the terminal text to their ID.
"""
class Clause:
    def __init__(self, terminal=None, production=None):
        assert (terminal==None) is not (production==None)
        self.is_term    = terminal != None
        self.terminal   = terminal
        self.production_uuid = production_uuid
        self.clause_idx = -1
        self.__determineWhetherCanMatchZeroChars()

    """
        Can this clause match zero chars
    """
    def __determineWhetherCanMatchZeroChars(self):
        self.can_match_zero_chars = False

    """
        ?
    """
    def addAsSeedParentClause(self):
        pass

    """
        Can all necessary subclauses match the memo table
            Returns: Match if match,
                     None otherwise
    """
    def match(self):
        pass

    def name(self):
        if self.is_term:
            return self.terminal.name()
        else:
            return self.production.name


def __get_all_prod_with_name(rule_manager, name):
    prods = []
    for prod in rule_manager.productions:
        if prod.name == name:
            prods.append(prod)
    return prods

def __get_sub_clauses(rule_manager, clause):
    subs = []
    if not isinstance(clause, Terminal):
        assert isinstance(clause, NonTerminal)
        prod = rule_manager.getProduction(uuid)
        for step in clause.input_steps:
            subs.append(step)
    return subs

"""
    clause: this is a STRING of the name of the production
"""
def __findReachableClauses(rule_manager, clause, visited, revTopoOrderOut):
    if not (clause in visited):
        print(f'findReachableClauses clause ({clause} has not yet been visited)')
        visited.add(clause)
        for sub_clause in __get_sub_clauses(rule_manager, clause):
            print(f'findReachableClauses: sub_clause: {sub_clause.name()}, type: {type(sub_clause)}')
            __findReachableClauses(rule_manager, sub_clause, visited, revTopoOrderOut)
        revTopoOrderOut.append(clause)

def __findCycleHeadClauses(rule_manager, clause, discovered, finished, cycle_head_clauses_out):
    #print(f'__findCycleHeadClauses_1: {type(clause).__name__}')
    print(f'findCycleHeadClauses_2: \'{clause.name()}\', discovered: {discovered}, finished: {finished}')
    discovered.add(clause)

    subclauses = __get_sub_clauses(rule_manager, clause)
    print(f'findCycleHeadClauses_3: subclauses:{subclauses}')
    for sub_clause in subclauses:
        print(f'\tsub_clause: {sub_clause}')
        if sub_clause in discovered:
            # We're in a cycle
            cycle_head_clauses_out.add(sub_clause)
        elif not (sub_clause in finished):
            #print(f'sub_clause ({sub_clause}) is not in finished)')
            __findCycleHeadClauses(rule_manager, sub_clause, discovered, finished, cycle_head_clauses_out)
    discovered.remove(clause)
    finished.add(clause)


def __findClauseTopoSortOrder(rule_manager: RuleManager, lowest_precedence_clauses: list, begin_rules = None):
    print('----')
    # Find top level clauses
    # WARN: THIS IS DIFFERENT FROM PAPER?
    top_level_clauses = []
    if begin_rules == None:
        top_level_clauses = [NonTerminal(Token(rule_manager.productions[0].name,{}))]
    else:
        assert False
    
    # AFTER top level clauses 
    #   Start depth first search _from_
    #   all lowest precedence clauses in each precedence hierarchy
    depth_first_search_roots = top_level_clauses
    depth_first_search_roots.extend(lowest_precedence_clauses)

    # Add toplevel clauses the set of all " head clauses " of cycles ( all clauses reachable twice )
    cycle_discovered   = set()
    cycle_finished     = set()
    cycle_head_clauses = set()
    for clause in top_level_clauses:
        #print(f'Iterating over top_level_clauses: type: {type(clause)}')
        print(f'Iterating over top_level_clauses: name: \t{clause.name()}')
        __findCycleHeadClauses(rule_manager, clause, cycle_discovered, cycle_finished, cycle_head_clauses)
    print('\n')
    for prod in rule_manager.productions:
        print(f'Iterating over all productions: \t name: {prod.name}, \t uuid: {prod.uuid}')
        __findCycleHeadClauses(rule_manager, prod, cycle_discovered, cycle_finished, cycle_head_clauses)
    depth_first_search_roots.extend(cycle_head_clauses)

    print(f'\ndepth_first_search_roots: {[str(a.name()) for a in depth_first_search_roots]}')

    # Topologically sort all clauses into bottom-up order
    all_clauses = []
    reachable_visited = set()
    for top_level_clause in depth_first_search_roots:
        __findReachableClauses(rule_manager, top_level_clause.name, reachable_visited, all_clauses)
    
    # Give each clause an index in the topological sort order, bottom-up
    NonTerminal_idx = {}
    Terminal_idx    = {}
    for i in range(len(all_clauses)):
        print(f'Giving clause {all_clauses[i]} id: {i}')
        assert isinstance(all_clauses[i], Clause)
        
        all_clauses[i].clause_idx = i

    print('-----')
    return [NonTerminal_idx, Terminal_idx]


def parse(ruleManager: RuleManager, tokens: list[str], beginRules: list = None) -> MemoTable:
    
    all_clauses = __findClauseTopoSortOrder(ruleManager, [])
    #print(f'all_clauses, Terminal: {[f'{i}:{a}' for a,b in all_clauses[1]]}')
    print(f'all_clauses, Terminal: {all_clauses[1]}')
    print(f'all_clauses, NonTerminal: {all_clauses[0]}')

    queue = PriorityQueue()
    memoTable = MemoTable(tokens)
    terminals = set()

    # Get terminals
    for prod in ruleManager.productions:
        for step in prod.input_steps:
            print(f'{prod.name}: "{step.name()}" \tis terminal: {isinstance(step, Terminal)}')
            if isinstance(step, Terminal):
                terminals.add(step.name())
    print(f'Terminals: {terminals}')

    # Main parsing loop
    for startPos in range(len(tokens)-1, 0, -1):
        #Add all terminals to the queue
        for t in terminals:
            queue.put(t)
        while not queue.empty():
            curr_clause = queue.get()
            memoKey     = MemoKey(curr_clause, startPos)
            match       = curr_clause.match(memoTable, memoKey, input)
            memoTable.addMatch(memoKey, match, queue)
    return memoTable