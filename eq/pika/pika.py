import copy
from queue import PriorityQueue
from dataclasses import dataclass

from eq.shared.expression import *
from eq.shared.state import *


class MemoKey:
    def __init__(self, clause, startPos):
        self.clause = clause
        self.startPos = startPos
    
    def __str__(self):
        return f'(start pos: {self.startPos}:\'{self.clause.name()}\')'

    
    def __eq__(self, other):
        #print(f'__eq__: {self.startPos == other.startPos}, {self.clause.name() == other.clause.name()}')
        return (self.startPos == other.startPos) and (self.clause.name() == other.clause.name())
    
    def __hash__(self):
        return hash((self.clause.name(), self.startPos))
    
@dataclass
class Match:
    memoKey: MemoKey
    len:     int
    first_matching_sub_clause_idx: int
    sub_clause_matches: list#[Match]

    def is_better_than(self, other):#: Match):
        # If first -> earlier is better
        # a longer match is better
        return (self.memoKey.clause.is_first and \
         self.first_matching_sub_clause_idx < other.first_matching_sub_clause_idx) or \
         self.len > other.len

class MemoTable:
    def __init__(self, tokens: list[str]):
        self.tokens    = tokens
        self.memoTable = {}
    
    def add_match(self, memoKey: MemoKey, new_match: Match, priority_queue):
        updated = False
        if new_match != None:
            old_match = self.memoTable.get(memoKey)
            
            if (old_match == None) or new_match.is_better_than(old_match):
                self.memoTable[memoKey] = new_match
                updated = True
        else:
            print('\tadd_match: no match provided')

        for seed_parent_clause in memoKey.clause.seed_parent_clauses:
            if updated or seed_parent_clause.can_match_zero_chars:
                print(f'priority_queue adding seed_parent_clause: {seed_parent_clause}')
                priority_queue.put(seed_parent_clause)
    
    def look_up_best_match(self, memoKey: MemoKey) -> Match:
        
        #print(f'LOOK_UP_BEST_MATCH 1: type: {type(memoKey)}')
        best_match = self.memoTable.get(memoKey)
        print(f'LOOK_UP_BEST_MATCH 2: {self}')
        print(f'LOOK_UP_BEST_MATCH 3: memoKey: {memoKey},\n\t best_match: {best_match}')
        #print('-----look_up_best_match 2')
        if best_match != None:
            print(f'LOOK_UP_BEST_MATCH 4: FOUND')
            return best_match
        #elif memoKey.clause is NotFollowedBy
        #    return memoKey.clause.match(self, memoKey, tokens)
        elif memoKey.clause.can_match_zero_chars:
            print(f'LOOK_UP_BEST_MATCH 5: FOUND ZERO LENGTH')
            return Match(memoKey, 0, 0, [])

    def __str__(self):
        out = ''
        #for a, b in self.memoTable.items():
        #    out = out+f'{a}: {b}\n'
        for k in self.memoTable.keys():
            out = out+f'{k}'
        return out
        #return str(self.memoTable)

"""
    What is a clause?
        It is either a terminal or a non terminal.
        So, one array that converts NonTerminal to their IDs and one for the terminal text to their ID.

        If it is a terminal   it is: Terminal
        If it is a production it is: Seq
            A seq knows how to deal with optionals
        if it is a production uuids: First
"""
class Clause:
    def __init__(self, terminal_name=None, production=None, production_uuids = None, rule_name = None):
        #print(f'production: {production}')
        assert (int(terminal_name==None) + int(production==None) + int(production_uuids==None)) == 2
        assert (production_uuids==None) == (rule_name==None)
        self.is_term  = terminal_name    != None
        self.is_seq   = production       != None
        self.is_first = production_uuids != None

        self.rule_name = rule_name
        self.terminal_name = terminal_name
        self.production = production
        self.clause_idx = None
        self.production_uuids = production_uuids
        self.labeled_sub_clauses = []
        self.seed_parent_clauses = set()

        self.__determineWhetherCanMatchZeroChars()



    def calculate_sub_clauses(self, terminal_to_clause: list, rule_to_clause: list, prod_uuid_to_clause: list):
        #print(f'calculate_sub_clauses PRE: {self.labeled_sub_clauses}')
        if self.is_first:
            self.labeled_sub_clauses = [prod_uuid_to_clause[uuid] for uuid in self.production_uuids]
        elif self.is_seq:
            for step in self.production.input_steps:
                if isinstance(step, Terminal):
                    self.labeled_sub_clauses.append(terminal_to_clause[step.name()])
                elif isinstance(step, NonTerminal):
                    self.labeled_sub_clauses.append(rule_to_clause[step.name()])
                else:
                    assert False
        elif self.is_term:
            pass
        else:
            assert False
        #print(f'labeled_sub_clauses {self.name()}: {[str(a) for a in self.labeled_sub_clauses]}')
        #print(f'calculate_sub_clauses POST: {self.labeled_sub_clauses}')

    """
        Can this clause match zero chars
    """
    def __determineWhetherCanMatchZeroChars(self):
        self.can_match_zero_chars = False

    """
        ?
    """
    def addAsSeedParentClause(self):
        for sub in self.labeled_sub_clauses:
            assert isinstance(sub, Clause)
            #print(f'QQQQ addAsSeedParentClause self name: {self.name()} \tsub name {sub.name()}, \tsub id {id(sub)}, id self: {id(self)}')
            sub.seed_parent_clauses.add(self)

    """
        Can all necessary subclauses match the memo table
            Returns: Match if match,
                     None otherwise
    """
    def match(self, memoTable: MemoTable, memoKey: MemoKey, tokens: list[str]):
        
        if self.is_term:
            print('MATCHing TERM')
            if memoKey.startPos < len(tokens) and self.terminal_name == tokens[memoKey.startPos]:
                return Match(memoKey, 1, 0, [])
        elif self.is_seq:
            print('MATCHing SEQ')
            curr_start_pos = memoKey.startPos
            sub_clause_matches = [None]*len(self.labeled_sub_clauses)
            print(f'MATCH SEQ pending {self.production.name}, input steps len: {len(self.production.input_steps)}, sub clauses len: {len(self.labeled_sub_clauses)}')
            
            if not (len(self.production.input_steps) == len(self.labeled_sub_clauses)):
                print(f'WARN: self.labeled_sub_clauses: {[str(a) for a in self.labeled_sub_clauses]}')
            #    assert False

            for sub_clause_iteration_idx in range(len(self.labeled_sub_clauses)):
                sub            = self.labeled_sub_clauses[sub_clause_iteration_idx]
                print(f'\tMATCH SEQ pending sub_clause_iteration_idx: {sub_clause_iteration_idx}, sub: {str(sub)}')
                
                sub_memo_key   = MemoKey(sub, curr_start_pos)
                sub_match = memoTable.look_up_best_match(sub_memo_key)
                if(sub_match == None):
                    print(f'\tMATCH SEQ pending sub_clause_iteration_idx: {sub_clause_iteration_idx},  sub {sub.name()} did not result in a match')
                    return None
                sub_clause_matches[sub_clause_iteration_idx] = sub_match
                curr_start_pos += sub_match.len
            return Match(memoKey, curr_start_pos - memoKey.startPos, 0, sub_clause_matches)
        elif self.is_first:
            print('MATCHing FIRST')
            curr_start_pos = memoKey.startPos
            for sub_clause_iteration_idx in range(len(self.labeled_sub_clauses)):
                sub = self.labeled_sub_clauses[sub_clause_iteration_idx]
                sub_memo_key   = MemoKey(sub, curr_start_pos)
                sub_match = memoTable.look_up_best_match(sub_memo_key)
                if(sub_match != None):
                    return Match(memoKey, sub_match.len, sub_clause_iteration_idx, [sub_match])
        else:
            assert False
        return None

    def name(self):
        if self.is_term:
            return self.terminal_name
        elif self.is_first:
            return ''.join(self.production_uuids)
        elif self.is_seq:
            return self.production.name

    def __str__(self):
        #if self.is_term:
        #    return f'(Term: {self.terminal_name}, id: {id(self)}, idx: {self.clause_idx}, seed_parent_clauses: {[str(a) for a in self.seed_parent_clauses]})'
        #elif self.is_first:
        #    return f'(First: {"".join(self.production_uuids)}, id: {id(self)}, idx: {self.clause_idx})'
        #elif self.is_seq:
        #    return f'(Seq: {self.production.name}, id: {id(self)}, idx: {self.clause_idx}, seed_parent_clauses: {[str(a) for a in self.seed_parent_clauses]})' 
        if self.is_term:
            return f'(Term: {self.terminal_name}, idx: {self.clause_idx})'
        elif self.is_first:
            return f'(First: {"".join(self.production_uuids)}, idx: {self.clause_idx})'
        elif self.is_seq:
            return f'(Seq: {self.production.name}, idx: {self.clause_idx})' 

    def __lt__(self, other):
        return self.clause_idx < other.clause_idx

    """
    def __eq__(self, other):
        return (self.clause_idx == other.clause_idx) and \
            (self.is_term == other.is_term) and \
            (self.is_seq == other.is_seq) and \
            (self.is_first == other.is_first) and \
            (self.production == other.pro)
    """

def __get_sub_clauses(rule_manager, terminal_to_clause, rule_to_clause, rule_to_sub_clauses, clause: Clause):
    """
    prods = __get_all_prod_with_name(rule_manager, clause)
    
    subs = set()
    for prod in prods:j
        for step in prod.input_steps:
            subs.add(step.name())
    """
    if clause.is_term:
        return []
    elif clause.is_first:
        #print(f'rule_to_clause: {rule_to_clause}')
        #print(f'rule_to_sub_clauses: {rule_to_sub_clauses}')
        #print(f'clause.name(): {clause.name()}')
        return rule_to_sub_clauses[clause.rule_name]
    elif clause.is_seq:
        subs = []
        for step in clause.production.input_steps:
            if isinstance(step, Terminal):
                subs.append(terminal_to_clause[step.name()])
            else:
                assert isinstance(step, NonTerminal)
                #print(f'step.name(): {step.name()}, rule_to_clause[step.name()]: {[ a.name() for a in rule_to_clause[step.name()]]}')
                subs.append(rule_to_clause[step.name()])
        #print(f'get_sub_clauses: clause ({clause}): {[a for a in subs]}')
        return subs


def __get_all_prod_with_name(rule_manager, name):
    prods = []
    for prod in rule_manager.productions:
        if prod.name == name:
            prods.append(prod)
    return prods


"""
    clause: this is a STRING of the name of the production
"""
def __findReachableClauses(rule_manager, terminal_to_clause: list, rule_to_clause: list, rule_to_sub_clauses: list, clause, visited, revTopoOrderOut):
    if not (clause in visited):
        #print(f'findReachableClauses clause ({clause} has not yet been visited)')
        visited.add(clause)
        for sub_clause in __get_sub_clauses(rule_manager, terminal_to_clause, rule_to_clause, rule_to_sub_clauses, clause):
            #print(f'findReachableClauses: sub_clause: {sub_clause}')
            __findReachableClauses(rule_manager, terminal_to_clause, rule_to_clause, rule_to_sub_clauses, sub_clause, visited, revTopoOrderOut)
        revTopoOrderOut.append(clause)

def __findCycleHeadClauses(rule_manager, terminal_to_clause: list, rule_to_clause: list, rule_to_sub_clauses: list, clause, discovered, finished, cycle_head_clauses_out):
    #print(f'__findCycleHeadClauses_1: {type(clause).__name__}')
    #print(f'findCycleHeadClauses_2: \'{clause.name()}\', discovered: {discovered}, finished: {finished}')
    #print(f'findCycleHeadClauses_3: \'{clause.name()}\'')
    discovered.add(clause)

    subclauses = __get_sub_clauses(rule_manager, terminal_to_clause, rule_to_clause, rule_to_sub_clauses, clause)
    #print(f'findCycleHeadClauses_4: subclauses:{subclauses}')
    for sub_clause in subclauses:
        #print(f'\tsub_clause: {sub_clause}')
        if sub_clause in discovered:
            # We're in a cycle
            cycle_head_clauses_out.add(sub_clause)
        elif not (sub_clause in finished):
            #print(f'sub_clause ({sub_clause}) is not in finished)')
            __findCycleHeadClauses(rule_manager, terminal_to_clause, rule_to_clause, rule_to_sub_clauses, sub_clause, discovered, finished, cycle_head_clauses_out)
    discovered.remove(clause)
    finished.add(clause)

def __create_all_clauses(rule_manager: RuleManager, terminal_to_clause: list, rule_to_clause: list, rule_to_sub_clauses: list, prod_uuid_to_clause: list):
    print(f'__create_all_clauses: {rule_manager.rule_to_ordered_productions}')
    for rule, prod_uuids in rule_manager.rule_to_ordered_productions.items():
        first_subs = []
        for prod_uuid in prod_uuids:
            print(f'create_all_clauses rule: {rule}, \tproduction uuid: {prod_uuid}')
            prod = rule_manager.getProduction(prod_uuid)
            curr_clause = Clause(production=prod)
            prod_uuid_to_clause[prod.uuid] = curr_clause
            first_subs.append(curr_clause)
            for step in prod.input_steps:
                if isinstance(step, Terminal):
                    name = step.name()
                    if not name in terminal_to_clause:
                        terminal_to_clause[name] = Clause(terminal_name=name)
        rule_to_sub_clauses[rule] = first_subs
        rule_to_clause[rule]      = Clause(production_uuids=prod_uuids,rule_name=rule)

def __findClauseTopoSortOrder(rule_manager: RuleManager, top_level_clauses: list, terminal_to_clause: list, rule_to_clause: list\
    , rule_to_sub_clauses: list, prod_uuid_to_clause: list, lowest_precedence_clauses: list, begin_rules = None):
   
    # AFTER top level clauses 
    #   Start depth first search _from_
    #   all lowest precedence clauses in each precedence hierarchy
    #print(f'top_level_clauses: {top_level_clauses}')
    #depth_first_search_roots = copy.copy(top_level_clauses)
    depth_first_search_roots = top_level_clauses
    print(f'top_level_clauses: {[str(a) for a in top_level_clauses]}')
    depth_first_search_roots.extend(lowest_precedence_clauses)

    # Add toplevel clauses the set of all " head clauses " of cycles ( all clauses reachable twice )
    cycle_discovered   = set()
    cycle_finished     = set()
    cycle_head_clauses = set()
    for clause in top_level_clauses:
        #print(f'Iterating over top_level_clauses: name: \t{clause.name()}')
        __findCycleHeadClauses(rule_manager, terminal_to_clause, rule_to_clause, rule_to_sub_clauses, clause, cycle_discovered, cycle_finished, cycle_head_clauses)
    print('\n')
    for prod in rule_manager.productions:
        #print(f'Iterating over all productions: \t name: {prod.name}, \t uuid: {prod.uuid}')
        __findCycleHeadClauses(rule_manager, terminal_to_clause, rule_to_clause, rule_to_sub_clauses, Clause(production=prod), cycle_discovered, cycle_finished, cycle_head_clauses)
    depth_first_search_roots.extend(cycle_head_clauses)

    #print(f'\ndepth_first_search_roots: {[str(a.name()) for a in depth_first_search_roots]}')

    # Topologically sort all clauses into bottom-up order
    all_clauses = []
    reachable_visited = set()
    for top_level_clause in depth_first_search_roots:
        __findReachableClauses(rule_manager, terminal_to_clause, rule_to_clause, rule_to_sub_clauses, top_level_clause, reachable_visited, all_clauses)
    
    # Give each clause an index in the topological sort order, bottom-up
    for i in range(len(all_clauses)):
        print(f'Giving clause {all_clauses[i]} id: {i}')
        assert isinstance(all_clauses[i], Clause)
        
        all_clauses[i].clause_idx = i

    print('-----')
    
    return all_clauses

def __memoKey_to_state(rule_manager: RuleManager, match: Match):
    print(f'__memoKey_to_state match: {match.memoKey.clause.name()}')
    
    if match.memoKey.clause.is_first:
        assert len(match.sub_clause_matches) == 1

        return __memoKey_to_state(rule_manager, match.sub_clause_matches[0])
    elif match.memoKey.clause.is_seq:
        print(f'__memoKey_to_state match: {match.memoKey.clause.production.name}')
        prod = match.memoKey.clause.production
        s = State(prod)
        values = [None]*len(prod.input_steps)
        for ind, step in enumerate(prod.input_steps):
            if isinstance(step, Terminal):
                pass
            elif isinstance(step, NonTerminal):
                values[ind] = __memoKey_to_state(rule_manager, match.sub_clause_matches[ind])
            else:
                assert False
        s.values = values
        #print(f'__memoKey_to_state match: {match.memoKey.clause.name()} CLOSE, vals: {values}')
    else:
        assert False
    return s

def __memoTable_to_state(rule_manager: RuleManager, top_level_clauses: list, memoTable: MemoTable) -> State:
    #print('AAAAAAAA')
    #print(f'__memoTable_to_state: {top_level_clauses}')
    assert len(top_level_clauses) == 1
    #print(f'__memoTable_to_state: {top_level_clauses[0].name()}')

    match = memoTable.look_up_best_match(MemoKey(top_level_clauses[0],0))
    if match != None:
        
        #for s_match in match.sub_clause_matches:
        #    s2 = State(s_match.memoKey.clause.production)
        #    values.append(s2)
        #s.values = values
        s = __memoKey_to_state(rule_manager, match)
        

        return s
    print('ERROR memoTable does not have complete parse')
    return None

def parse(rule_manager: RuleManager, tokens: list[str], begin_rules: list = None) -> MemoTable:
    print(f'parse: {rule_manager.rule_to_ordered_productions}')
    #indexed by name
    terminal_to_clause  = {}
    rule_to_clause      = {}
    rule_to_sub_clauses = {} 
    prod_uuid_to_clause = {}

    top_level_clauses   = []

    #print('----')
    __create_all_clauses(rule_manager, terminal_to_clause, rule_to_clause, rule_to_sub_clauses, prod_uuid_to_clause)
    
    # step 2: update sub_clauses
    for c in prod_uuid_to_clause.values():
        c.calculate_sub_clauses(terminal_to_clause, rule_to_clause, prod_uuid_to_clause)
    for c in rule_to_clause.values():
        c.calculate_sub_clauses(terminal_to_clause, rule_to_clause, prod_uuid_to_clause)
    
    # step 3: add seed parent clauses
    for c in prod_uuid_to_clause.values():
        c.addAsSeedParentClause()
    for c in rule_to_clause.values():
        c.addAsSeedParentClause()
    for c in  terminal_to_clause.values():
        c.addAsSeedParentClause()
    #print('__create_all_clauses: Finished')


    # Find top level clauses
    # WARN: THIS IS DIFFERENT FROM PAPER?
    if begin_rules == None:
        top_level_clauses.append(prod_uuid_to_clause[rule_manager.productions[0].uuid])
    else:
        assert False

    all_clauses = __findClauseTopoSortOrder(rule_manager, top_level_clauses, terminal_to_clause, rule_to_clause, rule_to_sub_clauses, prod_uuid_to_clause, [])
    #print(f'top_level_clauses: {top_level_clauses}')
    #print(f'all_clauses, Terminal: {[f'{i}:{a}' for a,b in all_clauses[1]]}')
    print(f'all_clauses: {[str(a) for a in all_clauses]}')
    print(f'top_level_clauses: {[str(a) for a in top_level_clauses]}')
    #return

    queue = PriorityQueue()
    memoTable = MemoTable(tokens)
    terminals = terminal_to_clause.values()

    # Get terminals
    """
    for prod in rule_manager.productions:
        for step in prod.input_steps:
            print(f'{prod.name}: "{step.name()}" \tis terminal: {isinstance(step, Terminal)}')
            if isinstance(step, Terminal):
                terminals.add(step.name())
    print(f'Terminals: {terminals}')
    """
    

    # Main parsing loop
    for startPos in range(len(tokens)-1, -1, -1):
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(f'PARSE startPos: {startPos}')
        #Add all terminals to the queue
        for t in terminals:
            queue.put(t)
        while not queue.empty():
            
            curr_clause = queue.get()
            print('|||||||||||||||||||||')
            print(f'PARSE queue curr_clause: {curr_clause}')
            memoKey     = MemoKey(curr_clause, startPos)
            match       = curr_clause.match(memoTable, memoKey, tokens)
            memoTable.add_match(memoKey, match, queue)
    
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(f'PARSE FINISHED memoTable: {memoTable}')
    s = __memoTable_to_state(rule_manager, top_level_clauses, memoTable)
    return s



