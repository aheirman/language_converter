from queue import PriorityQueue

class MemoKey:
    def __init__(production_uuid, startPos):
        self.production_uuid = production_uuid
        self.startPos = startPos

def parse(ruleManager: RuleManager, tokens: List[String]) -> MemoTable:
    queue = PriorityQueue()
    memoTable = MemoTable(tokens)
    terminals = [] # todo

    #Add
    for prod in ruleManager.productions:
        for step in prod.input_steps:
            print(f'{prod.name}: {step.name}')
    
