from .shared.expression import * 


class ProjectManager:
    def __init__(self, rManagers):
        self.rManagers = rManagers
        self.permanent_marks = set()
        self.temporary_marks = set()
        self.sorted_indexes  = []

    def addRuleManager(self, rManager: RuleManager):
        self.rManagers.append(rManager)

    """
        NOTE: Depth-first search
    """
    def processProductions(self):
        self.to_visit = set(range(len(self.rManagers)))
        self.name_to_index = {}
        
        for index in self.to_visit:
            name = self.rManagers[index].name
            self.name_to_index[name] = index
        #print(self.name_to_index)

        def __set_first(s: set):
            for e in s:
                break
            return e

        while len(self.to_visit) != 0:
            self.__visit(__set_first(self.to_visit))

        for index in self.sorted_indexes:
            productions = []
            noitcudorps = []
            
            for dep in self.rManagers[index].imports:
                depRM = self.getRuleManager(dep)
                productions = depRM.productions
                noitcudorps = depRM.noitcudorps
            self.rManagers[index].process((productions, noitcudorps))

    def __visit(self, index):
        if index in self.permanent_marks:
            return
        if index in self.temporary_marks:
            print(f'{bcolors.FAIL}ERROR: this project has dependency circles (not a directed acyclic graph){bcolors.ENDC}')
            assert False
        
        self.temporary_marks.add(index)
        for import_name in self.rManagers[index].imports:
            if import_name not in self.name_to_index:
                print(f'{bcolors.FAIL}ERROR: unknown import "{import_name}" is required!{bcolors.ENDC}')
                assert False
            self.__visit(self.name_to_index[import_name])

        self.temporary_marks.remove(index)
        self.permanent_marks.add(index)
        self.to_visit.remove(index)
        self.sorted_indexes.append(index)
    
    def getRuleManager(self, name: str):
        index = self.name_to_index[name]
        return self.rManagers[index]

