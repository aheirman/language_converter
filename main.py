from late.tests import *
from late.helper import print

unittest.main()

def __readFile(url: str):
    with open(url) as f:
         return f.readlines()

def __parseFile(lines: list[str]):
    pass

def handle(ufl: str):
    productions = __parseFile(__readFile(url))
