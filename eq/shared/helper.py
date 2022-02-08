from inspect import getframeinfo, stack

import builtins
import os
import sys


def print(message, sep=' ', end='\n', file=sys.stdout, flush=False):
    #caller = getframeinfo(stack()[1][0])
    #builtins.print(os.path.basename(caller.filename) + "(" + (caller.function + ": ").ljust(10, " ") + f'{caller.lineno:03d}' + "): \t" + str(message), sep=sep, end=end, file=file, flush=flush)
    builtins.print(message)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def containsAndTrue(dict, key):
    return (key in dict) and (dict[key] == True)

def containsNotNoneAndPresent(dict, key):
    return dict[key] if dict != None and (key in dict) else None

def containsAndTrueAny(dict, keys):
    for key in keys:
        if (key in dict) and (dict[key] == True):
            return True
    return False
