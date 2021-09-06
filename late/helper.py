from inspect import getframeinfo, stack

import builtins
import os
import sys

def print(message, sep=' ', end='\n', file=sys.stdout, flush=False):
    caller = getframeinfo(stack()[1][0])
    builtins.print(os.path.basename(caller.filename) + "(" + (caller.function + ": ").ljust(10, " ") + f'{caller.lineno:03d}' + "): \t" + str(message), sep=sep, end=end, file=file, flush=flush)
