import unittest

from eq.early.late import *
from eq.shared.tokenize import *

class CheckUnit:
    """
        Match against the first and esrap with the second.
    """
    def checkRegex(self, ruleManagerA: RuleManager, ruleManagerB: RuleManager, input: str, outputRegex: str, begin = None):
        matched = match(ruleManagerA, tokenize(input), begin)
        #print(f'input: "{input}", matched: "{matched}", expected output regex: "{outputRegex}"')
        if matched == None:
            self.assertEqual(outputRegex, None)
        else:
            self.assertNotEqual(outputRegex, None)
            vals = matched.fullStr()
            esr = matched.esrap(ruleManagerA, ruleManagerB)
            reg = re.compile(outputRegex)
            r = reg.match(esr)
            t = False if r==None else (r.start() == 0)
            #print(f'input: {input}, esr: "{esr}", expected output regex: "{outputRegex}"')
            self.assertTrue(t)

    """
        Match against the first and esrap with the second.
    """
    def check(self, ruleManagerA: RuleManager, ruleManagerB: RuleManager, input: str, output: str, beginRules = None):
        matched = match(ruleManagerA, tokenize(input), beginRules)
        #print(f'input: "{input}", matched: "{matched}", expected output: "{output}"')
        if matched == None:
            print(f'{bcolors.FAIL}ERROR: MATCH WAS NONE!{bcolors.ENDC}')
            self.assertEqual(output, None)
        else:
            self.assertNotEqual(output, None)
            vals = matched.fullStr()
            esr = matched.esrap(ruleManagerA, ruleManagerB)
            #print(f'input: "{input}", esr: "{esr}", expected output: "{output}"')
            self.assertEqual(esr, output)

    def runSubtests(self, ruleManagerA, ruleManagerB, inputs, outputs, beginRules = None):
        for input, output in zip(inputs, outputs):
            with self.subTest(input=input):
                self.check(ruleManagerA, ruleManagerB, input, output, beginRules)

    def runSubtestsRegex(self, ruleManagerA, ruleManagerB, inputs, outputs, beginRules = None):
        for input, output in zip(inputs, outputs):
            with self.subTest(input=input):
                self.checkRegex(ruleManagerA, ruleManagerB, input, output, beginRules)
