import os
import select
import uuid
import json
import socket

from late.late import *
import late.read



def test_mul_distributivity(input):
    uuids = [uuid.uuid4() for i in range(10)]

    prodA = Productiongenerator.createAllProductions([
        ([uuids[0]], 'calculation', 'term'),
        ([uuids[1]], 'term', 'number "*" "(" term "+" term ")"'),
        ([uuids[2]], 'term', 'number'),
        ([uuids[3]], 'number', '[0-9]')])
    prodB = Productiongenerator.createAllProductions([
        ([uuids[0]], 'calculation', 'term'),
        ([uuids[4]], 'term', '"(" number{"id": 0} "*" "(" term{"id": 1} ")" ")" "+" "(" number{"id": 0} "*" "(" term{"id": 2} ")" ")"', uuids[1]),
        ([uuids[2]], 'term', 'number'),
        ([uuids[3]], 'number', '[0-9]')])
    
    matched = match(Productions(prodA), tokenizeFromJson(input))

    if matched != None:
        vals = matched.fullStr()
        esr = matched.esrap(Productions(prodB))
        return {"output_source": esr}
    else:
        return {"error": "Source failed to satisfy ebnf"}

def python(code):
    #matched = late.read.parse('./languages/python/ebnf', './languages/python/python.peg', tokenizeFromJson(code))
    prods = late.read.getMetaIrProductions('./languages/python/ebnf')
    matched = match(prods, tokenizeFromJson(code))

    if matched != None:
        vals = matched.fullStr()
        esr = matched.esrap(prods)
        return {"output_source": esr}
    else:
        return {"error": "Source failed to satisfy ebnf"}


def handleInput(input):
    allData = json.loads(input)
    print(allData)
    metaData = allData['metadata']
    code     = allData['code']

    lang = metaData['language']

    match lang:
        case 'mul':
            return test_mul_distributivity(code)
        case 'python':
            return python(code)
        case _:
            print(f'ERROR wrong language: {lang}')


    

def init():
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    host = 'localhost'
    print(f'socket.gethostname(): {host}')
    port = 12350
    s.bind((host, port))
    backlog = 5
    s.listen(backlog)
    

    while True:
        c, addr = s.accept()
        try:
            print(f'got connection from {addr}')
            received = c.recv(4096).decode("utf-8")
            print(f'received: {received}')
            ret = handleInput(received)
            print(f'ret: {ret}')
            c.send(json.dumps(ret).encode(encoding='UTF-8',errors='strict'))
        finally:
            c.close()
init()