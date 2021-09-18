import os
import select
import uuid
import json
import socket

from late.late import *






def test_add_rename(input):
    uuids = [uuid.uuid4() for i in range(10)]
    prodA = Productiongenerator.createAllProductions([
        ([uuids[0]], 'calculation', 'term'),
        ([uuids[1]], 'term', 'number "+" term'),
        ([uuids[2]], 'term', 'number'),
        ([uuids[3]], 'number', '[0-9]')])
    prodB = Productiongenerator.createAllProductions([
        ([uuids[0]], 'calculation', 'term'),
        ([uuids[1]], 'term', 'number "plus"{"pad": true} term'),
        ([uuids[2]], 'term', 'number'),
        ([uuids[3]], 'number', '[0-9]')])
    matched = match(Productions(prodA), tokenizeFromJson(json.loads(input)))

    if matched != None:
        vals = matched.fullStr()
        esr = matched.esrap(Productions(prodB))
        return {"output_source": esr}
    else:
        return {"error": "Source failed to satisfy ebnf"}

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
            ret = test_add_rename(received)
            print(f'ret: {ret}')
            c.send(json.dumps(ret).encode(encoding='UTF-8',errors='strict'))
        finally:
            c.close()
init()