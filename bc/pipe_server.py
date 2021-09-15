import os
import select
import uuid
import json

from late.late import *

#https://levelup.gitconnected.com/inter-process-communication-between-node-js-and-python-2e9c4fda928d

IPC_FIFO_NAME_A = "../webv2/late_in"
IPC_FIFO_NAME_B = "../webv2/late_out"
HEADER_SIZE = 10

def get_message(fifo):
    pkg_length = int(os.read(fifo, 10))
    return os.read(fifo, pkg_length)



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
    matched = match(Productions(prodA), tokenize(input))

    if matched != None:
        vals = matched.fullStr()
        esr = matched.esrap(Productions(prodB))
        return {"output_source": esr}
    else:
        return {"error": "Source failed to satisfy ebnf"}

def process_msg(msg):
    '''Process message read from pipe'''
    print(f'msg: {msg}')
    ret = test_add_rename(msg)
    encoded = json.dumps(ret)

    header = bytearray(str(len(ret)).rjust(10,'0').encode(encoding='UTF-8',errors='strict'))
    return header+ret


def main():
    os.mkfifo(IPC_FIFO_NAME_A)

    try:
        fifo_a = os.open(IPC_FIFO_NAME_A, os.O_RDONLY | os.O_NONBLOCK)
        print('Pipe A ready')

        while True:
            try:
                fifo_b = os.open(IPC_FIFO_NAME_B, os.O_WRONLY)
                print("Pipe B ready")
                break
            except:
                # Wait until Pipe B has been initialized
                pass

        try:
            poll = select.poll()
            poll.register(fifo_a, select.POLLIN)

            try:
                while True:
                    timeout = 100
                    if (fifo_a, select.POLLIN) in poll.poll(timeout):
                        inMsg = get_message(fifo_a)
                        
                        print('----- Received from JS -----')
                        print("    " + inMsg.decode("utf-8"))

                        retMsg = process_msg(inMsg)
                        os.write(fifo_b, retMsg)


            finally:
                poll.unregister(fifo_a)
        finally:
            os.close(fifo_a)
    finally:
        os.remove(IPC_FIFO_NAME_A)
        os.remove(IPC_FIFO_NAME_B)

if __name__ == "__main__":
    main()