from enum import Enum


class TokenizeSettings(Enum):
    UNKNOWN              = -1,
    NORMAL               = 0,
    NORMAL_SETTINGS_INT  = 1,
    QUOTE_SINGLE         = 2,
    QUOTE_DOUBLE         = 3,
    SQUARE_BRACKET       = 4


def tokenize(input: str, interupts = ['+', '-', '*', ':', '/', '(', ')', '\n', ',', '{', '}', '\'', '→', '⇇', ';'], delete = []):
    tokens = []
    curr = ''
    status = TokenizeSettings.NORMAL
    old_char = ''
    escaped = False
    for c in input:
        if c in delete:
            continue
        
        if c == '\\' and not escaped:
            escaped = True
            #print('NOW ESCAPED')
            continue
        #tokens_txt = ' '.join(map(str, tokens))
        #print(f'status: {status}, escaped: {escaped}, curr: "{curr}" adding char: ' + c)
        
        if status == TokenizeSettings.NORMAL_SETTINGS_INT and c == '{':
            tokens.append(curr)
            status = TokenizeSettings.NORMAL
            curr = '{'
        elif status == TokenizeSettings.NORMAL_SETTINGS_INT and c == '}':
            tokens.append(curr)
            tokens.append('}')
            status = TokenizeSettings.NORMAL
            curr = ''
        elif (status in [TokenizeSettings.NORMAL, TokenizeSettings.NORMAL_SETTINGS_INT] and c in [' ', '\n']): 
            tokens.append(curr)
            if c == '\n':
                tokens.append(c)
            status = TokenizeSettings.NORMAL
            curr = ''
        elif (status == TokenizeSettings.NORMAL and c == '"' and not escaped):
            tokens.append(curr)
            curr = c
            status = TokenizeSettings.QUOTE_DOUBLE
        elif (status == TokenizeSettings.NORMAL and c == '['):
            tokens.append(curr)
            curr = c
            status = TokenizeSettings.SQUARE_BRACKET
        elif (status == TokenizeSettings.NORMAL and c == '-' and (not old_char in "0123456789")):
            curr += c
        elif (status == TokenizeSettings.QUOTE_DOUBLE):
            curr += c
            if c == '"' and not escaped:
                status = TokenizeSettings.NORMAL
        elif (status == TokenizeSettings.SQUARE_BRACKET):
            curr += c
            if c == ']':
                status = TokenizeSettings.NORMAL_SETTINGS_INT
        elif ((status == TokenizeSettings.NORMAL) and (c in interupts)):
            tokens.append(curr)
            tokens.append(c)
            curr = ''
        elif (status == TokenizeSettings.NORMAL and c not in ['"', '[']) or status == TokenizeSettings.NORMAL_SETTINGS_INT:
            curr += c
        else:
            print(f'ERROR: status: {status}')
            assert False
        old_char = c
        escaped = False

    if(curr != ''):
        tokens.append(curr)
    
    tokens = [tok for tok in tokens if (len(tok) != 0)]
    return tokens

def tokenize_c(input: str):
    return tokenize(input, delete = ['\n'])


def tokenizeFromJson(code: list):
    tokens = []
    #print(f'code: {code}')
    for line in code:
        #print(f'line: {line}')
        for obj in line['words']:
            #print(f'obj: {obj}')
            #If type is 1 it;s modified
            if (obj['style'] == 1):
                newTokens = tokenize(obj['word'])
                tokens.extend(newTokens)
            else:
                tokens.append(obj['word'])
    
    tokens = [tok for tok in tokens if (len(tok) != 0)]
    return tokens
