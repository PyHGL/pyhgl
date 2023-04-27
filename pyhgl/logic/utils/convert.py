
from typing import Any, Dict, List, Set, Union, Tuple, Literal
import gmpy2
import re 


def width_infer(a: Union[int, gmpy2.mpz, gmpy2.xmpz], *, signed = False) -> int:
    """ return positive least number of bits that can represent a
    
    unsigned: 0 -> 1,  1 -> 1,  2 -> 2,  3 -> 2,  4 -> 3, ... 
             -1 -> 1, -2 -> 2, -3 -> 3, -4 -> 3, -5 -> 4, ...
    signed:   0 -> 1,  1 -> 2,  2 -> 3,  3 -> 3,  4 -> 4, ... 
             -1 -> 1, -2 -> 2, -3 -> 3, -4 -> 3, -5 -> 4, ...
    
    """ 
    if a == 0:
        return 1
              
    if signed:
        if a > 0:
            return a.bit_length() + 1
        else:
            return (-1-a).bit_length() + 1
    else:
        if a > 0:
            return a.bit_length()
        else:
            return (-1-a).bit_length() + 1  


_str_split = re.compile(r"[:']")
_str_sub = re.compile(r"[^0-9a-z.:?'+-]") 
_hex2bin = lambda x: f'{int(x, 16):0>4b}' 

_logic_split = re.compile(r"(.+?)(\.[01x])?(:.+)?$")

def str2logic(a: str) -> Tuple[int, int, int]:
    """ python str to 3-valued logic, '0' -> 00, '1' -> 10, 'x' -> 01

    - return: (value, unknown, width)
    - format: {width}.{prefix}:{sign}{radix}{value}
    - ex. `32.x:-hffff_xxxx_0000` -> `xxxxffffxxxx0000`

    - '?' regard as 'x'
    - split by : or '
    - default width:
        - '-111000' is 6,  
        - '-d1'  is 2 
        - '-hff' is 8,  
    - overflow value is clipped
    - 'x' is only allowed in unsigned bin/hex

    ex. 001x    - 4bit  
        hff     - 8bit  
        -d1     - 2bit 
        +1      - 2bit 
        8:d5    - 8bit 
        8:-hff  - overflow, return 1

        8.x:d1  - xxxx_xxx1 
        8.1     - 1111_1111 
        8.0:b11 - 0000_0011

    TODO return negative value with width, don't cut off here
    """
    _a = a 
    a = _str_sub.sub('', a.lower())  
    a = re.sub(r'\?', 'x', a)
    a = re.sub(r"'", ':', a)
    assert a, f"invalid value {_a}"
    # 8.x:d1 -> ('8', '.x', ':d1')
    r = _logic_split.match(a).groups()
    if r[1] is None and r[2] is None:
        body = r[0]
        prefix = None
        width = None
    elif r[1] is None:
        width = int(r[0])
        prefix = None 
        body = r[2][1:]
    elif r[2] is None:
        width = int(r[0])
        prefix = r[1][-1]
        body = prefix
    else:
        width = int(r[0])
        prefix = r[1][-1]
        body = r[2][1:]

    # radix and sign 
    radix = None 
    sign = None
    if body.startswith('+'):
        sign = True 
        body = body[1:]
    elif body.startswith('-'):
        sign = False 
        body = body[1:]
    if body.startswith('d'):
        radix = 10 
        body = body[1:] 
    elif body.startswith('b'):
        radix = 2 
        body = body[1:]
    elif body.startswith('h'):
        radix = 16 
        body = body[1:]
    else:
        radix = 2 

    if radix == 10:
        assert 'x' not in body, f'{_a} unknown value only allowed in unsigned bin/hex'
    if sign is not None:
        assert 'x' not in body, f'{_a} unknown value only allowed in unsigned bin/hex'
        assert prefix is None, f'{_a} signed value no need for prefix'

    # defalt width
    default_width = None
    if radix == 2:
        default_width = len(body)
    elif radix == 16:
        default_width = len(body) * 4 

    # get value 
    v, x = _str2logic(body, radix=radix) 
    # sign 
    if sign is None:
        default_width = default_width or width_infer(v)
    else:
        v = v if sign else -v
        default_width = default_width or width_infer(v, signed=True)
    # extend
    width = width or default_width 
    # just clip
    if width <= default_width or prefix is None or prefix == '0':
        mask = gmpy2.bit_mask(width)
        v = int(v & mask) 
        x = int(x & mask)
        return v, x, width
    else: # prefix is '1' or 'x', extend
        mask = gmpy2.bit_mask(width-default_width) << default_width 
        if prefix == '1':
            v = int(v | mask)
        else:
            x = int(x | mask)
        return v, x, width


def _str2logic(a: str, radix: int = 2) -> Tuple[int, int]: 
    """ return (v,x)
    """
    v = gmpy2.xmpz(0)
    x = gmpy2.xmpz(0) 
    if radix == 2:
        for i, bit in enumerate(reversed(a)):
            if bit == '1':
                v[i] = 1 
            elif bit == 'x':
                x[i] = 1 
            elif bit != '0':
                raise ValueError(f'{a}')
        return int(v), int(x)
    elif radix == 16:
        for i, bit in enumerate(reversed(a)):
            if bit == 'x':
                x[i*4:i*4+4] = 15  
            else:
                v[i*4:i*4+4] = int(bit, 16) 
        return (int(v), int(x))
    else: # radix == 10 
        return int(a), 0



def logic2str(v, x, width: int = None, prefix: bool=True, radix: Literal['b','h'] = 'h') -> str:
    """ turn logic value to verilog style literal 

    ex. 4'bxx11, 16'hffff

    - negative value is not allowed 
    - prefix: 8'h, 4'b
    - radix: binary or hex 
    - width: if None, use maximum width of v and x 
    - radix: use hex only if radix == 'h' and no unknown state
    """
    v = gmpy2.mpz(v) 
    x = gmpy2.mpz(x) 
    if width is None:
        assert v >= 0 and x >= 0, 'negative value without width not supported'
        width = max(width_infer(v), width_infer(x))
        
    v = v & gmpy2.bit_mask(width)
    x = x & gmpy2.bit_mask(width)
    
    if radix == 'h' and x == 0:     # return hex only if no `x`
        ret = hex(v)[2:]
        return f"{width}'h{ret}" if prefix else ret
    elif x == 0:                    # return binary
        ret = bin(v)[2:]
        return f"{width}'b{ret}" if prefix else ret 
    else:
        ret = []
        for i in range(max(width_infer(v), width_infer(x))):
            if x[i]:
                ret.append('x')
            else:
                ret.append('1' if v[i] else '0') 
        if prefix:
            ret.append(f"{width}'b")
        return ''.join(reversed(ret))


def logic2bin(v, x, width: int) -> str:
    """ fixed width, without prefix 
    ex. 0011, 0000xxxx
    """
    v = gmpy2.mpz(v) 
    x = gmpy2.mpz(x)
    ret = []
    for i in range(width):
        if x[i]:
            ret.append('x')
        else:
            ret.append('1' if v[i] else '0') 
    return ''.join(reversed(ret))

def logic2hex(v, x, width: int) -> str:
    mask = gmpy2.bit_mask(width)
    v = v & mask 
    x = x & mask
    ret = []
    for i in range((width+3)//4):
        vi = v[i*4:(i+1)*4]
        xi = x[i*4:(i+1)*4]
        if xi:
            ret.append('x')
        else:
            ret.append(hex(vi)[-1])
    return ''.join(reversed(ret))



def uint2sint(value: int, width: int) -> int:
    """ return real value of 2's complement
    """
    msb = 1 << (width-1)
    # 1 ext
    if value & msb:
        return value | ((-1) << width) 
    # 0 ext
    else:
        return value & (msb - 1)


def int2str(value: int, width: int, *, radix = 'b') -> str:
    """ return str of integer with out prefix
    radix: 
        b: bin 
        h: hex 
        u: unsigned dec
        s: signed dec
    """
    value = int(value)
    value = value & ( (1 << width) - 1)
    if radix == 'b':
        return ('{:0>'+str(width)+'b}').format(value)
    elif radix == 'h':
        return ('{:0>'+str((width + 3)//4)+'X}').format(value)
    elif radix == 'u':
        return str(value)
    elif radix == 's':
        str(uint2sint(value, width))
    else:
        raise Exception(f'unknow radix {radix}')
        
        

def binary2onehot(v: int) -> int:
    return 1 << v

def binary2gray(v: int) -> int:
    return v ^ (v >> 1)

def gray2binary(v: int) -> int:
    mask = v 
    while mask: 
        mask >>= 1 
        v ^= mask 
    return v 


def str2bitpat(a: str) -> List[str]:
    """
    1?10 
    b1?10 
    
    h?f 
    8:b??11 
    0>8:b??11   # 0 ext
    ?>8:b1      # ? ext
    """
    _a = a
    a = _str_sub.sub('', a.lower())
    width = None  
    radix = 2 
    ext = '0'
     
    r = _str_split.split(a)
    if len(r) > 1:
        width = int(r[0])
        a = r[1]
    if a[0] == 'b':
        radix = 2 
        a = a[1:] 
    elif a[1] == 'b':
        radix = 2 
        ext = a[0]
        a = a[2:]
    elif a[0] == 'h':
        radix = 16 
        a = a[1:] 
    elif a[1] == 'h':
        radix = 16 
        ext = a[0]
        a = a[2:]
        
    assert ext in '01?'
    
    if radix == 2:
        width = width or len(a) 
        for i in a:
            assert i in '01?'
    elif radix == 16:
        width = width or len(a) * 4    
        temp = []
        for i in a:
            if i == '?':
                temp.append('????')
            else:
                temp.append(_hex2bin(i))
        a = ''.join(temp)
    
    _next = width - len(a)
    if _next > 0:
        a = ext * _next + a 
    elif _next < 0:
        a = a[-width:] 
    
    return a.lstrip('0')
        

def split64(v: gmpy2.xmpz, w: int) -> List[int]:
    """ split big int to int64s
    """
    ret = []
    for idx in range((w + 63) // 64 ):
        ret.append(v[idx*64:(idx+1)*64])
    return ret 

def merge64(l: List[int]) -> gmpy2.xmpz:
    ret = gmpy2.xmpz(0)
    for idx, v in enumerate(l):
        ret[idx*64:(idx+1)*64] = int(v)
    return ret 

    
def mem_split(v: int, shape: Tuple[int,...]) -> List:
    """ TODO nd array
    """
    ret = []
    v = gmpy2.mpz(v)
    n, width = shape 
    for idx in range(n):
        ret.append(v[idx*width:(idx+1)*width])
    return ret
        
def mem2str(v: int, x: int, shape: Tuple[int]):
    """ ex. [8'hff, 8'hab]
    """
    v = mem_split(v, shape)
    x = mem_split(x, shape)
    ret = []
    for vi, xi in zip(v,x):
        ret.append(f"{shape[-1]}'h{logic2hex(vi, xi, width=shape[-1])}")
    return ret

def bin2mem(data: bytes, n: int = 1) -> list:
    """ data: bytes, n: memory data of n bytes 
    """
    ret = []
    for i in range((len(data) + n - 1)//n):
        ret.append(int.from_bytes(data[i*n:i*n+n], 'little'))
    return ret

_number = re.compile(r'\s*([0-9a-zA-Z]+).*')


def memory_init(s: str, xlen: int=32, radix=16) -> gmpy2.xmpz:
    """
    ex: memory_init('''
        5678abcd   # line1
        ffff0000   # line2
    ''', xlen=32,  radix=16)
    """
    idx = 0 
    ret = gmpy2.xmpz(0)
    for data in s.splitlines():  
        if r:=_number.match(data):
            v = int(r.groups()[0], radix) 
            ret[idx:idx+xlen] = v
            idx += xlen 
    return ret 


