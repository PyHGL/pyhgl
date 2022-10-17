
from typing import Any, Dict, List, Set, Union, Tuple
import gmpy2
import re 

def width_infer(a: Union[int, gmpy2.mpz, gmpy2.xmpz], *, signed = False) -> int:
    """ return positive least number of bits that can represent a
    
    unsigned: 0 -> 1,  1 -> 1,  2 -> 2,  3 -> 2,  4 -> 3, ... 
    signed:   0 -> 1,  1 -> 2,  2 -> 3,  3 -> 3,  4 -> 4, ... 
             -1 -> 1, -2 -> 2, -3 -> 3, -4 -> 3, -5 -> 4, ...
    
    """ 
    if a == 0:
        return 1
    
    if not signed:
        if a > 0:
            return a.bit_length()
        else:
            raise Exception(f'{a} is not unsigned')             
    else:
        if a > 0:
            return a.bit_length() + 1
        else:
            return (-1-a).bit_length() + 1

def const_int(a: Union[int, gmpy2.mpz, gmpy2.xmpz]) -> int:
    return f"{width_infer(a)}'d{a}"


_str_split = re.compile(r"[:']")
_str_sub = re.compile(r"[^0-9a-z.:?'+-]") 
_hex2bin = lambda x: f'{int(x, 16):0>4b}' 

def str2int(a:str) -> Tuple[int, int]:
    """ initial value of Logic 
    
    - support bin/hex/dec 
    - support sign 
    - unsigned bin/hex has implict width
    
    ex. 8-bit unsigned binary
        8:1111_0000     
        b1111_0000   
        
    ex. hex & dec 
        h12ab
        hff
        xff
        d123
        d123
    
    ex. signed 
        12:+ f
        5:+hf 
        6:-hf 
        7:+hff   error
        -d5
        
    return:
        (value, width)
    FIXME overflow 
    TODO
    1>0011 
    1>8:ha
    x>32:hff
    """
    _a = a
    a = _str_sub.sub('', a.lower())
    width = None
    sign = None  
    radix = None
    r = _str_split.split(a) 
    if len(r) > 1:
        width = int(r[0]) 
        assert width > 0
        a = r[1]
    # signed or not 
    if a[0] == '+':
        sign = True  
        a = a[1:].strip()
    elif a[0] == '-':
        sign = False 
        a = a[1:].strip()
    # radix 
    if a[0] == 'd':
        radix = 10 
        a = a[1:]
    elif a[0] == 'b':
        radix = 2
        a = a[1:]
    elif a[0] == 'h':
        radix = 16 
        a = a[1:] 
    else:
        radix = 2
    
    # implict width 
    if radix == 2:
        width = width or len(a) 
    elif radix == 16:
        width = width or len(a) * 4 
    # to int 
    try:
        a = int(a, radix)
    except:
        raise TypeError(_a) 
    # get value, minimal width
    if sign is None:
        v, w = a, width_infer(a)
    elif sign == True:
        v, w = a, width_infer(a, signed=True)
    else:
        v, w = -a, width_infer(-a, signed=True)  
    width = width or w
    
    v = v & ((1 << width) - 1)
    return v, width



def _str2logic(a: str) -> Tuple[gmpy2.xmpz, gmpy2.xmpz, int]:
    """ fixed width and value
    bxx11 
    hffxx
    d123

    return: (value, x_value, width)
    """
    v = gmpy2.xmpz(0)
    x = gmpy2.xmpz(0) 
    if a[0] == 'b':
        a = a[1:]
    if a[0] in ['0','1','x']:
        for i, bit in enumerate(reversed(a)):
            if bit == '1':
                v[i] = 1 
            elif bit == 'x':
                x[i] = 1
        return v, x, i+1 
    elif a[0] == 'd':
        _v = int(a[1:]) 
        assert _v >= 0 
        w = width_infer(_v)
        v[0:w] = _v 
        return v, x, w 
    elif a[0] == 'h':
        for i, bit in enumerate(reversed(a[1:])):
            if bit == 'x':
                x[i*4:i*4+4] = 15  
            else:
                v[i*4:i*4+4] = int(bit, 16) 
        return v, x, (i+1)*4
    else:
        raise ValueError(a)


def str2logic(a:str) -> Tuple[gmpy2.xmpz, gmpy2.xmpz, int]:
    """ logic value with certain width
    
    - radix: bin/hex/dec 
    - unsigned bin/hex has implict width
    
    ex. 8-bit unsigned binary
        1111_0000     
        b1111_0000   
        
    ex. hex & dec 
        h12ab
        hff
        d123
        d123
        
    return:
        (value, x_value, width)

    32:b11xx
    32.x:hff
    32.1:d10
    """
    a = _str_sub.sub('', a.lower()) 
    if r:=re.fullmatch(r'(?:([0-9]+)(?:.([01x]))?[:])?([0-9a-z?]+)',a):
        width, prefix, body = r.groups() 
        if width is not None: 
            width = int(width)
    else:
        raise ValueError(a) 
    # body
    v, x, required_width = _str2logic(body) 
    if width is None or width == required_width:
        return v, x, required_width 
    # extend to width  
    assert width > required_width, f'{a} overflow'
    if prefix is None or prefix == '0':
        return v, x, width 
    elif prefix == '1':
        v[required_width:width] = -1 
        return v, x, width 
    elif prefix == 'x':
        x[required_width:width] = -1 
        return v, x, width 
    else:
        raise Exception()
    





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


