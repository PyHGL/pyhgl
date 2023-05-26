# Literals & Signals 

## Literals 

### Logic 

`Logic` is three-state (0, 1, and X) and has an arbitrary length. Two bits are used to represent a three-state value: `00` as zero, `10` as one, and both `01` and `11` as the unknown state.

```py 
Logic(1,2)                      # 0...x1 
Logic(3)                        # 0...11
Logic(-1)                       # 1...1
Logic('xx')                     # 0...xx 
Logic('x1x1') | Logic('1x1x')   # 0...1111
```

For convenience, `==` and `!=` between `Logic` and other literals return a python `bool`. 

```py 
Logic('11') == 3            # True 
Logic('x1') == 3            # False 
Logic(0,3)  == Logic(3,3)   # True
```

Explicit `Logic` literal are usually unnecessary. 

### BitPat 

`BitPat` is three-state (0, 1, and don't care) and has a fixed length. It is only used in comparation.

```py 
BitPat('11??')          # 4'b11??
```

### strings 

Python strings are usually  converted into `Logic` automatically. 

```py
'0011_1100'     # 8'b00111100 
'4:b111111'     # 4'b1111 
'8:hab'         # 8'hab 
'-d3'           # 3'b101
'4:-d3'         # 4'b1101 
'16.x:habc'     # 16'hxabc
```

## Signals

PyHGL models digital circuits as a direct graph of `Gate` and `SignalData` nodes. `Writer` is the edge from gate to data, while `Reader` is the edge from data to gate. 

- Signals are `Reader`, which contains a `SignalType` and a `SignalData` 
  - `SignalType` indicates signal type and bit width. 
  - `SignalData` contains the real three-state singal value that is unique for each signal. 
- Type castings are only allowed for same bit length.


### UInt 

`UInt[w]` is the most basic signal type, similar as `logic [w-1:0]` in SystemVerilog. A `SignalType` is callable and return a signal instance.

```py
# lines below are the same
UInt[8](1)       
UInt[8]('1')
UInt('8:b1')
UInt(1, w=8)
```

### Other

```py 
uint8_t  = UInt[8]          # signal type: 8-bit unsigned integer
sint32_t = SInt[32]         # signal type: 32-bit signed integer
vector_t = 4 ** uint8_t     # signal type: packed array of 4x8 bits
mem_t    = MemArray[1024,8] # signal type: unpacked array of 1024x8 bits
struct_t = Struct(
    a = vector_t  @ 0,      # field 'a' starts from 0
    b = sint32_t  @ 0,      # field 'b' starts from 0 
    c = UInt[32]  @ 32,     # field 'c' starts from 32
)                           # signal type: 64-bit structure
x = struct_t({
    'a': [1,2,3,4], 
    'c': Logic('xxxx1100'),
})                          # signal x has an initial value of 64'hxc04030201
y = UInt[64](x)             # type casting        
``` 





## Array 

`Array` is not a hardware type, but a tree-like data struct like [awkward-array](https://awkward-array.readthedocs.io/en/latest/). `Array` is used for vectorized operations and module ports. 


- `Array` can be either a named array like `Dict[str, Any]` or a list `List[Any]`
- supports advanced slicing 
- operators on `Array` are vectorized and broadcast 
- use function `Bundle` to generate a named array 


```py 
# 1-d vector 
Array([1,2,3,4])  
Bundle(1,2,3,4)
# 2-d vector
Array([[1,2],[3,4]], recursive=True)
# named array 
Array({'a':1, 'b':2})
Bundle(a=1, b=2)
``` 


## Useful Functions 

```py 
Array.zeros(2,3)  # Array([[0,0,0],[0,0,0]], recursive=True)
UInt.zeros(2,3)   # Array([[UInt(0), UInt(0), UInt(0)],[UInt(0),UInt(0),UInt(0)]], recursive=True)
Logic(-1).to_bin(4)   # '1111'
Logic(-1).to_hex(16)  # 'ffff'
Logic(-1).to_int(1)   # -1
```