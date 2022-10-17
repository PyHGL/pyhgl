# Signals 

PyHGL uses an object oriented model which components are mainly Signals and Gates. They construct a directed graph that represents digital circuits. 

- Signals' type is `Reader`, which contains a `SignalType` and a `SignalData` 
  - `SignalType` indicates signal type and bit width 
  - `SignalData` contains real data that is updated during simultion

```py 
from pyhgl.logic.hardware import UInt, Reader, SignalType, SignalData
u32_t = UInt[32]
a = u32_t()

print(isinstance(a, Reader))            # True
print(isinstance(a._type, SignalType))  # True 
print(isinstance(a._data, SignalData))  # True 
print(a._type is u32_t)                 # True
``` 



## UInt 

```py 
a = UInt('1111') 
a = UInt('4:d15')
a = UInt(15)
a = UInt(15, w=4)
a = UInt[4](15)
a = UInt[4]('1111')
```

## Enum 

```py 
sel_t = Enum['add', 'sub', 'shift']
```


## SInt 



## Struct 

```py
struct_t = Struct(
    a = UInt[3]             @2,
    b = (3,4) * UInt[4]     @7,
    c = UInt[1]             @13,
    d = Struct(
        x = UInt[1],
        y = UInt[2]
    )                       @14
)
```


## Vector

```py 
ram_t = UInt[32] * 100
``` 


## Type casting


## Array 

`Array` is not a hardware type. It is a tree-like data struct like [awkward-array](https://awkward-array.readthedocs.io/en/latest/). `Array` is used for vectorized operations and io containers. 


- combination of `list` and `dict` of `{name:value}`
- convenient slicing 
- operators on `Array` are vectorized and broadcast 

```py 
# 1-d vector 
Array([1,2,3,4])  
Bundle(1,2,3,4)
# 2-d vector
Array([[1,2],[3,4]])
# named array 
Array({'a':1, 'b':2})
Bundle(
    a=1,
    b=2
)

``` 


## Literals & Signal Type