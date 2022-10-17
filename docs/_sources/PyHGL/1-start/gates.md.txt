# Gates 

- operations on signals usually generate a gate and return a new signal 
- gate is the driver of its output signals. a signal cannot have multiple drivers
- most verilog operators are supported
- global functions like `Not, And, Add` are more general than python operators 
- python literals are usually turned into signals automatically

## PyHGL Functions 

```py 
from pyhgl.logic import * 

a = UInt('0000')
b = a + 1           # add gate of 2 inputs
b = Add(a, 1)       # add gate of 2 inputs
b = Add(a, 1, 1)    # add gate of 3 inputs
Add([1,2,3], [4,5,6])  # return Array(s0, s1, s2)\
Add([[1,2,3], [4,5,6]])   # sum all signals
Add([[1,2,3], [4,5,6]], axis=0) # numpy-like

``` 


| Function | Description       | Operator |
| -------- | ----------------- | -------- |
| And      | bitwise and, zext | `&`      |
| Not      |
| Or       |
| Nand     |
| Nor      |
| Xor      |
| Nxor     |
| AndR     |
| XorR     |
 


## Examples 


### FullAdder 


```py
@module FullAdder:
    a, b, cin = UInt('0'), UInt('0'), UInt('0')
    s = a ^ b ^ cin 
    cout = a & b | (a ^ b) & cin 
```


### D flip flop  

```py 
@module Register:
    data = Input(UInt())
    clk = Input(UInt()) 
    set = Input(UInt())
    reset = Input(UInt())
    
    nand1, nand2, nand3, nand4, Q, Qn = (UInt() for _ in range(6)) 
    nand1 <== Nand(set, nand4, nand2)
    nand2 <== Nand(nand1, clk, reset)
    nand3 <== Nand(nand2, clk, nand4)
    nand4 <== Nand(nand3, data, reset)
    Q <== Nand(set, nand2, Qn)
    Qn <== Nand(Q, nand3, reset)
```
