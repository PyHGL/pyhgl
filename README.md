
# PyHGL

PyHGL is a simple and flexible **Py**thon based **H**ardware **G**eneration **L**anguage. PyHGL has rich features including:

- A few but necessary extended grammars to reduce grammar noises
- RTL and Gate Level hardware description, generates SystemVerilog files
- Support any two-valued logic circuit, including multi-clocks and combinational loop 
- Object-oriented modeling, easy to distinguish between synthetic and not synthetic 
- Asynchronous delay accurate event based simulation in pure Python 
- SVA like assertions for verification

--- 

- Documentation 
- Similar Projects 
  
  [https://github.com/drom/awesome-hdl](https://github.com/drom/awesome-hdl)


# Install

```
python -m pip install pyhgl
``` 

# Examples 

## N-bit Adder 

```py
from pyhgl.logic import *
import random

@module FullAdder:
    a, b, cin = UInt('0'), UInt('0'), UInt('0')
    s = a ^ b ^ cin 
    cout = a & b | (a ^ b) & cin 

@module AdderN(w: int):
    x = UInt[w](0)
    y = UInt[w](0)
    out = UInt[w](0)

    adders = Array(FullAdder() for _ in range(w))
    adders[:,'a'] <== x.split()
    adders[:,'b'] <== y.split()
    adders[1:, 'cin'] <== adders[:-1, 'cout']
    out <== Cat(adders[:,'s']) 
    cout = adders[-1, 'cout']

#--------------------------------- test ----------------------------------

with Session() as sess:
    w = 8
    mask = (1 << w) - 1 
    dut = AdderN(w)
    sess.track(dut.x, dut.y, dut.out)

    for _ in range(100):
        x = random.randint(0, mask)
        y = random.randint(0, mask)
        setv(dut.x, x) 
        setv(dut.y, y) 
        sess.run(100) 
        out = getv(dut.out)
        print(f'{x} + {y} = {out}\t{(x+y)&mask==out}') 

    sess.emitVCD('Adder.vcd')
    sess.emitVerilog('Adder.sv')
    sess.emitGraph('Adder.gv')
    sess.emitSummary()
``` 

## Vending Machine 


```py 
@module VendingMachine:
    nickel, dime, valid = UInt('0'), UInt('0'), UInt('0') 
 
    switch s:=EnumReg():
        once 'sIdle':
            when nickel: 
                s <== 's5'
            when dime:
                s <== 's10' 
        once 's5':
            when nickel:
                s <== 's10'
            when dime: 
                s <== 's15'
        once 's10':
            when nickel: 
                s <== 's15'
            when dime: 
                s <== 'sOk' 
        once 's15': 
            when nickel: 
                s <== 'sOk'
            when dime: 
                s <== 'sOk'
        once 'sOk':
            s <== 'sIdle'
            valid <== 1
```

