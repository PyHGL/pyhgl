
# PyHGL

PyHGL is a Python-based Hardware Generation Language. Similar languages are: Verilog, Chisel, PyRTL, etc. PyHGL provides hardware modeling, simulation and verification in pure Python environment. Some features are:

- Pythonic syntax
- Three-state (0, 1, and X) logic
- Vectorized operation
- Asynchronous event-driven simulaton 
- Simulation tasks and concurrent assertions

--- 

- Documentation: [https://pyhgl.github.io/pyhgl/](https://pyhgl.github.io/pyhgl/)
- Similar Projects: [https://github.com/drom/awesome-hdl](https://github.com/drom/awesome-hdl)


# Install

```
python -m pip install pyhgl
``` 

# Example

Design and verify a 32-bit Carry Ripple Adder and a 64-bit Kogge Stone Adder. 

```py
from pyhgl.logic import *
 
@conf Config:
    @conf RippleCarry:
        w = 32
    @conf KoggeStone:
        w = 64
 
AdderIO = lambda w: Bundle(
    x   = UInt[w  ](0) @ Input,
    y   = UInt[w  ](0) @ Input,
    out = UInt[w+1](0) @ Output,
)
 
@module FullAdder: 
    a, b, cin = UInt([0,0,0])
    s    = a ^ b ^ cin 
    cout = a & b | (a ^ b) & cin 
 
@module RippleCarry:
    io = AdderIO(conf.p.w) 
    adders = Array(FullAdder() for _ in range(conf.p.w))
    adders[:,'a'  ] <== io.x.split()
    adders[:,'b'  ] <== io.y.split()
    adders[:,'cin'] <== 0, *adders[:-1,'cout']
    io.out <== Cat(*adders[:,'s'], adders[-1,'cout']) 
 
@module KoggeStone:
    io = AdderIO(conf.p.w) 
    P_odd = io.x ^ io.y
    P = P_odd.split()
    G = (io.x & io.y).split()
    dist = 1 
    while dist < conf.p.w:
        for i in reversed(range(dist,conf.p.w)): 
            G[i] = G[i] | (P[i] & G[i-dist])
            if i >= dist * 2:
                P[i] = P[i] & P[i-dist]
        dist *= 2 
    io.out <== Cat(0, *G) ^ P_odd
 
@task tb(self, dut, N): 
    for _ in range(N):
        x, y = setr(dut.io[['x','y']]) 
        yield self.clock_n() 
        self.AssertEq(getv(dut.io.out), x + y)
 
with Session(Config()) as sess:
    adder1, adder2 = RippleCarry(), KoggeStone()
    sess.track(adder1, adder2)  
    sess.join(tb(adder1, 100), tb(adder2, 200))
    sess.dumpVCD('Adders.vcd') 
    sess.dumpVerilog('Adders.sv') 
``` 
