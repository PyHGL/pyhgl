# Gates 

In the **circuit abstraction**, hardware is described as a graph of connected components. 

- Operations on PyHGL signals usually generate a hardware gate.
- Directional assignment `<==` to a signal will automatically insert a `Wire` if the signal is not assignable.

## Functions 

- Unary functions: `Not`, `Bool`, `LogicNot`, `AndR`, `OrR`, `XorR`
- Binary functions: `Lshift`, `Rshift`, `Eq`, `Ne`, `Lt`, `Gt`, `Le`, `Ge`, `Floordiv`, `Mod`
- Multi-inputs functions: `And`, `Or`, `Xor`, `Nand`, `Nor`, `Nxor`, `Cat`, `Pow`, `LogicAnd`, `LogicOr`, `Add`, `Sub`, `Mul`

| Function           | Description      | Operator | Output Width       |
| ------------------ | ---------------- | -------- | ------------------ |
| `Not(x)`           | bitwise not      | `~`      | `len(x)`           |
| `Bool(x)`          | logic bool       |          | 1                  |
| `LogicNot(x)`      | logic not        | `!`      | 1                  |
| `AndR(x)`          | and reduce       |          | 1                  |
| `OrR(x)`           | or reduce        |          | 1                  |
| `XorR(x)`          | xor reduce       |          | 1                  |
| `Lshift(x,y)`      | left shift       | `<<`     | `len(x)`           |
| `Rshift(x,y)`      | right shift      | `>>`     | `len(x)`           |
| `Eq(x,y)`          | comparation      | `==`     | 1                  |
| `Floordiv(x, y)`   | divide           | `//`     | `len(x)`           |
| `Mod(x, y)`        | mod              | `%`      | `len(x)`           |
| `And(x,...)`       | bitwise and      | `&`      | `max(len(x), ...)` |
| `Nand(x,...)`      | bitwise nand     |          | `max(len(x), ...)` |
| `Cat(x, ...)`      | concatenation    |          | `sum(len(x), ...)` |
| `Pow(x, n)`        | bits duplication | `**`     | `len(x) * n`       |
| `LogicAnd(x, ...)` | logic and        | `&&`     | 1                  |
| `LogicOr(x, ...)`  | logic or         | `x || y`     | 1                  |
| `Add(x, ...)`      | add              | `+`      | `max(len(x), ...)` |
| `Add(x, ...)`      | add              | `+`      | `max(len(x), ...)` |
| `Mul(x, ...)`      | mul              | `*`      | `max(len(x), ...)` |

## Netlist 

Netlists are gate that is **assignable**. Unlike Verilog, registers and latches should be explicitly declared, while wires are usually implicitly generated.


| Function           | Description                                                          |
| ------------------ | -------------------------------------------------------------------- |
| `Wire(x)`          | connect a wire behind x if x is not assignable                       |
| `WireNext(x)`      | connect a wire after x and return new signal                         |
| `Reg(x)`           | a register whose input/output is x, reset/init is current value of x |
| `RegNext(x)`       | a register whose input is x, output is new signal                    |
| `Latch(x, enable)` | latch                                                                |
| `Wtri(x)`          | tri-state wire |  

### ClockDomain

`Reg` will get the default clock and reset signals, which can be set by `ClockDomain`. 

```py 
# negedge clk, asynchronous low-valid rst_n
with ClockDomain(clock=(clk,1), reset=(rst_n,0)):
    register = Reg(UInt[8](0))
```


## Examples



### FullAdder 


```py
@module FullAdder:
    a, b, cin = UInt.zeros(3)
    s = a ^ b ^ cin 
    cout = a & b | (a ^ b) & cin 
```


### D flip flop  

```py 
@module Register:
    data, clk, set, reset = UInt.zeros(4)
    nand1, nand2, nand3, nand4, Q, Qn = UInt.zeros(6)
    
    nand1 <== Nand(set, nand4, nand2)
    nand2 <== Nand(nand1, clk, reset)
    nand3 <== Nand(nand2, clk, nand4)
    nand4 <== Nand(nand3, data, reset)
    Q <== Nand(set, nand2, Qn)
    Qn <== Nand(Q, nand3, reset)
```
