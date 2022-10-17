# Assignment 

- `Wire, Reg, Latch, Tri, TriOr, TriAnd` are assignable gates with dynamic inputs and single output 
- if there is no assignable gate between signal and its driver, a `Wire` will be automatically inserted during `<==` operator


```py 
a = UInt()          # no driver
b = UInt()          # no driver
c = a & b           # driver is not assignable 
a <== UInt(1)       # insert a Wire Gate 
c <== UInt(1)       # insert a Wire Gate, which stores two inputs: output of the And gate and UInt(1)
```

reference verilog 

```sv 
logic a;    // signal a
logic b;    // signal b
logic temp; // output of And gate
assign temp = a & b;  // And gate
always_comb begin 
    a = 1;  // Assignment to a
end 
always_comb begin 
    c = 1; 
    c = temp; // Assignment to c
end

``` 


| Function           | Description                                                          |
| ------------------ | -------------------------------------------------------------------- |
| `Wire(x)`          | connect a wire behind x if x is not assignable                       |
| `WireNext(x)`      | connect a wire after x and return new signal                         |
| `Reg(x)`           | a register whose input/output is x, reset/init is current value of x |
| `RegNext(x)`       | a register whose input is x, output is new signal                    |
| `Latch(x, enable)` | latch                                                                |



## Conditional 



## ClockDomain