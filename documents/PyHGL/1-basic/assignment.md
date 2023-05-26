# Conditional Assignment 

Conditional assignments are core semantics of the **register-transfer abstraction**. There is no `always` block and sensitive list in PyHGL, and all assignments are non-blocking.


## Condition Statements

- `when` statement is the special case of `switch` statement (same as `switch 1`). It sotres conditions in Python's local frame. These conditions will only influence the operator `<==`.  
- `switch` statement is more general, which is similar to the `case` statement in Verilog.

## Dynamic Enumerated Type 

The `Enum` type maps states to a specific encoding dynamically. Unlike `UInt`, a `Enum` type has variable bit-length. 

```py 
state_t = Enum['idle', ...]             # binary encoded enum type 
state_t = Enum['a', 'b', 'c']           # a 2-bit frozen enum type 
state_t = EnumOnehot['a','b','c',...]   # a dynamic onehot-encoded enum type 
```

VendingMachine example comes from [Chisel Tutorials](https://github.com/ucb-bar/chisel-tutorial)
```py 
@module VendingMachine:
    nickel = UInt(0) @ Input  
    dime   = UInt(0) @ Input  
    valid  = UInt(0) @ Output
    switch s:=Reg(EnumOnehot()):            # a onehot-encoded signal
        once 'sIdle':                       # record state 'sIdle'
            when nickel: s <== 's5'         # record state 's5'
            when dime  : s <== 's10'        # record state 's10'
        once 's5':
            when nickel: s <== 's10'
            when dime  : s <== 's15'        # record state 's15'
        once 's10':
            when nickel: s <== 's15'
            when dime  : s <== 'sOk'        # record state 'sOk'
        once 's15': 
            when nickel: s <== 'sOk'
            when dime  : s <== 'sOk'
        once 'sOk':
            s, valid <== 'sIdle', 1
```

