# Quick Start 

`python -m pip install pyhgl`


## Syntax 


PyHGL adds some syntactic sugar to Python and uses a parser generated by [Pegen](https://github.com/we-like-parsers/pegen). PyHGL source files (with `.pyh` suffix) should be imported by a normal python file that has imported `pyhgl`.

```py
# in Adder.pyh 
from pyhgl.logic import *
...
# in main.py 
import pyhgl  
import Adder 
``` 

new operators and statements are listed below. 

### Operator


| Operator       | Description           |
| -------------- | --------------------- |
| `!x`           | LogicNot              |
| `x && y`       | LogicAnd              |
| `x \|\| y`     | LogicOr               |
| `x \|-> y`     | Imply(Assertion)      |
| `x >>> y`      | Sequential(Assertion) |
| `x <== y`      | Assignment            |
| `x[idx] <== y` | Partial Assignment    |
| `x <=> y`      | Connect               |


### One-line Decorator

```py 
# pyhgl
@decorator NAME:
    ...  
# python 
@decorator 
def NAME():
    ...
    return locals()
``` 

### When Statement 

```py 
when signal:
    ... 
elsewhen signal:
    ... 
otherwise:
    ... 
```

### Switch Statement

```py 
switch signal:
    once a, b:
        ... 
    once c: 
        ... 
    once ...:
        ...
```


## Example 

a Game of Life example comes from [Chisel Tutorials](https://github.com/ucb-bar/chisel-tutorial).

in `life.pyh`

```py 
from pyhgl.logic import *

@module Life(rows, cols):                               # declares a module `Life`
    cells     = Reg(Array.zeros(rows, cols)) @ Output   # a rows*cols array of 1-bit regs
    running   = UInt(0)                      @ Input    # control signal. if 1, run; if 0, write data
    wdata     = UInt(0)                      @ Input    # data to write
    waddr_row = UInt(rows)                   @ Input    # write address
    waddr_col = UInt(cols)                   @ Input    # write address

    def make_cell(row, col):                            # row and col are indexes
        neighbors = cells[[row-1,row,(row+1)%rows],     # select a 3*3 range circularly
                          [col-1,col,(col+1)%cols]]
        cell = neighbors[1,1]                           # select current cell
        neighbors[1,1] = UInt('3:b0')                   # 3 bits to count 8 neighbors
        count_neighbors = Add(neighbors, axis=(0,1))    # sum up the 3*3 array, return a 3-bit uint
        when !running:                                  # write data
            when waddr_row == row && waddr_col == col:  
                cell <== wdata 
        otherwise:                                      # next state
            when count_neighbors == 3:                  # n=3, becomes a live cell
                cell <== 1 
            elsewhen count_neighbors != 2:              # n!=3 and n!=2, dies
                cell <== 0 
                                                        # otherwise keep the state
    for i in range(rows):
        for j in range(cols):
            make_cell(i,j)                              # map rules on each cell


@task test_life(self, dut):                             # coroutine-based simulation tasks
                                                        # dut is a `Life` module instance
    @task set_mode(self, run: bool):                    # declares a task 
        setv(dut.running, run)                          # `setv` set value to signal/signals
        yield self.clock_n()                            # wait until next negedge of default clock
    @task write(self, row, col):                        # write `1` to a specific cell
        setv(dut.wdata, 1)
        setv(dut.waddr_row, row)        
        setv(dut.waddr_col, col)
        yield self.clock_n()
    @task init_glider(self):      
        # wait and execute these tasks sequentially                      
        yield write(3,3), write(3,5), write(4,4), write(4,5), write(5,4)
    @task show(self):
        cells_str = Map(lambda v: '*' if v==1 else ' ', getv(dut.cells))   
        for row in cells_str:
            print(' '.join(row)) 
        yield self.clock_n()
        
    yield set_mode(0), init_glider(), set_mode(1)
    for _ in range(20):
        yield show()


with Session() as sess:                                 # enter a `Session` before initialization
    life = Life(10,10)                                  # a 10*10 game of life
    sess.track(life)                                    # track all singals in the module
    sess.join(test_life(life))                          # wait until the task finishes
    sess.dumpVCD('life.vcd')                            # generate waveform to ./build/life.vcd
    sess.dumpVerilog('life.sv')                         # generate RTL to ./build/life.sv
    print(sess)                                         # a summary 
```

in `main.py`

```py
import pyhgl 
import life
```

run: `python main.py`