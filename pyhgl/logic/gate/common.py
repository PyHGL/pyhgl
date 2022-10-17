####################################################################################################
#
# PyHGL - A Python-embedded Hardware Generation Language 
# Copyright (C) 2022 Jintao Sun
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
####################################################################################################


from __future__ import annotations
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union

import inspect 
import random

from pyhgl.array import *
from pyhgl.logic.hardware import *
import pyhgl.logic.utils as utils


def _align(*args: Reader) -> list:  
    """ return aligned signals
    """
    args = [Signal(i) for i in args]
    widths = [len(i) for i in args] 
    assert min(widths) > 0, 'signal variable length'
    max_w = max(widths)  
    return [Cat(s, UInt[max_w-w](0)) if w != max_w else s for s, w in zip(args, widths)]
    
    
class _GateNto1(Gate):
    """ 
    Args:
        args:
            any Logic Signals | Immds
        width:
            output width, default width of first input
    Return: 
        UInt Signal 
    """
    
    __slots__ = 'inputs', 'output'
    
    def __head__(self, *args: Reader, width: int = None, id: str=''):
        
        if len(args) < 1: 
            raise ValueError('at least 1 signal')
        
        self.id: str = id or self.id
        
        self.inputs: List[Reader] = []
        
        for i in args:
            if not isinstance(i._data, LogicData):
                raise ValueError(f'{i} is not logic signal')
            self.inputs.append(self.read(i))
            
        ret_width = width or len(self.inputs[0])
        ret = UInt[ret_width](0)
        self.output = self.write(ret)
        return ret
    
    def _verilog_op(self, *args: str) -> str:
        return ''
    
    def emitVerilog(self, v: verilogmodule.Verilog) -> str:
        """ 
        ex.
            assign y = a & b & 3'd5;
            assign #1 y = a & b;
        """
        x = [self.get_name(i) for i in self.inputs]
        y = self.get_name(self.output)
        op = self._verilog_op(*x) 
        
        if self._sess.verilog.emit_delay:
            delay = '#' + str(self.timing["delay"]) 
        else:
            delay = ''
        return f'assign {delay} {y} = {op};'
        
        

    

#-------------
# bitwise gate
#-------------

@dispatch('Not', Any, None)
class _Not(_GateNto1):
    
    id = 'Not'
    
    def __head__(self, x: Reader, id: str = ''):
        return super().__head__(x, id=id)
        
    def forward(self):
        self.output._setval(~self.inputs[0]._getval(), dt=self.delay)

    def _verilog_op(self, x: str) -> str:
        return f'~ {x}' 
    
    
@dispatch('And', Any, [Any, None])
class _And(_GateNto1):
        
    id = 'And'
    
    def __head__(self, *args: Reader, id: str = ''):
        return super().__head__(*_align(*args), id=id)
    
    def forward(self): 
        v = self.inputs[0]._getval()
        for i in self.inputs[1:]:
            v &= i._getval() 
        self.output._setval(v, self.delay)

    def _verilog_op(self, *args: str) -> str:
        return ' & '.join(args)
    
    
@dispatch('Or', Any, [Any, None])
class _Or(_GateNto1):
    
    id = 'Or'
    
    def __head__(self, *args: Reader, id: str = ''):
        return super().__head__(*_align(*args), id=id)
        
    def forward(self):
        v = self.inputs[0]._getval()
        for i in self.inputs[1:]:
            v |= i._getval() 
        self.output._setval(v, self.delay)

    def _verilog_op(self, *args: str) -> str:
        return ' | '.join(args)


@dispatch('Xor', Any, [Any, None])
class _Xor(_GateNto1):
    
    id = 'Xor'
    
    def __head__(self, *args: Reader, id: str = ''):
        return super().__head__(*_align(*args), id=id)
        
    def forward(self):
        v = self.inputs[0]._getval()
        for i in self.inputs[1:]:
            v ^= i._getval() 
        self.output._setval(v, self.delay)

    def _verilog_op(self, *args: str) -> str:
        return ' ^ '.join(args)


    
@dispatch('Nand', Any, [Any, None])
class _Nand(_GateNto1):
    
    id = 'Nand'
    
    def __head__(self, *args: Reader, id: str = ''):
        return super().__head__(*_align(*args), id=id)
        
    def forward(self):
        v = self.inputs[0]._getval()
        for i in self.inputs[1:]:
            v &= i._getval() 
        self.output._setval(~v, dt=self.delay)

    def _verilog_op(self, *args: str) -> str:
        return f'~({" & ".join(args)})'
    
    
@dispatch('Nor', Any, [Any, None])
class _Nor(_GateNto1):
    
    id = 'Nor'
    
    def __head__(self, *args: Reader, id: str = ''):
        return super().__head__(*_align(*args), id=id)
        
    def forward(self):
        v = self.inputs[0]._getval()
        for i in self.inputs[1:]:
            v |= i._getval() 
        self.output._setval(~v, dt=self.delay)

    def _verilog_op(self, *args: str) -> str:
        return f'~({" | ".join(args)})'


@dispatch('Nxor', Any, [Any, None])
class _Nxor(_GateNto1):
    
    id = 'Nxor'
    
    def __head__(self, *args: Reader, id: str = ''):
        return super().__head__(*_align(*args), id=id)
        
    def forward(self):
        v = self.inputs[0]._getval()
        for i in self.inputs[1:]:
            v ^= i._getval() 
        self.output._setval(~v, dt=self.delay)

    def _verilog_op(self, *args: str) -> str:
        return f'~({" ^ ".join(args)})'


@dispatch('AndR', Any) 
class _AndR(_GateNto1):
    
    id = 'AndR'
    
    def __head__(self, x: Reader, id: str = ''):
        return super().__head__(x, width=1, id=id)
    
    def forward(self):
        v: gmpy2.mpz = self.inputs[0]._getval()
        for i in range(len(self.inputs[0])):
            if v[i] == 0:
                self.output._setval(gmpy2.mpz(0), dt=self.delay)
                return 
        self.output._setval(gmpy2.mpz(1), dt=self.delay) 
        
    def _verilog_op(self, x: str) -> str:
        return f'& {x}'


@dispatch('NandR', Any) 
class _NandR(_GateNto1):
    
    id = 'NandR'
    
    def __head__(self, x: Reader, id: str = ''):
        return super().__head__(x, width=1, id=id)
    
    def forward(self):
        v = self.inputs[0]._getval()
        for i in range(len(self.inputs[0])):
            if v[i] == 0:
                self.output._setval(gmpy2.mpz(1), dt=self.delay)
                return 
        self.output._setval(gmpy2.mpz(0), dt=self.delay)

    def _verilog_op(self, x: str) -> str:
        return f'~& {x}'

    
@dispatch('OrR', Any) 
class _OrR(_GateNto1):
    
    id = 'OrR'
    
    def __head__(self, x: Reader, id: str = ''):
        return super().__head__(x, width=1, id=id)
    
    def forward(self):
        if self.inputs[0]._getval() == 0:
            self.output._setval(gmpy2.mpz(0), dt=self.delay) 
        else:
            self.output._setval(gmpy2.mpz(1), dt=self.delay) 
        
    def _verilog_op(self, x: str) -> str:
        return f'| {x}'


@dispatch('NorR', Any) 
class _NorR(_GateNto1):
    
    id = 'NorR'
    
    def __head__(self, x: Reader, id: str = ''):
        return super().__head__(x, width=1, id=id)
    
    def forward(self):
        if self.inputs[0]._getval() == 0:
            self.output._setval(gmpy2.mpz(1), dt=self.delay) 
        else:
            self.output._setval(gmpy2.mpz(0), dt=self.delay) 
        
    def _verilog_op(self, x: str) -> str:
        return f'~| {x}'
    

@dispatch('XorR', Any) 
class _XorR(_GateNto1):
    
    id = 'XorR'
    
    def __head__(self, x: Reader, id: str = ''):
        return super().__head__(x, width=1, id=id)
    
    def forward(self):
        v = self.inputs[0]._getval()
        ret = utils.parity(v)
        self.output._setval(gmpy2.mpz(ret), dt=self.delay) 
        
    def _verilog_op(self, x: str) -> str:
        return f'^ {x}' 
    

@dispatch('NxorR', Any) 
class _NxorR(_GateNto1):
    
    id = 'NxorR'
    
    def __head__(self, x: Reader, id: str = ''):
        return super().__head__(x, width=1, id=id)
    
    def forward(self):
        v = self.inputs[0]._getval()
        ret = utils.parity(v)
        self.output._setval(gmpy2.mpz(1-ret), dt=self.delay) 
        
    def _verilog_op(self, x: str) -> str:
        return f'~^ {x}' 


        
@dispatch('Cat', Any, [Any, None])
class _Cat(_GateNto1):
    """ Cat signals, return UInt
    """
    
    id = 'Cat'
    
    __slots__ = 'widths'
    
    def __head__(self, *args: Reader, id: str=''): 
        self.widths = [len(i) for i in args]
        return super().__head__(*args, width=sum(self.widths), id=id)
        
    def forward(self):
        v = gmpy2.xmpz(0) 
        start = 0
        for width, s in zip(self.widths, self.inputs): 
            end = start + width
            v[start:end] = s._getval()
            start = end
        self.output._setval(gmpy2.mpz(v), self.delay)
    
    def _verilog_op(self, *args: str) -> str:
        args = reversed(args)
        return f"{{{','.join(args)}}}"    
    
    def emitGraph(self, g):
        self._emit_graphviz(g, reversed(self.inputs), [self.output], self.id) 

    
@dispatch('Pow', Any, int)
def _pow(x: Reader, n: int, id: str='') -> Reader:
    if n < 1:
        raise ValueError('duplicate bits positive times')
    return _Cat(*(x for _ in range(n)), id=id)
    

#-------
# logic
#-------

@dispatch('Bool', Any) 
def _bool(x: Reader, id: str='') -> Reader: 
    if len(x) == 1:
        return x 
    else:
        return _OrR(x, id=id)


@dispatch('LogicNot', Any)
def _logicnot(x: Reader, id: str='') -> Reader:
    return _NorR(x, id=id)



@dispatch('LogicAnd', Any, [Any, None])
class _LogicAnd(_GateNto1):
        
    id = 'LogicAnd'
    
    def __head__(self, *args: Reader, id: str = ''):
        return super().__head__(*args, width=1, id=id)
    
    def forward(self): 
        v = all(i._getval() > 0 for i in self.inputs)
        self.output._setval(gmpy2.mpz(v), self.delay)

    def _verilog_op(self, *args: str) -> str:
        return ' && '.join(args) 
    
    
@dispatch('LogicOr', Any, [Any, None])
class _LogicOr(_GateNto1):
        
    id = 'LogicOr'
    
    def __head__(self, *args: Reader, id: str = ''):
        return super().__head__(*args, width=1, id=id)
    
    def forward(self):
        v = any(i._getval() > 0 for i in self.inputs)
        self.output._setval(gmpy2.mpz(v), self.delay)

    def _verilog_op(self, *args: str) -> str:
        return ' || '.join(args)
     
    
#------- 
# shift
#------- 
    
@dispatch('Lshift', Any, [UIntType, int]) 
class _Lshift(_GateNto1):
    
    id = 'Lshift'
    
    def __head__(self, a: Reader, b: Union[int, Reader], id: str = ''):
        return super().__head__(a, Signal(b), id=id)
    
    def forward(self):
        v = self.inputs[0]._getval() << self.inputs[1]._getval()
        self.output._setval(v, dt=self.delay) 
        
    def _verilog_op(self, a, b) -> str:
        return f'{a} << {b}' 
    
    
@dispatch('Rshift', Any, [UIntType, int]) 
class _Rshift(_GateNto1):
    
    id = 'Rshift'
    
    def __head__(self, a: Reader, b: Union[int, Reader], id: str = ''):
        return super().__head__(a, Signal(b), id=id)
    
    def forward(self):
        v = self.inputs[0]._getval() >> self.inputs[1]._getval()
        self.output._setval(v, dt=self.delay) 
        
    def _verilog_op(self, a, b) -> str:
        return f'{a} >> {b}' 
    
# TODO round shift, SInt shift
    

#-----------
# arithmetic
#-----------

@dispatch('Pos', Any, None)
class _Pos(_GateNto1):
    
    id = 'Pos'
    
    def __head__(self, x: Reader, id: str = ''):
        return super().__head__(x, id=id)
        
    def forward(self):
        self.output._setval(self.inputs[0]._getval(), dt=self.delay)

    def _verilog_op(self, x: str) -> str:
        return f'+ {x}' 
    
    
@dispatch('Neg', Any, None)
class _Neg(_GateNto1):
    
    id = 'Neg'
    
    def __head__(self, x: Reader, id: str = ''):
        return super().__head__(x, id=id)
        
    def forward(self):
        self.output._setval( - self.inputs[0]._getval(), dt=self.delay)

    def _verilog_op(self, x: str) -> str:
        return f'- {x}' 


@dispatch('Add', Any, [Any, None])
class _Add(_GateNto1):
        
    id = 'Add'
    
    def __head__(self, *args: Reader, id: str = ''):
        return super().__head__(*_align(*args), id=id)
    
    def forward(self):
        v = self.inputs[0]._getval()
        for i in self.inputs[1:]:
            v += i._getval() 
        self.output._setval(v, dt=self.delay)

    def _verilog_op(self, *args: str) -> str:
        return ' + '.join(args) 
    
    
@dispatch('AddFull', Any, [Any, None])
class _AddFull(_GateNto1):
        
    id = 'Add'
    
    def __head__(self, x: Reader, y: Reader, id: str = ''):
        return super().__head__(x, y, width=max(len(x), len(y))+1, id=id)
    
    def forward(self):
        v = self.inputs[0]._getval() + self.inputs[1]._getval()
        self.output._setval(v, dt=self.delay)

    def _verilog_op(self, *args: str) -> str:
        return ' + '.join(args) 


@dispatch('Sub', Any, [Any, None])
class _Sub(_GateNto1):
        
    id = 'Sub'
    
    def __head__(self, *args: Reader, id: str = ''):
        return super().__head__(*_align(*args), id=id)
    
    def forward(self):
        v = self.inputs[0]._getval()
        for i in self.inputs[1:]:
            v -= i._getval() 
        self.output._setval(v, dt=self.delay)

    def _verilog_op(self, *args: str) -> str:
        return ' - '.join(args)  
    
    
@dispatch('Mul', Any, [Any, None])
class _Mul(_GateNto1):
        
    id = 'Mul'
    
    def __head__(self, *args: Reader, id: str = ''):
        return super().__head__(*_align(*args), id=id)
    
    def forward(self):
        v = self.inputs[0]._getval()
        for i in self.inputs[1:]:
            v *= i._getval() 
        self.output._setval(v, dt=self.delay)

    def _verilog_op(self, *args: str) -> str:
        return ' * '.join(args) 
    
    
@dispatch('MulFull', Any, [Any, None])
class _MulFull(_GateNto1):
        
    id = 'Mul'
    
    def __head__(self, x: Reader, y: Reader, id: str = ''):
        return super().__head__(x, y, width = len(x) + len(y), id=id)
    
    def forward(self):
        v = self.inputs[0]._getval() * self.inputs[1]._getval()
        self.output._setval(v, dt=self.delay)

    def _verilog_op(self, *args: str) -> str:
        return ' * '.join(args) 
    
    
# TODO divide by zero, return random value and emit a warning
@dispatch('Floordiv', Any, Any) 
class _Floordiv(_GateNto1):
    
    id = 'Floordiv'
    
    def __head__(self, a: Reader, b: Union[int, Reader], id: str = ''):
        return super().__head__(a, b, id=id)
    
    def forward(self):
        a = self.inputs[0]._getval()
        b = self.inputs[1]._getval()
        if b == 0:
            temp = (1 << len(self.inputs[0])) - 1
            v = gmpy2.mpz(random.randint(0, temp))
            # self.sess.log.warning('SimulationWarning: divide by zero', start=5, end=10)
        else:
            v = a // b
        self.output._setval(v, dt=self.delay) 
        
    def _verilog_op(self, a, b) -> str:
        return f'{a} // {b}' 
    
    
@dispatch('Mod', Any, Any) 
class _Mod(_GateNto1):
    
    id = 'Mod'
    
    def __head__(self, a: Reader, b: Union[int, Reader], id: str = ''):
        """ ret bit length = len(a)
        """
        return super().__head__(a, b, id=id)
    
    def forward(self):
        a = self.inputs[0]._getval()
        b = self.inputs[1]._getval()
        if b == 0:
            temp = (1 << len(self.inputs[0])) - 1
            v = gmpy2.mpz(random.randint(0, temp))
            # self.sess.log.warning('SimulationWarning: divide by zero', start=1, end=10)
        else:
            v = a % b
        self.output._setval(v, dt=self.delay) 
        
    def _verilog_op(self, a, b) -> str:
        return f'{a} % {b}' 
    
# TODO Div 


@vectorize
class Mux(_GateNto1):
        
    id = 'Mux'
    
    def __head__(self, sel, a, b, id: str = ''):
        """ sel ? a: b """ 
        a, b = _align(a, b)
        return super().__head__(a, b, Signal(sel), id=id)
    
    def forward(self): 
        a, b, sel = self.inputs 
        if sel._getval():
            v = a._getval()
        else:
            v = b._getval() 

        self.output._setval(v, self.delay)

    def _verilog_op(self, *args: str) -> str:
        a, b, sel = args
        return f'{sel} ? {a}:{b}'  
    
    


