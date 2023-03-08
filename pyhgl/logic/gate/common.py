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
from itertools import chain

from pyhgl.array import *
from pyhgl.logic.hgl_basic import *
import pyhgl.logic.utils as utils
import pyhgl.logic.module_cpp as cpp
import pyhgl.logic.module_sv as sv


class _GateN(Gate):
    """ 
    args: immd | signal
        - immd: convert to UInt 
        - signal: should have fixed bit length
    name:
        name of the return signal 

    Return: a UInt Signal with maximum bit length of inputs
    """
    
    _op: Tuple[str, str] = ('','')   # verilog operator ex. Nand -> ('~', '&')
    
    def __head__(self, *args: Reader, id: str='', name: str='res_n') -> Reader:

        args = [Signal(i) for i in args]
        assert args, 'at least 1 signal'
        assert all(isinstance(i._data, LogicData) for i in args), 'not logic signal'
        assert all(i._type._storage == 'packed' for i in args), 'signal with variable length'

        self.id: str = id or self.id
        self.inputs: List[Reader] = [self.read(i) for i in args]    # read 
        self.width = max(len(i) for i in args) 
        ret: Reader = UInt[self.width](0, name=name)
        self.output: Writer = self.write(ret)                               # write 
        return ret

    def dump_cpp(self):
        raise NotImplementedError(self)

    def dump_sv(self, builder: sv.ModuleSV):
        x = [builder.get_name(i) for i in self.inputs]
        y = builder.get_name(self.output) 
        op1, op2 = self._op
        x = f' {op2} '.join(x) 
        x = f'{op1}({x})'
        builder.Assign(self, y, x, delay=self.delay)
     
        
        

#-------------
# bitwise gate
#-------------

@dispatch('Not', Any, None)
class _Not(_GateN):
    
    id = 'Not'
    _op = ('~', '')

    def forward(self):
        input_not: LogicData = ~self.inputs[0]._data._getval_py()
        mask = gmpy2.bit_mask(self.width) 
        out = LogicData(
            input_not.v & mask,
            input_not.x & mask,
        )
        self.output._data._setval_py(out, low_keys=None, dt=self.delay)


    def dump_cpp(self) -> None:
        input_v: List[cpp.TData] = self.inputs[0]._data._getval_cpp(low_key=None, part='v') 
        input_x: List[cpp.TData] = self.inputs[0]._data._getval_cpp(low_key=None, part='x') 
        output_v: List[cpp.TData] = self.output._data._getval_cpp(low_key=None, part='v') 
        output_x: List[cpp.TData] = self.output._data._getval_cpp(low_key=None, part='x') 

        for i in range(len(output_v)):
            node = cpp.Node(name='not_exec')
            node.dump_value() 
            node.Not(input_v[i], target=output_v[i], width=len(output_v[i]), delay=self.delay)
            node.dump_unknown()
            node.Not(input_x[i], target=output_x[i], width=len(output_x[i]), delay=self.delay)
        

    
@dispatch('And', Any, [Any, None])
class _And(_GateN):
        
    id = 'And'
    _op = ('', '&')
    
    def forward(self): 
        v = self.inputs[0]._data._getval_py()
        for i in self.inputs[1:]:
            v = v & i._data._getval_py() 
        self.output._data._setval_py(v, dt=self.delay)

    def dump_cpp(self) -> None:
        inputs_v = [i._data._getval_cpp(low_key=None, part='v') for i in self.inputs]
        inputs_x = [i._data._getval_cpp(low_key=None, part='x') for i in self.inputs]
        output_v = self.output._data._getval_cpp(low_key=None, part='v')
        output_x = self.output._data._getval_cpp(low_key=None, part='x') 
        
        for i in range(len(output_v)):
            in_v_i: List[cpp.TData] = []
            in_x_i: List[cpp.TData] = []
            for v, x in zip(inputs_v, inputs_x):
                if i < len(v):
                    in_v_i.append(v[i])
                    in_x_i.append(x[i])
            node = cpp.Node(name='and_exec')
            node.dump_value() 
            node.And(*in_v_i, target=output_v[i], delay=self.delay) 
            node.dump_unknown()
            out_v_i = in_v_i[0]
            out_x_i = in_x_i[0]
            for in_v, in_x in zip(in_v_i[1:], in_x_i[1:]):
                out_v_i, out_x_i = node.And(
                    in_v, out_v_i
                ), node.Or(
                    node.And(in_x, out_x_i),
                    node.And(in_x, out_v_i), 
                    node.And(in_v, out_x_i),
                ) 
            node.And(out_x_i, target=output_x[i], delay=self.delay)


    
@dispatch('Or', Any, [Any, None])
class _Or(_GateN):
    
    id = 'Or'
    _op = ('', '|')

    def forward(self): 
        v = self.inputs[0]._data._getval_py()
        for i in self.inputs[1:]:
            v = v | i._data._getval_py() 
        self.output._data._setval_py(v, dt=self.delay)

    def dump_cpp(self):
        return super().dump_cpp()


@dispatch('Xor', Any, [Any, None])
class _Xor(_GateN):
    
    id = 'Xor'
    _op = ('', '^')
        
    def forward(self): 
        v = self.inputs[0]._data._getval_py()
        for i in self.inputs[1:]:
            v = v ^ i._data._getval_py() 
        self.output._data._setval_py(v, dt=self.delay)

    def dump_cpp(self):
        return super().dump_cpp()

    
@dispatch('Nand', Any, [Any, None])
class _Nand(_GateN):
    
    id = 'Nand'
    _op = ('~', '&')

    def forward(self): 
        v = self.inputs[0]._data._getval_py()
        for i in self.inputs[1:]:
            v = v & i._data._getval_py() 
        input_not: LogicData = ~ v
        mask = gmpy2.bit_mask(self.width)
        out = LogicData(
            input_not.v & mask,
            input_not.x & mask,
        )
        self.output._data._setval_py(out, dt=self.delay) 

    def dump_cpp(self):
        return super().dump_cpp()
    
    
@dispatch('Nor', Any, [Any, None])
class _Nor(_GateN):
    
    id = 'Nor'
    _op = ('~', '|')
        
    def forward(self): 
        v = self.inputs[0]._data._getval_py()
        for i in self.inputs[1:]:
            v = v | i._data._getval_py() 
        input_not: LogicData = ~ v
        mask = gmpy2.bit_mask(self.width)
        out = LogicData(
            input_not.v & mask,
            input_not.x & mask,
        )
        self.output._data._setval_py(out, dt=self.delay) 

    def dump_cpp(self):
        return super().dump_cpp()


@dispatch('Nxor', Any, [Any, None])
class _Nxor(_GateN):
    
    id = 'Nxor'
    _op = ('~', '^')
        
    def forward(self): 
        v = self.inputs[0]._data._getval_py()
        for i in self.inputs[1:]:
            v = v ^ i._data._getval_py() 
        input_not: LogicData = ~ v
        mask = gmpy2.bit_mask(self.width)
        out = LogicData(
            input_not.v & mask,
            input_not.x & mask,
        )
        self.output._data._setval_py(out, dt=self.delay) 

    def dump_cpp(self):
        return super().dump_cpp()




# TODO reduce gate, accept variable length, return 1 bit uint
@dispatch('AndR', Any) 
class _AndR(_GateN):
    
    id = 'AndR'
    
    def __head__(self, x: Reader, id: str = ''):
        return super().__head__(x, width=1, id=id)
    
    def forward(self):
        v: gmpy2.mpz = self.inputs[0]._getval_py()
        for i in range(len(self.inputs[0])):
            if v[i] == 0:
                self.output._setval_py(gmpy2.mpz(0), dt=self.delay)
                return 
        self.output._setval_py(gmpy2.mpz(1), dt=self.delay) 
        
    def _verilog_op(self, x: str) -> str:
        return f'& {x}'


@dispatch('NandR', Any) 
class _NandR(_GateN):
    
    id = 'NandR'
    
    def __head__(self, x: Reader, id: str = ''):
        return super().__head__(x, width=1, id=id)
    
    def forward(self):
        v = self.inputs[0]._getval_py()
        for i in range(len(self.inputs[0])):
            if v[i] == 0:
                self.output._setval_py(gmpy2.mpz(1), dt=self.delay)
                return 
        self.output._setval_py(gmpy2.mpz(0), dt=self.delay)

    def _verilog_op(self, x: str) -> str:
        return f'~& {x}'

    
@dispatch('OrR', Any) 
class _OrR(_GateN):
    
    id = 'OrR'
    
    def __head__(self, x: Reader, id: str = ''):
        return super().__head__(x, width=1, id=id)
    
    def forward(self):
        if self.inputs[0]._getval_py() == 0:
            self.output._setval_py(gmpy2.mpz(0), dt=self.delay) 
        else:
            self.output._setval_py(gmpy2.mpz(1), dt=self.delay) 
        
    def _verilog_op(self, x: str) -> str:
        return f'| {x}'


@dispatch('NorR', Any) 
class _NorR(_GateN):
    
    id = 'NorR'
    
    def __head__(self, x: Reader, id: str = ''):
        return super().__head__(x, width=1, id=id)
    
    def forward(self):
        if self.inputs[0]._getval_py() == 0:
            self.output._setval_py(gmpy2.mpz(1), dt=self.delay) 
        else:
            self.output._setval_py(gmpy2.mpz(0), dt=self.delay) 
        
    def _verilog_op(self, x: str) -> str:
        return f'~| {x}'
    

@dispatch('XorR', Any) 
class _XorR(_GateN):
    
    id = 'XorR'
    
    def __head__(self, x: Reader, id: str = ''):
        return super().__head__(x, width=1, id=id)
    
    def forward(self):
        v = self.inputs[0]._getval_py()
        ret = utils.parity(v)
        self.output._setval_py(gmpy2.mpz(ret), dt=self.delay) 
        
    def _verilog_op(self, x: str) -> str:
        return f'^ {x}' 
    

@dispatch('NxorR', Any) 
class _NxorR(_GateN):
    
    id = 'NxorR'
    
    def __head__(self, x: Reader, id: str = ''):
        return super().__head__(x, width=1, id=id)
    
    def forward(self):
        v = self.inputs[0]._getval_py()
        ret = utils.parity(v)
        self.output._setval_py(gmpy2.mpz(1-ret), dt=self.delay) 
        
    def _verilog_op(self, x: str) -> str:
        return f'~^ {x}' 


        
@dispatch('Cat', Any, [Any, None])
class _Cat(_GateN):
    """ Cat signals, return UInt
    """
    
    id = 'Cat'
    
    __slots__ = 'widths'
    
    def __head__(self, *args: Reader, id: str=''): 
        self.widths = [len(i) for i in args]
        return super().__head__(*args, width=sum(self.widths), id=id)
        
    def forward(self):
        v = LogicData(0,0)
        start = 0
        for width, s in zip(self.widths, self.inputs): 
            v = v | (s._getval_py() << start)
            start += width 
        mask, v = self.output._type._setval_py(None, v)
        self.output._setval_py(v, dt = self.delay, mask=mask)
    
    def _verilog_op(self, *args: str) -> str:
        args = reversed(args)
        return f"{{{','.join(args)}}}"    

    
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
class _LogicAnd(_GateN):
        
    id = 'LogicAnd'
    
    def __head__(self, *args: Reader, id: str = ''):
        return super().__head__(*args, width=1, id=id)
    
    def forward(self): 
        v = all(i._getval_py() > 0 for i in self.inputs)
        self.output._setval_py(gmpy2.mpz(v), self.delay)

    def _verilog_op(self, *args: str) -> str:
        return ' && '.join(args) 
    
    
@dispatch('LogicOr', Any, [Any, None])
class _LogicOr(_GateN):
        
    id = 'LogicOr'
    
    def __head__(self, *args: Reader, id: str = ''):
        return super().__head__(*args, width=1, id=id)
    
    def forward(self):
        v = any(i._getval_py() > 0 for i in self.inputs)
        self.output._setval_py(gmpy2.mpz(v), self.delay)

    def _verilog_op(self, *args: str) -> str:
        return ' || '.join(args)
     
    
#------- 
# shift
#------- 
    
@dispatch('Lshift', Any, [UIntType, int]) 
class _Lshift(_GateN):
    
    id = 'Lshift'
    
    def __head__(self, a: Reader, b: Union[int, Reader], id: str = ''):
        return super().__head__(a, Signal(b), id=id)
    
    def forward(self):
        v = self.inputs[0]._getval_py() << self.inputs[1]._getval_py()
        self.output._setval_py(v, dt=self.delay) 
        
    def _verilog_op(self, a, b) -> str:
        return f'{a} << {b}' 
    
    
@dispatch('Rshift', Any, [UIntType, int]) 
class _Rshift(_GateN):
    
    id = 'Rshift'
    
    def __head__(self, a: Reader, b: Union[int, Reader], id: str = ''):
        return super().__head__(a, Signal(b), id=id)
    
    def forward(self):
        v = self.inputs[0]._getval_py() >> self.inputs[1]._getval_py()
        self.output._setval_py(v, dt=self.delay) 
        
    def _verilog_op(self, a, b) -> str:
        return f'{a} >> {b}' 
    
# TODO round shift, SInt shift
    

#-----------
# arithmetic
#-----------

@dispatch('Pos', Any, None)
class _Pos(_GateN):
    
    id = 'Pos'
    
    def __head__(self, x: Reader, id: str = ''):
        return super().__head__(x, id=id)
        
    def forward(self):
        self.output._setval_py(self.inputs[0]._getval_py(), dt=self.delay)

    def _verilog_op(self, x: str) -> str:
        return f'+ {x}' 
    
    
@dispatch('Neg', Any, None)
class _Neg(_GateN):
    
    id = 'Neg'
    
    def __head__(self, x: Reader, id: str = ''):
        return super().__head__(x, id=id)
        
    def forward(self):
        self.output._setval_py( - self.inputs[0]._getval_py(), dt=self.delay)

    def _verilog_op(self, x: str) -> str:
        return f'- {x}' 


@dispatch('Add', Any, [Any, None])
class _Add(_GateN):
        
    id = 'Add'
    
    def __head__(self, *args: Reader, id: str = ''):
        return super().__head__(*_align(*args), id=id)
    
    def forward(self):
        v = self.inputs[0]._getval_py()
        for i in self.inputs[1:]:
            v += i._getval_py() 
        self.output._setval_py(v, dt=self.delay)

    def _verilog_op(self, *args: str) -> str:
        return ' + '.join(args) 
    
    
@dispatch('AddFull', Any, [Any, None])
class _AddFull(_GateN):
        
    id = 'Add'
    
    def __head__(self, x: Reader, y: Reader, id: str = ''):
        return super().__head__(x, y, width=max(len(x), len(y))+1, id=id)
    
    def forward(self):
        v = self.inputs[0]._getval_py() + self.inputs[1]._getval_py()
        self.output._setval_py(v, dt=self.delay)

    def _verilog_op(self, *args: str) -> str:
        return ' + '.join(args) 


@dispatch('Sub', Any, [Any, None])
class _Sub(_GateN):
        
    id = 'Sub'
    
    def __head__(self, *args: Reader, id: str = ''):
        return super().__head__(*_align(*args), id=id)
    
    def forward(self):
        v = self.inputs[0]._getval_py()
        for i in self.inputs[1:]:
            v -= i._getval_py() 
        self.output._setval_py(v, dt=self.delay)

    def _verilog_op(self, *args: str) -> str:
        return ' - '.join(args)  
    
    
@dispatch('Mul', Any, [Any, None])
class _Mul(_GateN):
        
    id = 'Mul'
    
    def __head__(self, *args: Reader, id: str = ''):
        return super().__head__(*_align(*args), id=id)
    
    def forward(self):
        v = self.inputs[0]._getval_py()
        for i in self.inputs[1:]:
            v *= i._getval_py() 
        self.output._setval_py(v, dt=self.delay)

    def _verilog_op(self, *args: str) -> str:
        return ' * '.join(args) 
    
    
@dispatch('MulFull', Any, [Any, None])
class _MulFull(_GateN):
        
    id = 'Mul'
    
    def __head__(self, x: Reader, y: Reader, id: str = ''):
        return super().__head__(x, y, width = len(x) + len(y), id=id)
    
    def forward(self):
        v = self.inputs[0]._getval_py() * self.inputs[1]._getval_py()
        self.output._setval_py(v, dt=self.delay)

    def _verilog_op(self, *args: str) -> str:
        return ' * '.join(args) 
    
    
# TODO divide by zero, return random value and emit a warning
@dispatch('Floordiv', Any, Any) 
class _Floordiv(_GateN):
    
    id = 'Floordiv'
    
    def __head__(self, a: Reader, b: Union[int, Reader], id: str = ''):
        return super().__head__(a, b, id=id)
    
    def forward(self):
        a = self.inputs[0]._getval_py()
        b = self.inputs[1]._getval_py()
        if b == 0:
            temp = (1 << len(self.inputs[0])) - 1
            v = gmpy2.mpz(random.randint(0, temp))
            # self.sess.log.warning('SimulationWarning: divide by zero', start=5, end=10)
        else:
            v = a // b
        self.output._setval_py(v, dt=self.delay) 
        
    def _verilog_op(self, a, b) -> str:
        return f'{a} // {b}' 
    
    
@dispatch('Mod', Any, Any) 
class _Mod(_GateN):
    
    id = 'Mod'
    
    def __head__(self, a: Reader, b: Union[int, Reader], id: str = ''):
        """ ret bit length = len(a)
        """
        return super().__head__(a, b, id=id)
    
    def forward(self):
        a = self.inputs[0]._getval_py()
        b = self.inputs[1]._getval_py()
        if b == 0:
            temp = (1 << len(self.inputs[0])) - 1
            v = gmpy2.mpz(random.randint(0, temp))
            # self.sess.log.warning('SimulationWarning: divide by zero', start=1, end=10)
        else:
            v = a % b
        self.output._setval_py(v, dt=self.delay) 
        
    def _verilog_op(self, a, b) -> str:
        return f'{a} % {b}' 
    
# TODO Div 
def _align(*args: Reader) -> list:  
    """ return aligned signals
    XXX wrong, don't directly use cast 
    """
    args = [Signal(i) for i in args]
    assert all(i._type._storage == 'packed' for i in args), 'signal type has variable length'
    widths = [len(i) for i in args] 
    max_w = max(widths)  
    return [UInt[max_w](s) if w != max_w else s for s, w in zip(args, widths)]
    
    

@vectorize
class Mux(_GateN):
        
    id = 'Mux'
    
    def __head__(self, sel, a, b, id: str = ''):
        """ sel ? a: b """ 
        a, b = _align(a, b)
        return super().__head__(a, b, Signal(sel), id=id)
    
    def forward(self): 
        a, b, sel = self.inputs 
        if sel._getval_py():
            v = a._getval_py()
        else:
            v = b._getval_py() 

        self.output._setval_py(v, self.delay)

    def _verilog_op(self, *args: str) -> str:
        a, b, sel = args
        return f'{sel} ? {a}:{b}'  
    
    


