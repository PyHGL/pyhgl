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
from pyhgl.logic.hgl_core import *
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
    
    def __head__(self, *args: Reader, id: str='', name: str='temp_n') -> Reader:

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

    def sim_init(self):
        super().sim_init() 
        self.sim_x_count -= 1000

    def sim_vx(self):
        delay = self.timing['delay']
        data = self.inputs[0]._data 
        target = self.output._data 
        mask = gmpy2.bit_mask(self.width)
        simulator = self._sess.sim_py 
        while 1:
            yield 
            simulator.update_v(delay, target, (~data.v) & mask) 
            simulator.update_x(delay, target, data.x)
    
    def sim_v(self):
        delay = self.timing['delay']
        data = self.inputs[0]._data 
        target = self.output._data 
        mask = gmpy2.bit_mask(self.width)
        simulator = self._sess.sim_py 
        while 1:
            yield 
            simulator.update_v(delay, target, (~data.v) & mask) 
            
    def dump_cpp(self) -> None:
        input_v: List[cpp.TData] = self.inputs[0]._data._getval_cpp('v') 
        input_x: List[cpp.TData] = self.inputs[0]._data._getval_cpp('x') 
        output_v: List[cpp.TData] = self.output._data._getval_cpp('v') 
        output_x: List[cpp.TData] = self.output._data._getval_cpp('x') 

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
    
    def sim_init(self):
        super().sim_init() 
        self.sim_x_count -= 1000

    def sim_vx(self):
        delay = self.timing['delay']
        first = self.inputs[0]._data  
        rest = [i._data for i in self.inputs[1:]]
        target = self.output._data 
        simulator = self._sess.sim_py 
        while 1:
            yield  
            v = first._getval_py()
            for i in rest:
                v = v & i._getval_py()
            simulator.update_v(delay, target, v.v) 
            simulator.update_x(delay, target, v.x)
    
    def sim_v(self):
        delay = self.timing['delay']
        first = self.inputs[0]._data  
        rest = [i._data for i in self.inputs[1:]]
        target = self.output._data 
        simulator = self._sess.sim_py 
        while 1:
            yield  
            v = first.v
            for i in rest:
                v = v & i.v
            simulator.update_v(delay, target, v) 

    def dump_cpp(self) -> None:
        """
        v3 = v1 & v2 
        x3 = x1 & x2 | v1 & x2 | v2 & x1
        """
        inputs_v = [i._data._getval_cpp('v') for i in self.inputs]
        inputs_x = [i._data._getval_cpp('x') for i in self.inputs]
        output_v = self.output._data._getval_cpp('v')
        output_x = self.output._data._getval_cpp('x') 
        
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

    def sim_init(self):
        super().sim_init() 
        self.sim_x_count -= 1000

    def sim_vx(self):
        delay = self.timing['delay']
        first = self.inputs[0]._data  
        rest = [i._data for i in self.inputs[1:]]
        target = self.output._data 
        simulator = self._sess.sim_py 
        while 1:
            yield  
            v = first._getval_py()
            for i in rest:
                v = v | i._getval_py()
            simulator.update_v(delay, target, v.v) 
            simulator.update_x(delay, target, v.x)
    
    def sim_v(self):
        delay = self.timing['delay']
        first = self.inputs[0]._data  
        rest = [i._data for i in self.inputs[1:]]
        target = self.output._data 
        simulator = self._sess.sim_py 
        while 1:
            yield  
            v = first.v
            for i in rest:
                v = v | i.v
            simulator.update_v(delay, target, v) 


    def dump_cpp(self):
        """
        v3 = v1 | v2 
        x3 = x1 & x2 | (~v2) & x1 | (~v1) & x2
        """
        inputs_v = [i._data._getval_cpp('v') for i in self.inputs]
        inputs_x = [i._data._getval_cpp('x') for i in self.inputs]
        output_v = self.output._data._getval_cpp('v')
        output_x = self.output._data._getval_cpp('x') 
        
        for i in range(len(output_v)):
            in_v_i: List[cpp.TData] = []
            in_x_i: List[cpp.TData] = []
            for v, x in zip(inputs_v, inputs_x):
                if i < len(v):
                    in_v_i.append(v[i])
                    in_x_i.append(x[i])
            node = cpp.Node(name='and_exec')
            node.dump_value() 
            node.Or(*in_v_i, target=output_v[i], delay=self.delay) 
            node.dump_unknown()
            out_v_i = in_v_i[0]
            out_x_i = in_x_i[0]
            for in_v, in_x in zip(in_v_i[1:], in_x_i[1:]):
                out_v_i, out_x_i = node.Or(
                    in_v, out_v_i
                ), node.Or(
                    node.And(in_x, out_x_i),
                    node.And(in_x, node.Not(out_v_i)), 
                    node.And(out_x_i, node.Not(in_v)),
                ) 
            node.And(out_x_i, target=output_x[i], delay=self.delay)


@dispatch('Xor', Any, [Any, None])
class _Xor(_GateN):
    
    id = 'Xor'
    _op = ('', '^')
        
    def sim_init(self):
        super().sim_init() 
        self.sim_x_count -= 1000

    def sim_vx(self):
        delay = self.timing['delay']
        first = self.inputs[0]._data  
        rest = [i._data for i in self.inputs[1:]]
        target = self.output._data 
        simulator = self._sess.sim_py 
        while 1:
            yield  
            v = first._getval_py()
            for i in rest:
                v = v ^ i._getval_py()
            simulator.update_v(delay, target, v.v) 
            simulator.update_x(delay, target, v.x)
    
    def sim_v(self):
        delay = self.timing['delay']
        first = self.inputs[0]._data  
        rest = [i._data for i in self.inputs[1:]]
        target = self.output._data 
        simulator = self._sess.sim_py 
        while 1:
            yield  
            v = first.v
            for i in rest:
                v = v ^ i.v
            simulator.update_v(delay, target, v) 

    def dump_cpp(self):
        """
        v3 = v1 ^ v2 
        x3 = x1 | x2 
        """
        inputs_v = [i._data._getval_cpp('v') for i in self.inputs]
        inputs_x = [i._data._getval_cpp('x') for i in self.inputs]
        output_v = self.output._data._getval_cpp('v')
        output_x = self.output._data._getval_cpp('x') 
        
        for i in range(len(output_v)):
            in_v_i: List[cpp.TData] = []
            in_x_i: List[cpp.TData] = []
            for v, x in zip(inputs_v, inputs_x):
                if i < len(v):
                    in_v_i.append(v[i])
                    in_x_i.append(x[i])
            node = cpp.Node(name='and_exec')
            node.dump_value() 
            node.Xor(*in_v_i, target=output_v[i], delay=self.delay) 
            node.dump_unknown()
            node.Or(*in_x_i, target=output_x[i], delay=self.delay)

    
@dispatch('Nand', Any, [Any, None])
class _Nand(_GateN):
    
    id = 'Nand'
    _op = ('~', '&')


    def sim_init(self):
        super().sim_init() 
        self.sim_x_count -= 1000

    def sim_vx(self):
        delay = self.timing['delay']
        first = self.inputs[0]._data  
        rest = [i._data for i in self.inputs[1:]]
        mask = gmpy2.bit_mask(self.width)
        target = self.output._data 
        simulator = self._sess.sim_py  
        while 1:
            yield  
            v = first._getval_py()
            for i in rest:
                v = v & i._getval_py()
            simulator.update_v(delay, target, (~v.v) & mask) 
            simulator.update_x(delay, target, v.x)
    
    def sim_v(self):
        delay = self.timing['delay']
        first = self.inputs[0]._data  
        rest = [i._data for i in self.inputs[1:]]
        mask = gmpy2.bit_mask(self.width)
        target = self.output._data 
        simulator = self._sess.sim_py 
        while 1:
            yield  
            v = first.v
            for i in rest:
                v = v & i.v
            simulator.update_v(delay, target, (~v)&mask) 

    def dump_cpp(self):
        inputs_v = [i._data._getval_cpp('v') for i in self.inputs]
        inputs_x = [i._data._getval_cpp('x') for i in self.inputs]
        output_v = self.output._data._getval_cpp('v')
        output_x = self.output._data._getval_cpp('x') 
        
        for i in range(len(output_v)):
            in_v_i: List[cpp.TData] = []
            in_x_i: List[cpp.TData] = []
            for v, x in zip(inputs_v, inputs_x):
                if i < len(v):
                    in_v_i.append(v[i])
                    in_x_i.append(x[i])
            node = cpp.Node(name='and_exec')
            node.dump_value() 
            temp = node.And(*in_v_i, delay=self.delay) 
            node.Not(temp, target=output_v[i], delay = self.delay)
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
            node.And(out_x_i, target=output_x[i], delay=self.delay)  # x does not change
    
    
@dispatch('Nor', Any, [Any, None])
class _Nor(_GateN):
    
    id = 'Nor'
    _op = ('~', '|')
        
    def sim_init(self):
        super().sim_init() 
        self.sim_x_count -= 1000

    def sim_vx(self):
        delay = self.timing['delay']
        first = self.inputs[0]._data  
        rest = [i._data for i in self.inputs[1:]]
        mask = gmpy2.bit_mask(self.width)
        target = self.output._data 
        simulator = self._sess.sim_py  
        while 1:
            yield  
            v = first._getval_py()
            for i in rest:
                v = v | i._getval_py()
            simulator.update_v(delay, target, (~v.v) & mask) 
            simulator.update_x(delay, target, v.x)
    
    def sim_v(self):
        delay = self.timing['delay']
        first = self.inputs[0]._data  
        rest = [i._data for i in self.inputs[1:]]
        mask = gmpy2.bit_mask(self.width)
        target = self.output._data 
        simulator = self._sess.sim_py 
        while 1:
            yield  
            v = first.v
            for i in rest:
                v = v | i.v
            simulator.update_v(delay, target, (~v)&mask) 
        
    def dump_cpp(self):
        inputs_v = [i._data._getval_cpp('v') for i in self.inputs]
        inputs_x = [i._data._getval_cpp('x') for i in self.inputs]
        output_v = self.output._data._getval_cpp('v')
        output_x = self.output._data._getval_cpp('x') 
        
        for i in range(len(output_v)):
            in_v_i: List[cpp.TData] = []
            in_x_i: List[cpp.TData] = []
            for v, x in zip(inputs_v, inputs_x):
                if i < len(v):
                    in_v_i.append(v[i])
                    in_x_i.append(x[i])
            node = cpp.Node(name='and_exec')
            node.dump_value() 
            temp = node.Or(*in_v_i, delay=self.delay) 
            node.Not(temp, target=output_v[i], delay = self.delay)
            node.dump_unknown()
            out_v_i = in_v_i[0]
            out_x_i = in_x_i[0]
            for in_v, in_x in zip(in_v_i[1:], in_x_i[1:]):
                out_v_i, out_x_i = node.Or(
                    in_v, out_v_i
                ), node.Or(
                    node.And(in_x, out_x_i),
                    node.And(in_x, node.Not(out_v_i)), 
                    node.And(out_x_i, node.Not(in_v)),
                ) 
            node.And(out_x_i, target=output_x[i], delay=self.delay)


@dispatch('Nxor', Any, [Any, None])
class _Nxor(_GateN):
    
    id = 'Nxor'
    _op = ('~', '^')
        
    def sim_init(self):
        super().sim_init() 
        self.sim_x_count -= 1000

    def sim_vx(self):
        delay = self.timing['delay']
        first = self.inputs[0]._data  
        rest = [i._data for i in self.inputs[1:]]
        mask = gmpy2.bit_mask(self.width)
        target = self.output._data 
        simulator = self._sess.sim_py  
        while 1:
            yield  
            v = first._getval_py()
            for i in rest:
                v = v ^ i._getval_py()
            simulator.update_v(delay, target, (~v.v) & mask) 
            simulator.update_x(delay, target, v.x)
    
    def sim_v(self):
        delay = self.timing['delay']
        first = self.inputs[0]._data  
        rest = [i._data for i in self.inputs[1:]]
        mask = gmpy2.bit_mask(self.width)
        target = self.output._data 
        simulator = self._sess.sim_py 
        while 1:
            yield  
            v = first.v
            for i in rest:
                v = v ^ i.v
            simulator.update_v(delay, target, (~v)&mask) 

    def dump_cpp(self):
        inputs_v = [i._data._getval_cpp('v') for i in self.inputs]
        inputs_x = [i._data._getval_cpp('x') for i in self.inputs]
        output_v = self.output._data._getval_cpp('v')
        output_x = self.output._data._getval_cpp('x') 
        
        for i in range(len(output_v)):
            in_v_i: List[cpp.TData] = []
            in_x_i: List[cpp.TData] = []
            for v, x in zip(inputs_v, inputs_x):
                if i < len(v):
                    in_v_i.append(v[i])
                    in_x_i.append(x[i])
            node = cpp.Node(name='and_exec')
            node.dump_value() 
            temp = node.Xor(*in_v_i, delay=self.delay) 
            node.Not(temp, target=output_v[i], delay=self.delay)
            node.dump_unknown() 
            node.Or(*in_x_i, target=output_x[i], delay=self.delay)


class _Reduce(_GateN):

    def __head__(self, x: Reader, id: str = '', name: str = 'temp_r') -> Reader:
        """ reduce gate, accept variable length, return 1 bit uint
        """
        assert isinstance(x._data, LogicData), 'not logic signal' 
        self.id: str = id or self.id 
        self.input = self.read(x)
        ret: Reader = UInt[1](0, name=name)
        self.output = self.write(ret)
        return ret 

    def dump_sv(self, builder: sv.ModuleSV):
        x = builder.get_name(self.input)  
        y = builder.get_name(self.output) 
        op1, op2 = self._op
        x = f'{op1}{x}'
        builder.Assign(self, y, x, delay=self.delay)
        

@dispatch('AndR', Any) 
class _AndR(_Reduce):
    
    id = 'AndR'
    _op = ('&', '')
    
    def sim_init(self):
        super().sim_init() 
        self.sim_x_count -= 1000

    def sim_vx(self):
        delay = self.timing['delay']
        data = self.input._data  
        target = self.output._data 
        mask = gmpy2.bit_mask(len(self.input))  
        simulator = self._sess.sim_py  
        while 1:
            yield  
            v = data.v 
            x = data.x 
            if (~v) & (~x) & mask:                      # exists 0 
                out_v, out_x = gmpy2.mpz(0),gmpy2.mpz(0) 
            elif x:                                     # unknown
                out_v, out_x = gmpy2.mpz(0),gmpy2.mpz(1) 
            else:                                       # all 1
                out_v, out_x = gmpy2.mpz(1),gmpy2.mpz(0)  

            simulator.update_v(delay, target, out_v) 
            simulator.update_x(delay, target, out_x)
    
    def sim_v(self):
        delay = self.timing['delay']
        data = self.input._data  
        target = self.output._data 
        mask = gmpy2.bit_mask(len(self.input))  
        simulator = self._sess.sim_py  
        while 1:
            yield  
            if data.v == mask:
                out_v = gmpy2.mpz(1)   # all 1 
            else:
                out_v = gmpy2.mpz(0)   # exists 0

            simulator.update_v(delay, target, out_v) 

        
    def dump_cpp(self) -> None: 
        input_v: List[cpp.TData] = self.input._data._getval_cpp('v')
        input_x: List[cpp.TData] = self.input._data._getval_cpp('x')
        output_v: cpp.TData = self.output._data._getval_cpp('v')[0]
        output_x: cpp.TData = self.output._data._getval_cpp('x')[0]
        
        node = cpp.Node(name='andr_exec')
        node.dump_value() 
        node.AndR(*input_v, target=output_v, delay=self.delay)
        node.dump_unknown()  



@dispatch('OrR', Any) 
class _OrR(_Reduce):
    
    id = 'OrR'
    _op = ('|', '')
    
    def sim_init(self):
        super().sim_init() 
        self.sim_x_count -= 1000

    def sim_vx(self):
        delay = self.timing['delay']
        data = self.input._data  
        target = self.output._data 
        simulator = self._sess.sim_py  
        while 1:
            yield  
            v = data.v 
            x = data.x 
            if  v & (~x):                               # exists 1          
                out_v, out_x = gmpy2.mpz(1),gmpy2.mpz(0) 
            elif x:                                     # unknown
                out_v, out_x = gmpy2.mpz(0),gmpy2.mpz(1) 
            else:                                       # all 0
                out_v, out_x = gmpy2.mpz(0),gmpy2.mpz(0)  

            simulator.update_v(delay, target, out_v) 
            simulator.update_x(delay, target, out_x)
    
    def sim_v(self):
        delay = self.timing['delay']
        data = self.input._data  
        target = self.output._data 
        simulator = self._sess.sim_py  
        while 1:
            yield  
            if data.v:
                out_v = gmpy2.mpz(1)   
            else:
                out_v = gmpy2.mpz(0)   
            simulator.update_v(delay, target, out_v)





@dispatch('XorR', Any) 
class _XorR(_Reduce):
    
    id = 'XorR'
    _op = ('^', '')
    

    def sim_init(self):
        super().sim_init() 
        self.sim_x_count -= 1000

    def sim_vx(self):
        delay = self.timing['delay']
        data = self.input._data  
        target = self.output._data 
        simulator = self._sess.sim_py  
        while 1:
            yield  
            v = data.v 
            x = data.x 
            if x:                                     
                out_v, out_x = gmpy2.mpz(0),gmpy2.mpz(1) 
            else:                                        
                out_v, out_x = utils.parity(v),gmpy2.mpz(0)  

            simulator.update_v(delay, target, out_v) 
            simulator.update_x(delay, target, out_x)
    
    def sim_v(self):
        delay = self.timing['delay']
        data = self.input._data  
        target = self.output._data 
        simulator = self._sess.sim_py  
        while 1:
            yield   
            simulator.update_v(delay, target, utils.parity(data.v))
    
        
@dispatch('Cat', Any, [Any, None])
class _Cat(Gate):
    
    id = 'Cat'
    
    def __head__(self, *args: Reader, id: str='', name: str = 'temp_cat') -> Reader: 
        args = [Signal(i) for i in args]
        assert args, 'at least 1 signal'
        assert all(isinstance(i._data, LogicData) for i in args), 'not logic signal'
        assert all(i._type._storage == 'packed' for i in args), 'signal with variable length'

        self.id: str = id or self.id
        self.inputs: List[Reader] = [self.read(i) for i in args]    # read 
        self.widths: List[int] = [len(i) for i in args]
        ret: Reader = UInt[sum(self.widths)](0, name=name)
        self.output: Writer = self.write(ret)                       # write  
        return ret

    def sim_vx(self):
        delay = self.timing['delay']
        target = self.output._data 
        simulator = self._sess.sim_py 

        datas = [(w, s._data) for w,s in zip(self.widths, self.inputs)] 
        while 1:
            yield   
            v = gmpy2.mpz(0)
            x = gmpy2.mpz(0)
            start = 0
            for width, data in datas:
                _v = data.v 
                _x = data.x 
                v = v | (_v << start)
                x = x | (_x << start)
                start += width 
            simulator.update_v(delay, target, v) 
            simulator.update_x(delay, target, x)
    
    def sim_v(self):
        delay = self.timing['delay']
        target = self.output._data 
        simulator = self._sess.sim_py 

        datas = [(w, s._data) for w,s in zip(self.widths, self.inputs)]  
        while 1:
            yield   
            v = gmpy2.mpz(0)
            start = 0
            for width, data in datas:
                _v = data.v 
                v = v | (_v << start)
                start += width 
            simulator.update_v(delay, target, v) 


    def dump_sv(self, builder: sv.ModuleSV): 
        x = [builder.get_name(i) for i in reversed(self.inputs)]
        y = builder.get_name(self.output) 
        x = f"{{{','.join(x)}}}"
        builder.Assign(self, y, x, delay=self.delay) 

    
@dispatch('Pow', Any, int)
def _pow(x: Reader, n: int, id: str='') -> Reader:
    if n < 1:
        raise ValueError('signal duplication should be positive integer')
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
class _LogicNot(_Reduce):
    
    id = 'LogicNot'
    _op = ('!', '')
    
    def sim_init(self):
        super().sim_init() 
        self.sim_x_count -= 1000

    def sim_vx(self):
        delay = self.timing['delay']
        data = self.input._data  
        target = self.output._data 
        simulator = self._sess.sim_py  
        while 1:
            yield  
            v = data.v 
            x = data.x 
            if  v & (~x):                               # exists 1          
                out_v, out_x = gmpy2.mpz(0),gmpy2.mpz(0) 
            elif x:                                     # unknown
                out_v, out_x = gmpy2.mpz(0),gmpy2.mpz(1) 
            else:                                       # all 0
                out_v, out_x = gmpy2.mpz(1),gmpy2.mpz(0)  

            simulator.update_v(delay, target, out_v) 
            simulator.update_x(delay, target, out_x)
    
    def sim_v(self):
        delay = self.timing['delay']
        data = self.input._data  
        target = self.output._data 
        simulator = self._sess.sim_py  
        while 1:
            yield  
            if data.v:
                out_v = gmpy2.mpz(0)   
            else:
                out_v = gmpy2.mpz(1)   
            simulator.update_v(delay, target, out_v)


@dispatch('LogicAnd', Any, [Any, None])
class _LogicAnd(Gate):
        
    id = 'LogicAnd'
    _op = '&&'
    
    def __head__(self, *args: Reader, id: str='', name: str = 'temp_logic') -> Reader: 
        args = [Signal(i) for i in args]
        assert args, 'at least 1 signal'
        assert all(isinstance(i._data, LogicData) for i in args), 'not logic signal'
        assert all(i._type._storage == 'packed' for i in args), 'signal with variable length'

        self.id: str = id or self.id
        self.inputs: List[Reader] = [self.read(i) for i in args]    # read 
        ret: Reader = UInt[1](0, name=name)
        self.output: Writer = self.write(ret)                       # write 
        return ret

    def sim_init(self):
        super().sim_init() 
        self.sim_x_count -= 1000

    def sim_vx(self):
        delay = self.timing['delay']
        data = [i._data for i in self.inputs]
        target = self.output._data 
        simulator = self._sess.sim_py   
        while 1:
            yield  
            v, x = self._bool(data) 
            zeros = [(not _v) and (not _x) for _v, _x in zip(v,x)] 
            if any(zeros):                              # exists false
                out_v, out_x = gmpy2.mpz(0),gmpy2.mpz(0)
            elif any(x):                                  # unknown         
                out_v, out_x = gmpy2.mpz(0),gmpy2.mpz(1) 
            else:                                       # all true
                out_v, out_x = gmpy2.mpz(1),gmpy2.mpz(0)  
                
            simulator.update_v(delay, target, out_v) 
            simulator.update_x(delay, target, out_x)
    
    def sim_v(self):
        delay = self.timing['delay']
        data = [i._data for i in self.inputs]
        target = self.output._data 
        simulator = self._sess.sim_py   
        while 1:
            yield   
            if all(i.v for i in data):
                simulator.update_v(delay, target, gmpy2.mpz(1)) 
            else:
                simulator.update_v(delay, target, gmpy2.mpz(0))


    def _bool(self, l: List[LogicData]):
        """ perform Or Reduce
        """
        ret_v = []
        ret_x = []
        for data in l:
            if data.v & (~data.x):
                ret_v.append(1)
                ret_x.append(0)
            elif data.x:
                ret_v.append(0)
                ret_x.append(1)
            else:
                ret_v.append(0)
                ret_x.append(0) 
        return ret_v, ret_x

    def dump_sv(self, builder: sv.ModuleSV):
        x = self._op.join(builder.get_name(i) for i in self.inputs)
        y = builder.get_name(self.output) 
        builder.Assign(self, y, x, delay=self.delay) 
    
    
@dispatch('LogicOr', Any, [Any, None])
class _LogicOr(_LogicAnd):
        
    id = 'LogicOr'
    _op = '||'
    
    def sim_init(self):
        super().sim_init() 
        self.sim_x_count -= 1000

    def sim_vx(self):
        delay = self.timing['delay']
        data = [i._data for i in self.inputs]
        target = self.output._data 
        simulator = self._sess.sim_py   

        while 1:
            yield  
            v, x = self._bool(data) 
            ones = [_v and not _x for _v, _x in zip(v,x)]
            if any(ones):    # exists True
                out_v, out_x = gmpy2.mpz(1),gmpy2.mpz(0)
            elif any(x):                                  # unknown         
                out_v, out_x = gmpy2.mpz(0),gmpy2.mpz(1) 
            else:                                      # all False
                out_v, out_x = gmpy2.mpz(0),gmpy2.mpz(0)  
                
            simulator.update_v(delay, target, out_v) 
            simulator.update_x(delay, target, out_x)
    
    def sim_v(self):
        delay = self.timing['delay']
        data = [i._data for i in self.inputs]
        target = self.output._data 
        simulator = self._sess.sim_py   
        while 1:
            yield   
            if any(i.v for i in data):
                simulator.update_v(delay, target, gmpy2.mpz(1)) 
            else:
                simulator.update_v(delay, target, gmpy2.mpz(0))


    
#------- 
# shift
#------- 
    
@dispatch('Lshift', Any, Any) 
class _Lshift(Gate):
    
    id = 'Lshift'
    _op = '<<'
    
    def __head__(self, 
            a: Reader, 
            b: Union[int, Reader, Logic, str], 
            id: str = '', 
            name: str='temp_shift'
        ):
        assert isinstance(a._data, LogicData) 
        self.id: str = id or self.id
        self.width = len(a)
        self.a = self.read(a)
        if isinstance(b, Reader): 
            assert isinstance(b._type, UIntType)
            self.b = self.read(b)
        else:
            self.b = Logic(b)
            assert self.b.v >= 0 and self.b.x == 0
        ret: Reader = UInt[self.width](0, name=name) 
        self.output: Writer = self.write(ret)
        return ret
    
    def sim_init(self):
        super().sim_init() 
        self.sim_x_count -= 1000

    def sim_vx(self):
        delay = self.timing['delay']
        mask = gmpy2.bit_mask(self.width)
        target = self.output._data 
        simulator = self._sess.sim_py  
        a = self.a._data 
        if isinstance(self.b, Reader):
            b = self.b._data 
        else:
            b = self.b
        while 1:
            yield   
            # print(f'{a} << {b}')
            if b.x:
                ret_v = gmpy2.mpz(0)
                ret_x = mask 
            else:
                ret_v = (a.v << b.v) & mask 
                ret_x = (a.x << b.v) & mask 

            simulator.update_v(delay, target, ret_v) 
            simulator.update_x(delay, target, ret_x)
    
    def sim_v(self):
        delay = self.timing['delay']
        mask = gmpy2.bit_mask(self.width)
        target = self.output._data 
        simulator = self._sess.sim_py  
        a = self.a._data 
        if isinstance(self.b, Reader):
            b = self.b._data 
        else:
            b = self.b
        while 1:
            yield    
            ret_v = (a.v << b.v) & mask 
            simulator.update_v(delay, target, ret_v) 


    def dump_cpp(self):
        raise NotImplementedError(self)

    def dump_sv(self, builder: sv.ModuleSV):
        x = f"{builder.get_name(self.a)} {self._op} {builder.get_name(self.b)}"
        y = builder.get_name(self.output) 
        builder.Assign(self, y, x, delay=self.delay) 
    
    
@dispatch('Rshift', Any, Any) 
class _Rshift(_Lshift):
    
    id = 'Rshift'
    _op = '>>'
    
    def sim_init(self):
        super().sim_init() 
        self.sim_x_count -= 1000

    def sim_vx(self):
        delay = self.timing['delay']
        mask = gmpy2.bit_mask(self.width)
        target = self.output._data 
        simulator = self._sess.sim_py  
        a = self.a._data 
        if isinstance(self.b, Reader):
            b = self.b._data 
        else:
            b = self.b
        while 1:
            yield   
            if b.x:
                ret_v = gmpy2.mpz(0)
                ret_x = mask 
            else:
                ret_v = (a.v >> b.v) & mask 
                ret_x = (a.x >> b.v) & mask 

            simulator.update_v(delay, target, ret_v) 
            simulator.update_x(delay, target, ret_x)
    
    def sim_v(self):
        delay = self.timing['delay']
        mask = gmpy2.bit_mask(self.width)
        target = self.output._data 
        simulator = self._sess.sim_py  
        a = self.a._data 
        if isinstance(self.b, Reader):
            b = self.b._data 
        else:
            b = self.b
        while 1:
            yield    
            ret_v = (a.v >> b.v) & mask 
            simulator.update_v(delay, target, ret_v) 


#-----------
# arithmetic
#-----------

@dispatch('Pos', Any, None)
class _Pos(_GateN):
    
    id = 'Pos'
    _op = ('+', '')
    
    def sim_init(self):
        super().sim_init() 
        self.sim_x_count -= 1000

    def sim_vx(self):
        delay = self.timing['delay']
        inputs = [i._data for i in self.inputs] 
        target = self.output._data 
        simulator = self._sess.sim_py  
        while 1:
            yield  
            simulator.update_v(delay, target, inputs[0].v) 
            simulator.update_x(delay, target, inputs[0].x)
    
    def sim_v(self):
        delay = self.timing['delay']
        inputs = [i._data for i in self.inputs] 
        target = self.output._data 
        simulator = self._sess.sim_py  
        while 1:
            yield  
            simulator.update_v(delay, target, inputs[0].v) 

    
    
@dispatch('Neg', Any, None)
class _Neg(_GateN):
    
    id = 'Neg'
    _op = ('-', '')
    
    def sim_init(self):
        super().sim_init() 
        self.sim_x_count -= 1000

    def sim_vx(self):
        delay = self.timing['delay']
        inputs = [i._data for i in self.inputs] 
        target = self.output._data 
        simulator = self._sess.sim_py   
        mask = gmpy2.bit_mask(self.width)
        while 1:
            yield   
            if inputs[0].x:
                out_v = gmpy2.mpz(0)
                out_x = mask 
            else:
                out_v = (-inputs[0].v) & mask 
                out_x = gmpy2.mpz(0)
            simulator.update_v(delay, target, out_v) 
            simulator.update_x(delay, target, out_x)
    
    def sim_v(self):
        delay = self.timing['delay']
        inputs = [i._data for i in self.inputs] 
        target = self.output._data 
        simulator = self._sess.sim_py  
        mask = gmpy2.bit_mask(self.width)
        while 1:
            yield   
            out_v = (-inputs[0].v) & mask 
            simulator.update_v(delay, target, out_v) 



@dispatch('Add', Any, [Any, None])
class _Add(_GateN):
        
    id = 'Add'
    _op = ('', '+')
    

    def sim_init(self):
        super().sim_init() 
        self.sim_x_count -= 1000

    def sim_vx(self):
        delay = self.timing['delay']
        inputs = [i._data for i in self.inputs] 
        target = self.output._data 
        simulator = self._sess.sim_py   
        mask = gmpy2.bit_mask(self.width)
        while 1:
            yield   
            if any(i.x for i in inputs):
                out_v = gmpy2.mpz(0)
                out_x = mask 
            else:
                out_v = sum(i.v for i in inputs) & mask 
                out_x = gmpy2.mpz(0)

            simulator.update_v(delay, target, out_v) 
            simulator.update_x(delay, target, out_x)
    
    def sim_v(self):
        delay = self.timing['delay']
        inputs = [i._data for i in self.inputs] 
        target = self.output._data 
        simulator = self._sess.sim_py  
        mask = gmpy2.bit_mask(self.width)
        while 1:
            yield   
            out_v = sum(i.v for i in inputs) & mask 
            simulator.update_v(delay, target, out_v) 
    

    
@dispatch('AddFull', Any, [Any, None])
class _AddFull(_GateN):
        
    id = 'Add'
    
    def __head__(self, a: Reader, b: Reader, id: str = '', name: str='temp_addfull'):
        assert isinstance(a._data, LogicData) and isinstance(b._data, LogicData)
        assert a._type._storage == 'packed' and b._type._storage == 'packed'
        self.id: str = id or self.id
        self.width = max(len(a), len(b)) + 1 
        self.a = self.read(a)
        self.b = self.read(b)

        ret: Reader = UInt[self.width](0, name=name) 
        self.output: Writer = self.write(ret)
        return ret

    def sim_init(self):
        super().sim_init() 
        self.sim_x_count -= 1000

    def sim_vx(self):
        delay = self.timing['delay']
        target = self.output._data 
        simulator = self._sess.sim_py   
        mask = gmpy2.bit_mask(self.width)
        a, b = self.a._data, self.b._data
        while 1:
            yield   
            if a.x or b.x:
                out_v = gmpy2.mpz(0)
                out_x = mask 
            else:
                out_v = a.v + b.v
                out_x = gmpy2.mpz(0)

            simulator.update_v(delay, target, out_v) 
            simulator.update_x(delay, target, out_x)
    
    def sim_v(self):
        delay = self.timing['delay']
        target = self.output._data 
        simulator = self._sess.sim_py   
        a, b = self.a._data, self.b._data
        while 1:
            yield   
            out_v = a.v + b.v 
            simulator.update_v(delay, target, out_v) 


    def dump_cpp(self):
        raise NotImplementedError(self)

    def dump_sv(self, builder: sv.ModuleSV):
        x = f"{builder.get_name(self.a)} + {builder.get_name(self.b)}"
        y = builder.get_name(self.output) 
        builder.Assign(self, y, x, delay=self.delay) 


@dispatch('Sub', Any, [Any, None])
class _Sub(_GateN):
        
    id = 'Sub'
    _op = ('', '-')

    def sim_init(self):
        super().sim_init() 
        self.sim_x_count -= 1000

    def sim_vx(self):
        delay = self.timing['delay']
        inputs = [i._data for i in self.inputs] 
        target = self.output._data 
        simulator = self._sess.sim_py   
        mask = gmpy2.bit_mask(self.width)
        while 1:
            yield   
            if any(i.x for i in inputs):
                out_v = gmpy2.mpz(0)
                out_x = mask 
            else: 
                res = inputs[0].v 
                for i in inputs[1:]:
                    res -= i.v
                out_v = res & mask 
                out_x = gmpy2.mpz(0)

            simulator.update_v(delay, target, out_v) 
            simulator.update_x(delay, target, out_x)
    
    def sim_v(self):
        delay = self.timing['delay']
        inputs = [i._data for i in self.inputs] 
        target = self.output._data 
        simulator = self._sess.sim_py  
        mask = gmpy2.bit_mask(self.width)
        while 1:
            yield   
            res = inputs[0].v 
            for i in inputs[1:]:
                res -= i.v
            out_v = res & mask 
            simulator.update_v(delay, target, out_v) 

    def dump_cpp(self):
        raise NotImplementedError(self)
    

@dispatch('Mul', Any, [Any, None])
class _Mul(_GateN):
        
    id = 'Mul'
    _op = ('', '*')
    
    def sim_init(self):
        super().sim_init() 
        self.sim_x_count -= 1000

    def sim_vx(self):
        delay = self.timing['delay']
        inputs = [i._data for i in self.inputs] 
        target = self.output._data 
        simulator = self._sess.sim_py   
        mask = gmpy2.bit_mask(self.width)
        while 1:
            yield   
            if any(i.x for i in inputs):
                out_v = gmpy2.mpz(0)
                out_x = mask 
            else: 
                res = inputs[0].v 
                for i in inputs[1:]:
                    res *= i.v
                out_v = res & mask 
                out_x = gmpy2.mpz(0)

            simulator.update_v(delay, target, out_v) 
            simulator.update_x(delay, target, out_x)
    
    def sim_v(self):
        delay = self.timing['delay']
        inputs = [i._data for i in self.inputs] 
        target = self.output._data 
        simulator = self._sess.sim_py  
        mask = gmpy2.bit_mask(self.width)
        while 1:
            yield   
            res = inputs[0].v 
            for i in inputs[1:]:
                res *= i.v
            out_v = res & mask 
            simulator.update_v(delay, target, out_v) 

    def dump_cpp(self):
        raise NotImplementedError(self)
    
    
@dispatch('MulFull', Any, [Any, None])
class _MulFull(Gate):
        
    id = 'Mul'
    
    def __head__(self, a: Reader, b: Reader, id: str = '', name: str='temp_mulfull'):
        assert isinstance(a._data, LogicData) and isinstance(b._data, LogicData)
        assert a._type._storage == 'packed' and b._type._storage == 'packed'
        self.id: str = id or self.id
        self.width = len(a) + len(b)
        self.a = self.read(a)
        self.b = self.read(b)

        ret: Reader = UInt[self.width](0, name=name) 
        self.output: Writer = self.write(ret)
        return ret
    
    def sim_init(self):
        super().sim_init() 
        self.sim_x_count -= 1000

    def sim_vx(self):
        delay = self.timing['delay']
        target = self.output._data 
        simulator = self._sess.sim_py   
        mask = gmpy2.bit_mask(self.width)
        a, b = self.a._data, self.b._data
        while 1:
            yield   
            if a.x or b.x:
                out_v = gmpy2.mpz(0)
                out_x = mask 
            else:
                out_v = a.v * b.v
                out_x = gmpy2.mpz(0)

            simulator.update_v(delay, target, out_v) 
            simulator.update_x(delay, target, out_x)
    
    def sim_v(self):
        delay = self.timing['delay']
        target = self.output._data 
        simulator = self._sess.sim_py   
        a, b = self.a._data, self.b._data
        while 1:
            yield   
            out_v = a.v * b.v 
            simulator.update_v(delay, target, out_v) 

    def dump_cpp(self):
        raise NotImplementedError(self)

    def dump_sv(self, builder: sv.ModuleSV):
        x = f"{builder.get_name(self.a)} * {builder.get_name(self.b)}"
        y = builder.get_name(self.output) 
        builder.Assign(self, y, x, delay=self.delay) 
    
class _Gate2(Gate):
    """ binary 
    """
    _op: str = ''   
    
    def __head__(self, a: Reader, b: Reader, id: str='', name: str='temp_2') -> Reader:
        assert isinstance(a._data, LogicData) and isinstance(b._data, LogicData) 
        assert a._type._storage == 'packed' and b._type._storage == 'packed'
        self.id: str = id or self.id
        self.width = max(len(a), len(b))
        self.a = self.read(a)
        self.b = self.read(b)

        ret: Reader = UInt[self.width](0, name=name) 
        self.output: Writer = self.write(ret)
        return ret

    def dump_cpp(self):
        raise NotImplementedError(self)

    def dump_sv(self, builder: sv.ModuleSV):
        x = f"{builder.get_name(self.a)} {self._op} {builder.get_name(self.b)}"
        y = builder.get_name(self.output) 
        builder.Assign(self, y, x, delay=self.delay) 
     
        
    
@dispatch('Floordiv', Any, Any) 
class _Floordiv(_Gate2):
    
    id = 'Floordiv'
    _op = '/'

    def sim_init(self):
        """ may produce unknown 
        """
        super().sim_init() 
        self.sim_x_count += 1000

    def sim_vx(self):
        delay = self.timing['delay']
        target = self.output._data 
        simulator = self._sess.sim_py   
        mask = gmpy2.bit_mask(self.width)
        a, b = self.a._data, self.b._data
        while 1:
            yield   
            if a.x or b.x or b.v == 0:
                out_v = gmpy2.mpz(0)
                out_x = mask 
            else:
                out_v = a.v // b.v
                out_x = gmpy2.mpz(0)

            simulator.update_v(delay, target, out_v) 
            simulator.update_x(delay, target, out_x)

    
@dispatch('Mod', Any, Any) 
class _Mod(_Gate2):
    
    id = 'Mod'
    _op = '%'

    def sim_init(self):
        """ may produce unknown 
        """
        super().sim_init() 
        self.sim_x_count += 1000

    def sim_vx(self):
        delay = self.timing['delay']
        target = self.output._data 
        simulator = self._sess.sim_py   
        mask = gmpy2.bit_mask(self.width)
        a, b = self.a._data, self.b._data
        while 1:
            yield   
            if a.x or b.x or b.v == 0:
                out_v = gmpy2.mpz(0)
                out_x = mask 
            else:
                out_v = a.v % b.v
                out_x = gmpy2.mpz(0)

            simulator.update_v(delay, target, out_v) 
            simulator.update_x(delay, target, out_x)


@vectorize
class Mux(Gate):
        
    id = 'Mux'

    def __head__(self, sel, a, b, id: str='', name: str='temp_mux') -> Reader:
        sel: Reader = Signal(sel)
        a: Reader = Signal(a)
        b: Reader = Signal(b)
        assert isinstance(a._data, LogicData) and isinstance(b._data, LogicData) 
        assert a._type._storage == 'packed' and b._type._storage == 'packed'
        self.id: str = id or self.id
        self.width = max(len(a), len(b))
        self.a = self.read(a)
        self.b = self.read(b) 
        self.sel = self.read(sel)

        ret: Reader = UInt[self.width](0, name=name) 
        self.output: Writer = self.write(ret)
        return ret

    def sim_init(self):
        super().sim_init() 
        self.sim_x_count -= 1000

    def sim_vx(self):
        delay = self.timing['delay']
        simulator = self._sess.sim_py   
        target = self.output._data 
        a, b = self.a._data, self.b._data 
        sel = self.sel._data
        while 1:
            yield    
            if sel.x:
                out_v = a.v | b.v 
                out_x = a.x | b.x | (a.v^b.v) 
            elif sel.v:
                out_v = a.v 
                out_x = a.x 
            else:
                out_v = b.v 
                out_x = b.x

            simulator.update_v(delay, target, out_v) 
            simulator.update_x(delay, target, out_x)
    
    def sim_v(self):
        delay = self.timing['delay']
        simulator = self._sess.sim_py   
        target = self.output._data 
        a, b = self.a._data, self.b._data 
        sel = self.sel._data
        while 1:
            yield   
            if sel.v:
                out_v = a.v 
            else:
                out_v = b.v
            simulator.update_v(delay, target, out_v) 



    def dump_cpp(self):
        raise NotImplementedError(self)

    def dump_sv(self, builder: sv.ModuleSV):
        x = f"{builder.get_name(self.sel)} ? {builder.get_name(self.a)} : {builder.get_name(self.b)}"
        y = builder.get_name(self.output) 
        builder.Assign(self, y, x, delay=self.delay) 

