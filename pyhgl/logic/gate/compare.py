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

from pyhgl.array import *
from pyhgl.logic.hgl_basic import *
import pyhgl.logic.utils as utils 
from pyhgl.logic.hgl_assign import __hgl_when__, __hgl_elsewhen__, __hgl_partial_assign__
from pyhgl.logic.gate.common import _align



# TODO support unfixed signal width


@dispatch('Eq', Any, Any) 
class _Eq(Gate):
    
    id = 'Eq'
    
    def __head__(self, left: Reader, right: Union[int,str,list,dict,Reader,BitPat], id='') -> Reader:
        """ 
        """
        self.id = id or self.id 
        ret = UInt[1](0)
        self.output = self.write(ret)

        if isinstance(left, Reader) and isinstance(right, Reader):
            left, right = _align(left, right)
            self.left = self.read(left)
            self.right = self.read(right) 
            return ret
        else:
            self.left = self.read(left)
            self.right = self.left._type._eval(right)
            return ret
    
    def forward(self): 
        left, right = self.left, self.right 
        if isinstance(right, Reader):
            right = right._getval_py()
        
        v = gmpy2.mpz(right == left._getval_py())
        self.output._setval_py(v, dt=self.delay)


    def emitVerilog(self, v) -> str:
        """ 
        ex.
            assign y = a & b & 3'd5;
            assign #1 y = a & b;
        """
        x1 = self.get_name(self.left)
        y = self.get_name(self.output)
        
        if self._sess.verilog.emit_delay:
            delay = '#' + str(self.timing["delay"]) 
        else:
            delay = ''
        
        if isinstance(self.right, Reader):
            x2 = self.get_name(self.right)
            return f'assign {delay} {y} = {x1} == {x2};' 
        elif isinstance(self.right, BitPat):
            return f'assign {delay} {y} = {x1} ==? {self.right._verilog()};' 
        else:
            return f'assign {delay} {y} = {x1} == {self.right};'
        
    def emitGraph(self, g):
        inputs = [self.left]
        if isinstance(self.right, Reader):
            inputs.append(self.right)
        self._emit_graphviz(g, inputs, [self.output], self.id)
        
        
@dispatch('Ne', Any, Any) 
def _Ne(a, b, id=''):
    return Not(_Eq(a,b,id=id)) 


@dispatch('Lt', Any, Any) 
class _Lt(Gate):
    
    id = 'Lt'
    
    def __head__(self, left: Reader, right: Union[int,str,list,dict,Reader], id='') -> Reader:
        self.id = id or self.id
        
        self.left = self.read(left)
        
        if isinstance(right, Reader):
            self.right = self.read(right)
        elif isinstance(right, BitPat):
            raise ValueError('BitPat not valid comparation target')
        else:
            self.right = self.left._type._eval(right)
            
        ret = UInt[1](0)
        self.output = self.write(ret)
        return ret
    
    def forward(self): 
        left, right = self.left, self.right 
        
        if isinstance(right, Reader):
            right = right._getval_py()
            
        v = gmpy2.mpz(left._getval_py() < right)
        self.output._setval_py(v, dt=self.delay)


    def emitVerilog(self, v) -> str:
        """ 
        ex.
            assign y = a & b & 3'd5;
            assign #1 y = a & b;
        """
        x1 = self.get_name(self.left)
        y = self.get_name(self.output)
        
        if self._sess.verilog.emit_delay:
            delay = '#' + str(self.timing["delay"]) 
        else:
            delay = ''
        
        if isinstance(self.right, Reader):
            x2 = self.get_name(self.right)
            return f'assign {delay} {y} = {x1} < {x2};' 
        else:
            return f'assign {delay} {y} = {x1} < {self.right};'
        
    def emitGraph(self, g):
        inputs = [self.left]
        if isinstance(self.right, Reader):
            inputs.append(self.right)
        self._emit_graphviz(g, inputs, [self.output], self.id) 
        
        
@dispatch('Gt', Any, Any) 
class _Gt(_Lt):
    
    id = 'Gt'
    
    def forward(self): 
        left, right = self.left, self.right 
        
        if isinstance(right, Reader):
            right = right._getval_py()
            
        v = gmpy2.mpz(left._getval_py() > right)
        self.output._setval_py(v, dt=self.delay)


    def emitVerilog(self, v) -> str:
        """ 
        ex.
            assign y = a & b & 3'd5;
            assign #1 y = a & b;
        """
        x1 = self.get_name(self.left)
        y = self.get_name(self.output)
        
        if self._sess.verilog.emit_delay:
            delay = '#' + str(self.timing["delay"]) 
        else:
            delay = ''
        
        if isinstance(self.right, Reader):
            x2 = self.get_name(self.right)
            return f'assign {delay} {y} = {x1} > {x2};' 
        else:
            return f'assign {delay} {y} = {x1} > {self.right};'
        

        
@dispatch('Le', Any, Any) 
def _Le(a, b, id=''):
    return Not(_Gt(a,b,id=id)) 


        
@dispatch('Ge', Any, Any) 
def _Ge(a, b, id=''):
    return Not(_Lt(a,b,id=id))  






def MuxSel(sel: Reader, default: Any, table: dict) -> Any:
    """ ex. 
    MuxSel(x, [UInt('111'), UInt('00')], {
        0: [1,2],
        1: [3,4],
        '1??': [5,6]
    })

    MuxSel(x, UInt[4](0), [4,5,6,7])
    """ 
    assert isinstance(sel, Reader) 
    if isinstance(table, list):
        table = {i:v for i, v in enumerate(table)}
    return MuxSeq(default=default, table={sel==k:v for k, v in table.items()})


def MuxSeq(default: Any, table: dict ) -> Any:
    """ ex. 
    MuxSeq([UInt('111'), UInt('00')], {
        x == 0: [1,2],
        x == 1: [3,4],
        x == '1??': [5,6]
    })
    
    """
    default = ToArray(default) 
    ret = WireNext(default)
    sequence = [(cond,v) for cond, v in table.items()] 
    cond0, value0 = sequence[0]
    with __hgl_when__(cond0): 
        __hgl_partial_assign__(ret, ToArray(value0))
    for cond, value in sequence[1:]:
        with __hgl_elsewhen__(cond):
            __hgl_partial_assign__(ret, ToArray(value))
    return ret




