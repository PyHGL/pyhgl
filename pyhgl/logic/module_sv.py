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
from itertools import chain, islice
from typing import Any, Dict, List, Set, Union, Tuple


from pyhgl.array import *
import pyhgl.logic.hgl_basic as hgl_basic
import pyhgl.logic.module_hgl as module_hgl
import pyhgl.logic._session as _session
import pyhgl.logic.utils as utils



class ModuleSV(HGL):
    
    """
    verilog module level representation 

    creating with PyHGL module  
    
    Building Stage:
        record part of inputs/outputs
    Emiting stage:
        append a wire to inputs 
        record verilog representation of signals & gates
    """    
    
    _sess: _session.Session

    def __init__(self, module: module_hgl.Module):
        self.module: module_hgl.Module = module
        
        # if verilog content duplicated, use other's name 
        self.module_name: str = ''
        
        # ------------------------------- 
        # io signals
        # inputs belongs to father module 
        # outputs are inner or suboutputs 
        self.inputs:  Dict[hgl_basic.Reader, None] = {}
        self.outputs: Dict[hgl_basic.Reader, None] = {}        
        self.inouts:  Dict[hgl_basic.Reader, None] = {}

        # io datas
        self.inputs_data:  Dict[hgl_basic.SignalData, None] = {}  
        self.outputs_data: Dict[hgl_basic.SignalData, None] = {}         
        self.inouts_data:  Dict[hgl_basic.SignalData, None] = {} 

        # instance names, ex. uint_0, uint_1, ...
        self.names: Dict[Any, str] = {}
        # signal : type, ex. a:logic[7:0]
        self.signals: Dict[hgl_basic.SignalData, str] = {} 
        # verilog code of gate defination. ex. assign x = a & b;
        self.gates: Dict[Union[hgl_basic.Gate, hgl_basic.SignalData], str] = {}
        
    def clear(self):
        self.module_name = ''
        self.inputs_data.clear()
        self.outputs_data.clear() 
        self.inouts_data.clear()
        self.inouts_data.clear()  

        self.names.clear()
        self.signals.clear()
        self.gates.clear()
        
    def record_io(self):
        # deal with inputs, outputs, inouts of module
        for i in self.inputs: 
            if i._data.writer is None: 
                i._data._module = self.module._position[-2]  # input belongs to father
            self._sess.verilog._solve_dependency(self.module, i) 
        for o in self.outputs: 
            if o._data.writer is None:
                o._data._module = self.module
            self._sess.verilog._solve_dependency(self.module._position[-2], o) 
        for x in self.inouts:
            assert isinstance(x._data.writer._driver, hgl_basic.Analog) 
            self._sess.verilog._solve_dependency(self.module, x)


    def get_name( self, obj: Union[
            hgl_basic.Reader, 
            hgl_basic.Writer, 
            hgl_basic.SignalData, 
            module_hgl.Module
        ]):
        """ return name of verilog instance inside this module 
        """
        if isinstance(obj, (hgl_basic.Writer, hgl_basic.Reader)):    # only consider signaldata
            obj = obj._data 

        # has name 
        if ret:=self.names.get(obj):       
            return ret 
        # new name
        if isinstance(obj, hgl_basic.SignalData): 
            if obj._module is None:     # constant
                ret = obj._dump_sv_immd()
            else:
                ret = self.new_name(obj, obj._name)  # assign a new name
            obj._dump_sv(self)                          # will call get_name
            self._sess.verilog._solve_dependency(self.module, obj)
            return ret
        elif isinstance(obj, module_hgl.Module):
            assert obj in self.module._submodules      # only submodule is allowed 
            ret = self.new_name(obj, obj._name)        # assign a new name
            return ret 
        else:
            raise TypeError(obj)


    def new_name(self, obj: Any, prefered: str) -> str:
        """ get a new name of object. ex. uint_0, sint_1, ...
        """
        assert obj not in self.names
        ret = f"{prefered}_{len(self.names)}" 
        self.names[obj] = ret 
        return ret 


    def submodule_verilog(self, m: ModuleSV) -> str: 
        """ instance of submodules
        """
        
        io = []
        for i in chain(m.inputs_data, m.outputs_data, m.inouts_data):
            name1 = m.get_name(i)
            name2 = self.get_name(i)
            io.append(f".{name1}({name2})")

        return f"{m.module_name} {self.get_name(m.module)}({','.join(io)});"
        
        
        
    def gen_body(self) -> str:  

        submodules = [self.submodule_verilog(m._module) for m in self.module._submodules]
        
        inputs = self.inputs_data
        outputs = self.outputs_data 
        inouts = self.inouts_data
        
        io: List[str] = [] 
        signals: List[str] = []
             
        for signaldata, T in self.signals.items():
            s = f'{T} {self.get_name(signaldata)}'
            if signaldata in inputs:
                io.append(f'input {s};')
            elif signaldata in outputs:
                io.append(f'output {s};')
            elif signaldata in inouts:
                io.append(f'inout {s};')
            else:
                signals.append(f'{s};')   
        
        head = ','.join(self.get_name(i) for i in chain(inputs, outputs, inouts))
        head = f'({head});'            
            
        return '\n'.join(chain([head], io, signals, [''], submodules, [''], self.gates.values(), ['']))
         
        
        
    def dump_sv(self, v: Verilog) -> None:
        """
        check whether module body is duplicated, 
        if not, add a new verliog module declaration to modules and return instance statement
        
        modules: 
        key:
            (clock, a, b);
                input clock;
                output a, b;
            endmodule
        value:
            module_names
        """
        #  module_body : modle_name
        #-----------------------
        if self._sess.verbose_verilog:
            self._sess.print(f" module_sv: {self.module}", 1)
        #-----------------------
        
        body = self.gen_body()
        self.module_name = v.insert_module(body, self.module._unique_name)
            
        #-----------------------
        if self._sess.verbose_verilog:
            self._sess.print(None, -1)
        #-----------------------
        
    def __str__(self):
        return f"module_sv:{self.module._unique_name}"
            
        
    def Assign(
        self, 
        gate: hgl_basic.Gate,
        left: Union[str, Any], 
        right:  Union[str, Any], 
        delay: int = 1,
    ) -> None:
        if not isinstance(left, str):
            left = self.get_name(left)
        if not isinstance(right, str):
            right = self.get_name(right)
        if self._sess.verilog.emit_delay:
            body = f'assign #{delay} {left} = {right};'
        else:
            body = f'assign {left} = {right};'
        self.gates[gate] = body

    def AssignOP(
        self, 
        left: Union[LocalVar, Any], 
        right: List, 
        delay: int, 
        op: str,
    ) -> None:
        left = self.get_name(left)
        op.join(self.get_name(i) for i in right)

class AST:
    pass 

class LocalVar(AST):
    pass

class Verilog(HGL): 
    """
    each session contains one instance 
    
    modules:
        map all modules in this session to declaration name
    gates:
        map all gate instance in this session to their module
    """

    _sess: _session.Session
        
    def __init__(self, sess: _session.Session):
        # record all modules, the first is global module
        self.modules: List[module_hgl.Module] = []  
        # map all synthesizable gates to their modules
        self.gates: Dict[hgl_basic.Gate, module_hgl.Module] = {}  
        # body:name,  ex. {"(a,b,c,d) input a, b; output c,d; ..." : "my_module"}
        self.module_bodys: Dict[str, str] = {}
        
        # ex. assign #2 a = b & c
        self.emit_delay: bool = False
        
    def insert_module(self, body: str, unique_name: str) -> str:
        """ insert a module declaration. if duplicated, return odd name
        """
        if (name:=self.module_bodys.get(body)) is not None:
            return name 
        else:
            self.module_bodys[body] = unique_name 
            return unique_name 
        
        
    def emit(self, path: str, *, delay: bool = False, top = False) -> None:
        """ generate verilog for all modules in this session
        
        delay:
            True: assign #1 a = b;
            False: assign a = b;
        top:
            True: emit the global dummy module
        """                 
        with self._sess:
            self.emit_delay = delay
            self.module_bodys.clear()
            # clear odd port/gate info
            for m in self.modules:
                m._module.clear() 
            # solve recorded io dependency 
            for m in self.modules:
                m._module.record_io()
            
            # record gate to verilog module
            for gate, m in self.gates.items():
                gate.dump_sv(m._module)

            # reversed order so that submodules always gen first
            if top:
                modules = reversed(self.modules)
            else:
                modules = islice(reversed(self.modules), 0, len(self.modules)-1)
            for m in modules:  
                m._module.dump_sv(self) 
                
            output = [f'module {v}{k}\nendmodule\n' for k, v in self.module_bodys.items()]
            # timing
            _v, _unit = self._sess.timing.time_info
            output.insert(0, f'`timescale 1{_unit}/{_v}{_unit}\n')
            
            with open(path, 'w', encoding='utf8') as f:
                f.write('\n'.join(output))
    

    def _solve_dependency(
        self, 
        gate: Union[hgl_basic.Gate, module_hgl.Module], 
        right: Union[hgl_basic.Reader,hgl_basic.Writer, hgl_basic.SignalData],
    ):
        """ resolve implict input/outputs
        
        gate use right as input
        
        case1: Global.a.b,   Global.a.b.c
        case2: Global.a.b.c, Global.a.b
        case3: Global.a.b.c1 Global.a.b.c2
        """

        if isinstance(right, (hgl_basic.Reader,hgl_basic.Writer)):
            right = right._data
        # skip signals without position. regard as constant
        if right._module is None:
            return 
        right_pos = right._module._position

        if isinstance(gate, hgl_basic.Gate):
            left_pos = self.gates[gate]._position 
        else:
            left_pos = gate._position 
        
        idx = 0
        for x, y in zip(left_pos, right_pos):
            if x is y:
                idx += 1 
            else:
                break

        # analog, pull out to global
        if right.writer is not None and isinstance(right.writer, hgl_basic.Analog):
            for m in left_pos[idx:]:
                m._module.inouts_data[right] = None
            for m in right_pos[idx:]:
                m._module.inouts_data[right] = None
        else:
            for m in left_pos[idx:]:
                m._module.inputs_data[right] = None

            for m in right_pos[idx:]:
                m._module.outputs_data[right] = None
            
            
    def emitGraph(self, filename:str):
        import graphviz
            
        g = graphviz.Digraph(
            'G', 
            format='svg', 
            filename=filename,
            engine='dot',
            strict=True, 
            graph_attr={'overlap':'false'}
        ) 
            
        for gate in self.gates:
            gate.emitGraph(g)

        g.save()
        return g
        


