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
import pyhgl.logic.hardware as hardware
import pyhgl.logic.hglmodule as hglmodule
import pyhgl.logic._session as _session
import pyhgl.logic.utils as utils



class VerilogModule(HGL):
    
    """
    verilog module level representation 

    creating with PyHGL module  
    
    Building Stage:
        record part of inputs/outputs
    Emiting stage:
        append a wire to inputs 
        record verilog representation of signals & gates
    """    
    
    def __init__(self, module: hglmodule.Module):
        self.sess: _session.Session = module._sess
        self.module: hglmodule.Module = module
        
        # if verilog content duplicated, use other's name 
        self.module_name: str = ''
        
        # -------------------------------
        # inputs belongs to father module 
        # outputs are inner or suboutputs 
        # store inputs/outputs, don't clear
        self.inputs:  Dict[hardware.Reader, None] = {}
        self.outputs: Dict[hardware.Reader, None] = {}        
        self.inouts:  Dict[hardware.Reader, None] = {}

        # complete io
        self.inputs_data:  Dict[hardware.SignalData, None] = {}  
        self.outputs_data: Dict[hardware.SignalData, None] = {}         
        self.inouts_data:  Dict[hardware.SignalData, None] = {} 

        # instance names, ex. uint_0, uint_1, ...
        self.names: Dict[Any, str] = {}
        # verilog code: signal
        self.signals: Dict[str, Union[hardware.SignalData, hardware.Reader]] = {} 
        # verilog code of gate defination
        self.gates: List[str] = []
        
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
        # insert a wire to leaf signal
        for i in self.inputs: 
            if i._data.writer is None:
                i: hardware.Reader = Wire(i)
                gate = i._data.writer._driver 
                self.sess._add_gate(gate, self.module._position[-2]) 
            self.sess.verilog._solve_dependency(self.module, i) 
        for o in self.outputs: 
            if o._data.writer is None:
                o: hardware.Reader = Wire(o) 
                gate = o._data.writer._driver  
                self.sess._add_gate(gate, self.module) 
            self.sess.verilog._solve_dependency(self.module._position[-2], o) 
        for x in self.inouts:
            assert isinstance(x._data.writer._driver, hardware.Analog) 
            self.sess.verilog._solve_dependency(self.module, x)
            self.sess.verilog._solve_dependency(self.module._position[-2], x)

    def get_name(self, obj: Any) -> Optional[str]:
        return self.names.get(obj) 

    def new_name(self, obj: Any, prefered: str) -> str:
        """ get a new name of object. ex. uint_0, sint_1, ...
        """
        assert obj not in self.names
        ret = f"{prefered}_{len(self.names)}" 
        self.names[obj] = ret 
        return ret 
    
    def update_name(self, obj, name: str):
        self.names[obj] = name

    def new_signal(self, signal,  s: str):                 
        self.signals[s] = signal

    def new_gate(self, s: str):
        self.gates.append(s)


    def submodule_verilog(self, m: VerilogModule) -> str: 
        """ instance of submodules
        """
        
        io = []
        for i in chain(m.inputs_data, m.outputs_data, m.inouts_data):
            signal = i.writer 
            name1 = m.module._verilog_name(signal)
            name2 = self.module._verilog_name(signal)
            io.append(f".{name1}({name2})")

        assert m.module_name
        return f"{m.module_name} {self.module._verilog_name(m.module)}({','.join(io)});"
        
        
        
    def gen_body(self) -> str:  

        submodules = [self.submodule_verilog(m._module) for m in self.module._submodules]
        
        inputs = self.inputs_data
        outputs = self.outputs_data 
        inouts = self.inouts_data
        
        io: List[str] = [] 
        signals: List[str] = []
             
        for s, data in self.signals.items():
            if isinstance(data, hardware.SignalData):
                if data in inputs:
                    io.append(f'input {s};')
                elif data in outputs:
                    io.append(f'output {s};')
                elif data in inouts:
                    io.append(f'inout {s};')
                else:
                    signals.append(f'{s};')  
            else:
                signals.append(f'{s};') 
        
        head = ','.join(self.module._verilog_name(i.writer) for i in chain(inputs, outputs, inouts))
        head = f'({head});'            
            
        return '\n'.join(chain([head], io, signals, [''], submodules, [''], self.gates, ['']))
         
        
        
    def emitVerilog(self, v: Verilog) -> None:
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
        if self.sess.verbose_verilog:
            self.sess.print(f" VerilogModule: {self.module}", 1)
        #-----------------------
        
        body = self.gen_body()
        self.module_name = v.insert_module(body, self.module._unique_name)
            
        #-----------------------
        if self.sess.verbose_verilog:
            self.sess.print(None, -1)
        #-----------------------
        
    def __str__(self):
        return f"VerilogModule:{self.module._unique_name}"
            
        



class Verilog(HGL): 
    """
    each session contains one instance 
    
    modules:
        map all modules in this session to declaration name
    gates:
        map all gate instance in this session to their module
    """
        
    def __init__(self, sess: _session.Session):
        self.sess = sess
        # record all modules, first is global module
        self.modules: List[hglmodule.Module] = []  
        # map all synthesizable gates to their modules
        self.gates: Dict[hardware.Gate, hglmodule.Module] = {}  
        # body:name,  ex. {"(a,b,c,d) input a, b; output c,d; ..." : "my_module"}
        self.module_bodys: Dict[str, str] = {}
        
        # ex. assign #2 a = b & c
        self.emit_delay: bool = False
        # TODO register regard reset as sync
        self.sync_reset: bool = False
        
    def insert_module(self, body: str, unique_name: str) -> str:
        """ insert a module declaration. if duplicated, return odd name
        """
        if (name:=self.module_bodys.get(body)) is not None:
            return name 
        else:
            self.module_bodys[body] = unique_name 
            return unique_name 
        
    def insert_raw(self, v: str) -> str:
        """ input a full module, return its name 
        
        ex. module inner(); ... endmodule  
            return valid name
        """
        # TODO
        
    def emit(self, path: str, *, delay: bool = False, top = False) -> None:
        """ generate verilog for all modules in this session
        
        delay:
            True: assign #1 a = b;
            False: assign a = b;
        top:
            True: emit the global dummy module
        """                 
        with self.sess:
            self.emit_delay = delay
            self.module_bodys.clear()
            # clear io of global module
            self.modules[0]._module.inputs.clear()
            self.modules[0]._module.outputs.clear()
            # clear odd port/gate info
            for m in self.modules:
                m._module.clear() 
            # solve recorded io dependency 
            for m in self.modules:
                m._module.record_io()
            
            # record gate to verilog module
            for gate, m in self.gates.items():
                m._module.new_gate(gate.emitVerilog(self))

            # reversed order so that submodules always gen first
            if top:
                modules = reversed(self.modules)
            else:
                modules = islice(reversed(self.modules), 0, len(self.modules)-1)
            for m in modules:  
                m._module.emitVerilog(self) 
                
            output = [f'module {v}{k}\nendmodule\n' for k, v in self.module_bodys.items()]
            # timing
            _v, _unit = self.sess.timing.time_info
            output.insert(0, f'`timescale 1{_unit}/{_v}{_unit}\n')
            
            with open(path, 'w', encoding='utf8') as f:
                f.write('\n'.join(output))
    

    def _solve_dependency(
        self, 
        gate: Union[hardware.Gate, hglmodule.Module], 
        right: hardware.Reader
    ):
        """ resolve implict input/outputs
        
        gate use right as input
        
        case1: Global.a.b,   Global.a.b.c
        case2: Global.a.b.c, Global.a.b
        case3: Global.a.b.c1 Global.a.b.c2
        """

        # skip signals without driver. regard as constant
        if right._data.writer is None:
            return 

        if isinstance(gate, hardware.Gate):
            left_pos = self.gates[gate]._position 
        else:
            left_pos = gate._position 
            
        right_driver: hardware.Gate = right._data.writer._driver
        right_pos = self.gates[right_driver]._position  
        
        idx = 0
        for x, y in zip(left_pos, right_pos):
            if x is y:
                idx += 1 
            else:
                break

        # analog, pull out to global
        if isinstance(right_driver, hardware.Analog):
            for m in left_pos[idx:]:
                m._module.inouts_data[right._data] = None
            for m in right_pos[idx:]:
                m._module.inouts_data[right._data] = None
        else:
            for m in left_pos[idx:]:
                m._module.inputs_data[right._data] = None

            for m in right_pos[idx:]:
                m._module.outputs_data[right._data] = None
            
            
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
        


