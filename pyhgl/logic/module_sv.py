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
import gmpy2
import os 
import subprocess
import time 

from pyhgl.array import *
import pyhgl.logic.hgl_core as hgl_core
import pyhgl.logic.module_hgl as module_hgl
import pyhgl.logic._session as _session
import pyhgl.logic.utils as utils
import pyhgl.tester.utils as tester_utils


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
        self.inputs:  Dict[hgl_core.Reader, None] = {}
        self.outputs: Dict[hgl_core.Reader, None] = {}        
        self.inouts:  Dict[hgl_core.Reader, None] = {}

        # io datas
        self.inputs_data:  Dict[hgl_core.SignalData, None] = {}  
        self.outputs_data: Dict[hgl_core.SignalData, None] = {}         
        self.inouts_data:  Dict[hgl_core.SignalData, None] = {} 

        # instance names, ex. uint_0, uint_1, ...
        self.names: Dict[Any, str] = {}
        # {signal : type}, ex. {a:'logic[7:0] {}'}
        self.signals: Dict[hgl_core.SignalData, str] = {} 
        # verilog code of gate defination. ex. assign x = a & b;
        self.gates: Dict[Union[hgl_core.Gate, hgl_core.SignalData], str] = {} 
        # verilog code of unsynthesizable bolcks. ex. initial begin ... end 
        self.extra_blocks: Dict[Union[hgl_core.Gate, hgl_core.SignalData], str] = {} 
        
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
        # TODO record _module first
        # deal with inputs, outputs, inouts of module 
        up_module = self.module._position[-2]._module
        for i in self.inputs: 
            if i._data.writer is None and i._data._module is None: 
                i._data._module = self.module._position[-2]  # input belongs to father
            self.get_name(i) 
        for o in self.outputs: 
            if o._data.writer is None and o._data._module is None:
                o._data._module = self.module
            self.get_name(o)
            up_module.get_name(o)
        for x in self.inouts:
            assert isinstance(x._data.writer._driver, hgl_core.Analog) 
            self.get_name(x)


    def get_name( self, obj: Union[
            hgl_core.Reader, 
            hgl_core.Writer, 
            hgl_core.SignalData, 
            module_hgl.Module
        ]):
        """ return name of verilog instance inside this module 
        """
        if isinstance(obj, (hgl_core.Writer, hgl_core.Reader)):    # only consider signaldata
            obj = obj._data 

        # has name 
        if ret:=self.names.get(obj):       
            return ret 
        # new name
        if isinstance(obj, hgl_core.SignalData): 
            if obj._module is None:     # constant
                ret = self._sv_immd(obj)
            else:
                ret = self._new_name(obj, obj._name)  # assign a new name
                obj._dump_sv(self)                          # will call get_name
                self._sess.verilog._solve_dependency(self.module, obj)
            return ret
        elif isinstance(obj, module_hgl.Module):
            assert obj in self.module._submodules      # only submodule is allowed 
            ret = self._new_name(obj, obj._name)        # assign a new name
            return ret 
        elif isinstance(obj, (hgl_core.Logic, hgl_core.BitPat, int, gmpy2.mpz)):
            return str(obj)
        else:
            raise TypeError(f'{type(obj)}({obj})')

    def _sv_immd(self, data: hgl_core.SignalData) -> str:
        width = len(data)
        return utils.logic2str(data.v, data.x, width=width)

    def _new_name(self, obj: Any, prefered: str) -> str:
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
        io_signals = {}
        io_signals.update(inputs)
        io_signals.update(outputs)
        io_signals.update(inouts) 
        assert len(io_signals) == len(inputs) + len(outputs) + len(inouts), 'duplicated io'
        
        head = ','.join(self.get_name(i) for i in io_signals)
        head = f'({head});'    

        io: List[str] = [] 
        signals: List[str] = []

        for signaldata, T in self.signals.items():
            s = T.format(self.get_name(signaldata))
            if signaldata in inputs:
                io.append(f'input {s};')
            elif signaldata in outputs:
                io.append(f'output {s};')
            elif signaldata in inouts:
                io.append(f'inout {s};')
            else:
                signals.append(f'{s};')    
        
        if len(self.module._position) == 1:
            gates = self.extra_blocks.values()
        else:
            gates = chain(self.gates.values(), self.extra_blocks.values()) 
            
        return '\n'.join(chain([head], io, signals, [''], submodules, [''], gates, ['']))
        
        
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
            
    # TODO top module only allow initial block and blackbox 
    def Assign(
        self, 
        gate: hgl_core.Gate,
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

    def Block(self, gate: hgl_core.Gate, body: str):
        self.gates[gate] = body 

    def Task(self, gate: hgl_core.Gate, body: str): 
        self.extra_blocks[gate] = body
    



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
        # record all modules, the first is the global module
        self.modules: List[module_hgl.Module] = []  
        # map from gate to module
        self.gates: Dict[hgl_core.Gate, module_hgl.Module] = {}  
        # {body:name},  ex. {"(a,b,c,d) input a, b; output c,d; ..." : "my_module"}
        self.module_bodys: Dict[str, str] = {}
        
        # ex. assign #2 a = b & c;
        self.emit_delay: bool = False 
        # dumped file
        self.filepath: str = None
        
    def insert_module(self, body: str, unique_name: str) -> str:
        """ insert a module declaration. if duplicated, return odd name
        """
        if (name:=self.module_bodys.get(body)) is not None:
            return name 
        else:
            self.module_bodys[body] = unique_name 
            return unique_name 
        
        
    def dump(self, path: str, *, delay: bool = False, top = False) -> None:
        """ generate verilog for all modules in this session
        
        delay:
            True: assign #1 a = b;
            False: assign a = b;
        top:
            True: dump the global dummy module
        """                 
        with self._sess:
            self.emit_delay = delay
            self.module_bodys.clear()
            # clear odd port/gate info
            for m in self.modules:
                m._module.clear() 
            # solve recorded io dependency 
            for m in self.modules[1:]:
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
            self.filepath = path
    

    def _solve_dependency(
        self, 
        gate: Union[hgl_core.Gate, module_hgl.Module], 
        right: Union[hgl_core.Reader,hgl_core.Writer, hgl_core.SignalData],
    ):
        """ resolve implict input/outputs
        
        gate use right as input
        
        case1: Global.a.b,   Global.a.b.c
        case2: Global.a.b.c, Global.a.b
        case3: Global.a.b.c1 Global.a.b.c2
        """

        if isinstance(right, (hgl_core.Reader,hgl_core.Writer)):
            right = right._data
        # skip signals without position. regard as constant
        if right._module is None:
            return 
        right_pos = right._module._position 

        if isinstance(gate, hgl_core.Gate):
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
        if right.writer is not None and isinstance(right.writer, hgl_core.Analog):
            for m in left_pos[idx:]:
                m._module.inouts_data[right] = None
            for m in right_pos[idx:]:
                m._module.inouts_data[right] = None
        else:
            for m in left_pos[idx:]:
                m._module.inputs_data[right] = None

            for m in right_pos[idx:]:
                m._module.outputs_data[right] = None
            
            
    def dumpGraph(self, filename:str):
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
            gate.dumpGraph(g)

        g.save()
        return g 
    
    def sim_iverilog(self):
        """ test third-party simulator on dumped verilog """
        filename = self.filepath 
        if filename is None or not os.path.isfile(filename):
            return 
        if filename[-3:] == '.sv':
            target = filename[:-3]  
        elif filename[-2:] == '.v':
            target = filename[:-2] 
        else:
            raise Exception(f'invalid filename {filename}')
        
        t = time.time() 
        subprocess.run(['iverilog', '-g2012', '-o', f'{target}.vvp', filename])
        # os.system(f'iverilog -g2012 -o {target}.vvp {filename}')
        print(f'{tester_utils._yellow("iverilog_compile:")} {filename} -> {target}.vvp, cost {time.time()-t} s')
        t = time.time() 
        subprocess.run(['vvp', f'{target}.vvp'], cwd=self._sess.build_dir)
        # os.system(f'vvp {target}.vvp')
        print(f'{tester_utils._yellow("iverilog_sim:")} {time.time()-t} s') 



