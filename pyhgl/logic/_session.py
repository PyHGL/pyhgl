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
from typing import  Optional, Dict, Any, List, Tuple, Generator, Callable, Literal

import io
import os
import inspect
import traceback
import time
import shutil

import pyhgl.logic.utils as utils 
import pyhgl.tester.utils as tester_utils

from pyhgl.array import * 
import pyhgl.logic._config as config
import pyhgl.logic.hgl_core as hgl_core
import pyhgl.logic.module_sv as module_sv  
import pyhgl.logic.simulator_py as simulator_py
import pyhgl.logic.module_hgl as module_hgl
import pyhgl.logic.module_cpp as module_cpp


class Session(HGL):
    
    __slots__ = '__dict__'
    
    def __init__(
            self, 
            conf: config.HGLConf = None, 
            *,
            backend: Literal['python','cpp'] = 'python',
            build_dir: str = './build/',
            verbose_conf = False,
            verbose_hardware = False,
            verbose_verilog = False,
            verbose_sim = False,
            verbose_trace = False,
        ) -> None:
        
        self.verbose_conf       = verbose_conf 
        self.verbose_hardware   = verbose_hardware 
        self.verbose_verilog    = verbose_verilog
        self.verbose_sim        = verbose_sim  
        self.verbose_trace      = verbose_trace
        self.enable_assert = False 
        self.backend = backend 
        self.build_dir = tester_utils.relative_path(build_dir, 2)

        self._intend = 0    
        self._prev_sess = []
        
        # current module
        self.module: module_hgl.Module = None 
        # current simulator, unique
        self.sim_py = simulator_py.Simulator(self)  
        self.sim_cpp = module_cpp.Cpp(self)
        # verilog emitter, unique 
        self.verilog = module_sv.Verilog(self) 
        # timing info
        self.timing = config.TimingConf()   
        # waveform
        self.waveform = simulator_py.VCD(self)  
        # build/sumulate warnings
        self.log = _Logging()
        
        #-------------------------------------------------------
        if self.verbose_conf:
            print(tester_utils._fill_terminal('┏', '━', 'left')) 
            self.print(f"✭ New Session with config: {conf}")
        #-------------------------------------------------------        
        with self:
            # a dummy module before the real toplevel module 
            m = object.__new__(module_hgl.Module) 
            m.__head__()  
            self.module = m 
            m.dispatcher = default_dispatcher.copy # default dispatcher
            
            if conf is None:
                conf_ret = ({},{})       # parameter, subconfig
            else: 
                conf_ret = conf.exec()

            self.timing.update({})      # default timescale 
            
            new_paras: Dict[str, Any] = conf_ret[0]
            subconfigs: Dict[config.HGLConf, None] = conf_ret[1]
            # default clock and reset
            # clock: (clk, 0|1)  reset: (rst, 0|1)  
            if self.module.clock is None:
                self.module.clock = (hgl_core.Clock(), 1)  
            if self.module.reset is None:
                self.module.reset = (hgl_core.Wire(hgl_core.UInt(0, name='reset')), 1)  

            self.module._subconfigs.update(subconfigs)        
            self.module._conf = type("Conf_Global", (config.ModuleConf,), new_paras)

    
    def _new_module(self, m: module_hgl.Module) -> List[module_hgl.Module]:
        """ register to father module and return position in module tree
        """
        self.verilog.modules.append(m) 
        if self.module is None:             # root
            return [m]
        else:
            self.module._submodules[m] = None           # module tree
            return self.module._position + [m]          # position
        
    def _add_gate(self, gate: hgl_core.Gate, module: module_hgl.Module = None):
        """ verilog, mapping from (part of) gates to module
        """
        assert isinstance(gate, hgl_core.Gate)
        if module is None:
            module = self.module
        self.verilog.gates[gate] = module  
        
    def _remove_gate(self, gate: hgl_core.Gate):    
        if gate in self.verilog.gates:
            self.verilog.gates.pop(gate) 
            
    def print(self, obj: object, intend: int = 0):
        """ print for debug
        """
        if intend > 0:
            self._intend += intend 
            print(tester_utils._fill_terminal(self._intend * '┃  ' + '┏', '━', 'left'))
        elif intend < 0:
            print(tester_utils._fill_terminal(self._intend * '┃  ' + '┗', '━', 'left'))
            self._intend += intend 
        if obj is None: return 
        s = str(obj)
        i =  self._intend * '┃  ' + '┃'  
        print(i + f'\n{i}'.join(s.splitlines()))
                
    def enter(self):
        self._prev_sess.append(self._sess)
        self._sess = self 
        
    def exit(self):
        self._sess = self._prev_sess.pop()
        
    def __enter__(self):
        self.enter()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.exit()
    
    def dumpVCD(self, filename='test.vcd'):
        with self:
            if filename[-4:] != '.vcd':
                raise ValueError('invalid filename: use xxx.vcd')  
            
            filename = self._get_filepath(filename)
            self.waveform._dump(filename)
            print(tester_utils._yellow('Waveform: ') + filename)
            return filename
    
    
    def dumpVerilog(self, filename='test.sv', **kwargs):
        with self:
            if not (filename[-3:] == '.sv' or filename[-2:] == '.v'):
                raise ValueError('invalid filename')
            filename = self._get_filepath(filename)
            self.verilog.dump(filename, **kwargs) 
            print(tester_utils._yellow('Verilog: ') + filename)
            return 
    
    
    def dumpGraph(self, filename='test.gv'): 
        with self:
            if not (filename[-3:] == '.gv' or filename[-4:] == '.dot'):
                raise ValueError('invalid filename')
            filename = self._get_filepath(filename)
            ret = self.verilog.dumpGraph(filename) 
            print(tester_utils._yellow('Graphviz: ') + filename)
            return ret 

    def _get_filepath(self, relative_path: str) -> str:
        """ path relative to build directory
        """
        if os.path.isabs(relative_path):
            filepath = relative_path 
        else:
            filepath = os.path.abspath(os.path.join(self.build_dir, relative_path))
        directory = os.path.dirname(filepath)
        if os.path.isdir(directory):
            return filepath 
        else:
            os.makedirs(directory)
            return filepath
        

    def __str__(self):
        ret = [tester_utils._yellow("Summary: ")]
        ret.append(f'  n_modules: {len(self.verilog.modules)}')
        ret.append(f'  n_gates: {len(self.verilog.gates)}')
        ret.append(f'  t: {self.sim_py.t}')
        ret.append(str(self.log))
        return '\n'.join(ret)
                
    # waveform 
    def track(self, *args, **kwargs):
        self.waveform._track(*args, **kwargs)
        
    def run(self, dt: Union[int, str]) -> None:
        self.enter()
        if isinstance(dt, str):
            dt = self.timing._get_timestep(dt)
        if self.backend == 'python':
            if self.sim_py.t == 0:
                self.sim_py.init()
            self.sim_py.step(dt)
        else:
            if self.sim_cpp.dll is None:
                self.sim_cpp.dump() 
                self.sim_cpp.build()
            self.sim_py.step_cpp(dt)
        self.exit() 
        
    def task(self, *args):
        self.enter()
        for g in args:
            assert inspect.isgenerator(g)
            self.sim_py.insert_coroutine_event(0, g) 
        self.exit()

    def test_iverilog(self):
        with self:
            self.verilog.sim_iverilog()





class _Logging:
    def __init__(self) -> None:
        self.warnings = []
        self.n_exec_gates = 0
        
    def warning(self, msg: str, start: int = 3, end: int = 100):
        x = ['Traceback:\n', utils.format_hgl_stack(start, end)]
        msg = '  ' + '  '.join(msg.splitlines(keepends=True))
        x.insert(0, f"{tester_utils._red('Warning:')}\n{msg}\n")
        self.warnings.append(''.join(x)) 

    def error(self, msg):
        ... 
        
    def info(self, msg):
        ...
        
    def __str__(self) -> str:
        ret = [f'  n_exec_gates: {self.n_exec_gates}']
        for i in self.warnings:
            ret.append('─────────────────────────────────────────────────')
            ret.append(str(i))
        return '\n'.join(ret)

