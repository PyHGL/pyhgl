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
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union, Type

import inspect

from pyhgl.array import *
import pyhgl.logic._session as _session
import pyhgl.logic.verilogmodule as verilogmodule
import pyhgl.logic.config as hglconfig
import pyhgl.logic.hardware as hardware


 
def module(f):
    """ PyHGL module decorator
    
    return a subclass of `Module` 

    usage:
        @module name(*args, **kwargs):
            ... 
    or:
        @module name(self, *args, **kwargs):
            ... 
    """
    args = list(inspect.signature(f).parameters.keys())
    if args and args[0] == 'self':
        return type(f.__name__, (Module,),{'__body__': f})
    else:
        return type(f.__name__, (Module,),{'__body__': staticmethod(f)})  


class MetaModule(type):
    """ set config id of module instance
    
    ex. Adder['adder3','xxx'](8)
    """
    def __getitem__(self, keys: str):
        for i in keys:
            if not isinstance(i, str):
                raise KeyError(f'invalid module id: {i}')
        m: Module = object.__new__(self)
        m.__head__()
        m._ids = list(keys)
        return m 



class Module(Container, metaclass=MetaModule):
    
    clock: Tuple[hardware.Reader, int]
    reset: Tuple[hardware.Reader, int]

    __slots__ = '_sess', 'dispatcher', '__dict__'
    
    def __head__(self):  
        self._sess: _session.Session = HGL._sess 
        if not isinstance(self._sess, _session.Session):
            raise Exception('instantiating module under no Session')
        # prefered name (instance name, ex. adder)
        self._name = self.__class__.__name__.lower() 
        # unique name (declaration name, ex. Adder_1)
        self._unique_name = self._sess._new_module_name(self.__class__.__name__)
        # position in module tree (include self)
        self._position: List[Module] = self._sess._new_module_instance(self) 
        # submodules 
        self._submodules: Dict[Module, None] = {}
        # identifiers to be matched for configs
        self._ids: List[str] = [self._name]
        # store module level parameters, subclass of `hglconfig.ModuleConf`
        self._conf: Type[hglconfig.ModuleConf] = None
        # lazy-callables for child modules, ordered
        self._subconfigs: Dict[hglconfig.HGLConf, None] = {}
        # VerilogModule necessary for generating verilog 
        self._module = verilogmodule.VerilogModule(self) 

        # default module level parameters
        # dispatcher
        self.dispatcher: Dispatcher = None
        # parent module 
        self.up: Module = None 
        # io 
        self.io: Array = None 

        # temp stack
        self._prev: List[Module] = []
        self._temp_inputs: List[hardware.Reader] = []

    def __conf__(self): 
        """
        override: outside config tree < in module parameters
        """
        self.up = self._sess.module
        up_conf = self.up._conf
        self.dispatcher = self.up.dispatcher
            
        # record matched callables
        matched_configs: List[hglconfig.HGLConf] = []
        inherit: bool = True
        for config in self.up._subconfigs: 
            for _id in self._ids:
                if config.filter.match(_id):
                    matched_configs.append(config) 
                    break
            # config always valid
            if config.always:
                self._subconfigs[config] = None
            if not config.inherit:
                inherit = False
                    
        # ---------------------------------
        if self._sess.verbose_conf:
            self._sess.print(f"Module: {self._unique_name}", 1)
            self._sess.print(f"id: {', '.join(self._ids)}")
            self._sess.print(f"matched configs: {', '.join(str(i) for i in matched_configs)}")
        # ---------------------------------             
        
        with self:
            # three paras always inherit        
            paras = {
                'clock': self.up.clock,
                'reset': self.up.reset,
            }
            
            for config in matched_configs:
                new_paras, subconfigs = config.exec()
                paras.update(new_paras) 
                self._subconfigs.update(subconfigs)
                
            # ---------------------------------
            if self._sess.verbose_conf:
                self._sess.print(f"subconfigs: {', '.join(str(i) for i in self._subconfigs)}")
                _paras = {k:v for k,v in paras.items() if k != 'dispatcher'}
                self._sess.print(f"new_paras: {_paras}")
            # ---------------------------------
            
            bases = (up_conf,) if inherit else (hglconfig.ModuleConf,) 
            self._conf = type(f"Conf_{self._unique_name}", bases, paras) 
        

    def __body__(*args, **kwargs): 
        '''your hdl module definations here'''


    def __tail__(self, f_locals:dict):  
        # delete dummy gate
        for i in self._temp_inputs:
            driver = i._data.writer._driver
            assert isinstance(driver, hardware.DummyGate), 'unexcepted assignment to inputs'  
            driver.delete()

        # record 
        io = None
        for k, v in f_locals.items():
            if k == 'io':
                io = v 
            elif k[0] != '_': 
                if k not in self.__dict__:
                    self.__dict__[k] = v 
                if isinstance(v, hardware.Reader):
                    # update prefered name
                    v._name = k
                elif isinstance(v, Module) and v.up is self:
                    # update prefered name
                    v._name = k 
                elif isinstance(v, Array):
                    _rename_array(v, k)

        if io is not None:
            self.io = io 
            def make_io(x):
                assert isinstance(x, hardware.Reader)
                if x._direction == 'input':
                    assert x._data.writer is None 
                    self._sess.module._module.inputs[x] = None
                elif x._direction in ['inner','output']:
                    x._direction = 'output' 
                    self._sess.module._module.outputs[x] = None 
                    if x._data.writer is None:
                        Wire(x)  
                elif x._direction == 'inout':
                    self._sess.module._module.inouts[x] = None
            Map(make_io, self.io) 
            _rename_array(self.io, 'io')


        # ---------------------------------
        if self._sess.verbose_conf:
            self._sess.print(None, -1)
        # ---------------------------------

    def __new__(cls, *args, **kwargs) -> Module:
        m = object.__new__(cls)
        m.__head__()
        m.__conf__()
        with m:
            f_locals = m.__body__(*args, **kwargs)
            m.__tail__(f_locals)             
        return m 
    
    def __call__(self, *args, **kwargs) -> Module:
        if self._conf is not None:
            raise Exception('non-empty module is not callable')
        self.__conf__() 
        with self:
            f_locals = self.__body__(*args, **kwargs)
            self.__tail__(f_locals)
        return self 
    
    def __enter__(self):
        if HGL._sess is not self._sess:
            raise Exception('invalid session')
        self._prev.append(self._sess.module)
        self._sess.module = self 
        
    def __exit__(self, exc_type, exc_value, traceback): 
        self._sess.module = self._prev.pop()
    

    def _format_slice(self, key: Union[int, str, list]) -> Union[str, Iterable]:
        """ 
        return:
            str:
                single item 
            list|dict :
                multi items
        """
        if isinstance(key, str):
            return self.__dict__, key
        elif isinstance(key, list): 
            return self.__dict__, key
        elif isinstance(key, dict):
            return self.__dict__, key 
        elif key is None:
            return [self], [0] 
        elif key is ...:
            return self.__dict__, {k:k for k in list(self.__dict__.keys())} 
        else:
            raise KeyError(key) 


    def __setitem__(self, keys, value):
        raise AttributeError('__setitem__ invalid for module')
    
    
    def __getattr__(self, k: str) -> Any:
        try:
            return getattr(self._conf, k)
        except:
            raise AttributeError(f'{k} is neither Module attribute nor Config parameter')


    def __str__(self) -> str:
        """ return full name, ex. Global.Top.Adder_1
        """
        return '.'.join(i._unique_name for i in self._position)


    def _verilog_name(
        self, 
        obj: Union[hardware.Reader, hardware.Writer, verilogmodule.VerilogModule]
    ) -> str: 
        """ return instance name inside this module 
        """
        if isinstance(obj, hardware.Writer):
            # name already exists
            if ret:=self._module.get_name(obj._data):
                return ret  
            # make a new name
            else:
                return obj._type._verilog_name(obj._data, self._module) 
        elif isinstance(obj, hardware.Reader):
            # return constant for unrecorded constant signal
            if obj._data.writer is None:
                return obj._type._verilog_immd(obj._type._getimmd(obj._data, None)) 
            # name already exists
            elif ret:=self._module.get_name(obj): 
                return ret   
            # if no casting, return odd name, else new name
            else:
                self._sess.verilog._solve_dependency(self, obj) 
                if not self._module.get_name(obj._data):
                    obj._type._verilog_name(obj._data, self._module)
                return obj._type._verilog_name(obj, self._module)
        elif isinstance(obj, Module):
            assert obj in self._submodules 
            if ret:=self._module.get_name(obj._module):
                return ret 
            else:
                ret = self._module.new_name(obj._module, obj._name) 
                return ret
        else:
            raise TypeError(obj) 


def _rename_array(obj: Union[Array, hardware.Reader], name: str):
    if isinstance(obj, hardware.Reader):
        obj._name = name
    elif isinstance(obj, Array):
        for k, v in obj._items():
            new_name =  f"{name}_{k}"
            _rename_array(v, new_name)
    else:
        return 
    