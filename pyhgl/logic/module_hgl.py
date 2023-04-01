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
import pyhgl.logic.module_sv as module_sv
import pyhgl.logic._config as hglconfig
import pyhgl.logic.hgl_core as hgl_core


 
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
    """ syntax setting config id of module
    
    ex. Adder['adder3','xxx'](w=8)
    """
    def __getitem__(self, keys: str):
        for i in keys:
            if not isinstance(i, str):
                raise KeyError(f'invalid module id: {i}')
        m: Module = object.__new__(self)
        m.__head__() 
        if isinstance(keys, str):
            keys = [keys]
        m._ids = list(keys)
        return m 



class Module(Container, metaclass=MetaModule):
    

    __slots__ = '_sess', 'dispatcher', 'clock', 'reset', 'io', '__dict__'
    
    def __head__(self):  
        self._sess: _session.Session = HGL._sess 
        if not isinstance(self._sess, _session.Session):
            raise Exception('instantiating module under no Session')
        # prefered name (instance name, ex. adder)
        self._name = self.__class__.__name__.lower() 
        # unique name (declaration name, ex. Adder_1)
        self._unique_name = f'{self.__class__.__name__}_{len(self._sess.verilog.modules)}'
        # position in module tree (include self)
        self._position: List[Module] = self._sess._new_module(self) 
        # submodules 
        self._submodules: Dict[Module, None] = {}
        # identifiers to be matched for configs
        self._ids: List[str] = [self._name]
        # store module level parameters, subclass of `hglconfig.ModuleConf`
        self._conf: Type[hglconfig.ModuleConf] = None
        # lazy-callables for child modules, ordered
        self._subconfigs: Dict[hglconfig.HGLConf, None] = {}
        # module_sv necessary for generating verilog 
        self._module = module_sv.ModuleSV(self) 

        # default module level parameters
        # dispatcher
        self.dispatcher: Dispatcher = None 
        self.clock: Tuple[hgl_core.Reader, int] = None 
        self.reset: Tuple[hgl_core.Reader, int] = None 
        # io 
        self.io: Array = None 

        # temp stack
        self._prev: List[Module] = []
        self._temp_inputs: List[hgl_core.Reader] = []

    def __conf__(self): 
        up = self._sess.module
        up_conf = up._conf
        # default parameters
        self.dispatcher = up.dispatcher
        self.clock = up.clock 
        self.reset = up.reset
            
        # record matched callables
        matched_configs: List[hglconfig.HGLConf] = []
        inherit: bool = True
        for config in up._subconfigs: 
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
            paras = {}
            
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
        '''your hdl module defination'''


    def __tail__(self, f_locals:dict):  
        # delete input driver
        for i in self._temp_inputs:
            i._data._module = None

        # record 
        io = None
        for k, v in f_locals.items():
            if k == 'io':
                io = v 
            elif k[0] != '_': 
                if k not in self.__dict__:
                    self.__dict__[k] = v 
                if isinstance(v, hgl_core.Reader):
                    # update prefered name
                    v._data._name = k
                elif isinstance(v, Module) and v in self._submodules:
                    # update prefered name
                    v._name = k 
                elif isinstance(v, Array):
                    _rename_array(v, k)

        if io is not None:
            self.io = io 
            def make_io(x):
                assert isinstance(x, hgl_core.Reader)
                if x._direction == 'input':
                    self._sess.module._module.inputs[x] = None
                elif x._direction in ['inner','output']:
                    x._direction = 'output' 
                    self._sess.module._module.outputs[x] = None 
                    if x._data.writer is None and x._data._module is None:
                        x._data._module = self
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
        """ support array-like slicing
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
            raise AttributeError(f'{k} is neither Module member nor Config parameter for id:{self._ids}')


    def __str__(self) -> str:
        """ return full name, ex. Global.Top.Adder_1
        """
        return '.'.join(i._unique_name for i in self._position)
 


def _rename_array(obj: Union[Array, hgl_core.Reader], name: str):
    if isinstance(obj, hgl_core.Reader):
        obj._data._name = name
    elif isinstance(obj, Array):
        for k, v in obj._items():
            new_name =  f"{name}_{k}"
            _rename_array(v, new_name)
    else:
        return 
    