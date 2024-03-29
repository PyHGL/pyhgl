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
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, Type

import sys
import inspect
import re

from pyhgl.array import * 
import pyhgl.logic._session as _session
import pyhgl.logic.hgl_core as hgl_core
import pyhgl.logic.utils as utils



"""
construct a tree-like config as global config. 

@conf top:
    conf.dispatcher = ...
    conf.clock = ... 
    conf.reset = conf.up.reset
    conf.timing = Bundle(
        timescale='1ns',
        Wire = {'delay': 1}
    )
    
    width = 10 
    @conf.always('adder.*') Adder1:
        width = conf.up.width * 2 
    @conf('adder.*').clean Adder2:
        width = width * 3
"""

        
class _Conf(HGL):
    """ decorator function, return HGLConf
    """
    _sess: _session.Session

    def __init__(self, always = False, clear = False, filter: str = None):
        self._always = always 
        self._clear = clear  
        self._filter = filter

    @property 
    def always(self):
        return _Conf(always=True, clear=self._clear, filter=self._filter)

    @property 
    def clear(self):
        return _Conf(always=self._always, clear=True, filter=self._filter)
    
    def __call__(self, obj: Union[Callable, str]): 
        # set filter
        if isinstance(obj, str):
            return _Conf(always=self._always, clear=self._clear, filter=obj) 
        # record config
        else:
            assert callable(obj)
            filter = self._filter or obj.__name__ + '.*'
            return HGLConf(
                obj, 
                filter = filter, 
                always = self._always,
                inherit = not self._clear,
            ) 
        
    @property
    def up(self): 
        """ return father module's config
        """ 
        try: 
            return self._sess.module._position[-2]._conf
        except:
            return None
        
    @property 
    def p(self):
        """ return current module's config
        """
        return self._sess.module._conf 
    
    @property 
    def m(self):
        return self._sess.module 
    
    @property 
    def t(self):
        return self._sess.sim_py.t

    @property 
    def clock(self):  
        return self._sess.module._conf.clock 

    @clock.setter
    def clock(self, v): 
        signal, edge = v 
        assert isinstance(signal, hgl_core.Reader) and len(signal) == 1 
        assert edge in [0,1]
        self._sess.module._conf.clock = v 
    
    @property 
    def reset(self): 
        return self._sess.module._conf.reset

    @reset.setter
    def reset(self, v):
        signal, edge = v 
        assert isinstance(signal, hgl_core.Reader) and len(signal) == 1 
        assert edge in [0,1]
        self._sess.module._conf.reset = v 

    @property 
    def dispatcher(self):
        return self._sess.module._conf.dispatcher 

    @dispatcher.setter
    def dispatcher(self, v): 
        assert isinstance(v, Dispatcher)
        self._sess.module._conf.dispatcher = v 

    @property 
    def timing(self):
        return self._sess.timing._values # one timing config per session

    @timing.setter
    def timing(self, v: Union[dict, Array]):
        self._sess.timing.update(v) 
        
conf = _Conf()
    
    
class HGLConf(HGL):
    
    def __init__(
        self, 
        f: Callable,  # a function return locals
        filter: Union[str, re.Pattern], 
        args = (),
        kwargs = {},
        always = False, 
        inherit = True
    ) -> None:

        self.f = f 
        if isinstance(filter, str):
            self.filter = re.compile(filter, flags=re.IGNORECASE) 
        else:
            self.filter = filter 
        self.args = args
        self.kwargs = kwargs
        self.always = always        # config is valid for all submodules
        self.inherit = inherit      # inherit father's parameters 
        # self._arg_names = set(inspect.signature(f).parameters.keys())
    
    
    def __call__(self, *args, **kwargs):
        return HGLConf(
            f=self.f, 
            filter=self.filter,
            args=args,
            kwargs=kwargs,
            always=self.always,
            inherit=self.inherit,
        )
    
    
    def exec(self) -> Tuple[dict, Dict[HGLConf, None]]:
        """ return (parameters, subconfigs)
        """
        # execute config function
        local_var: Dict[str, Any] = self.f(*self.args, **self.kwargs)
            
        parameters = {}
        subconfigs = {}
        
        for k, v in local_var.items(): 
            if isinstance(v, HGLConf):
                subconfigs[v] = None
            elif k[0] != '_':
                parameters[k] = v   # parameters whose name not start with '_'
                
        return parameters, subconfigs
    
    def __str__(self):
        func = f"{self.f.__module__}.{self.f.__qualname__}"
        func = re.sub(r'.<locals>','', func) 
        filter = f"'{self.filter.pattern}'"
        
        ret = [func, filter]

        if self.always:
            ret.append('always')
        if self.inherit:
            ret.append('inherit') 
            
        return f"HGLConf({','.join(ret)})"
    
        

class ModuleConf: 
    """ parameters stored as class variables
    """
    clock: Tuple[hgl_core.Reader, int]
    reset: Tuple[hgl_core.Reader, int]
    dispatcher: Dispatcher

# ----------- 
# timing 
# -----------


class TimingConf(HGL):
    """        
    {
        'timescale': '1ns',
        'Wire': {
            'delay': 1,
            'power': 0
        },
        'Gate': {
            'delay': 1,
            'power':0
        },
        'Clock': {
            'period': 100
        }
    }
    """
    def __init__(self):
        self.timescale: float = None    # 1e-9s
        self.time_info: Tuple[int, str] = None  # default: (1, 'ns')
        self._values: Dict[str, dict] = {}

    def update(self, v: Union[dict, Array]):
        if isinstance(v, Array):
            v = v._dict 
        if timescale:=v.get('timescale'):
            assert self.timescale is None, 'timescale already exists'
            self.timescale, temp = utils.quantity(timescale) 
            self.time_info = (round(temp[0]), temp[1]) 
        else:
            if self.timescale is None:
                self.timescale: float = 1e-9
                self.time_info: Tuple[int, str] = (1, 'ns') 
        _v = self._get_quantity(v)  
        self._values.update(_v)

    def get(self, key: str):
        return self._values.get(key)
    
    def _get_timestep(self, v: str) -> int:
        _v, _ = utils.quantity(v)
        return round(_v / self.timescale)
    
    def _get_quantity(self, v: Dict[str, Union[int, float, str]]):
        """ 
        convert quantities with unit to desired value 
        
        time -> int(time/timescale)
        """
        ret = {}
        if isinstance(v, Array):
            items = v._items()
        else:
            items = v.items() 
            
        for key, value in items:
            if isinstance(value, (dict, Array)):
                ret[key] = self._get_quantity(value)
            elif isinstance(value, (int, float)):
                ret[key] = value
            elif isinstance(value, str):
                _v, _ = utils.quantity(value)
                ret[key] = round(_v/self.timescale)
        return ret
                
    def __bool__(self):
        return True

    def __str__(self):
        return str(self._values)
    
    
    

