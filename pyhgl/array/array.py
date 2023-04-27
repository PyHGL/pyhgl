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
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union
import inspect 
import re
import numpy as np 

from ._hgl import HGL


class SignalKey(HGL):
    """ a kind of slice key means slicing signal.
    """
    __slots__ = ()



class Container(HGL):

    __slots__ = ()

    def _format_slice(self, key: Union[int, str, list]) -> Tuple[Union[list, dict], Union[int, str, list, dict]]:
        """ return (data, keys)
        """
        raise NotImplementedError()

 
    def __getitem__(self, keys):
        """ 
        x[1,2,:,-1,'a']
        x[[1,2],:]   # list means select
        """
        # deal with multiple slice
        if isinstance(keys, tuple):       
            key = keys[0]
            rest_keys = keys[1:]
        else:
            key = keys
            rest_keys = tuple()  
        # if isinstance(key, SignalKey):
        #     rest_keys = (key, *rest_keys)
        #     key = ... 
        rest_keys_valid = rest_keys[0] if len(rest_keys) == 1 else rest_keys

        target, idxes = self._format_slice(key)  

        # ... means get all   
        rest_keys_container = (..., *rest_keys)
        def _get_all(x):
            if isinstance(x, Container):
                return x.__getitem__(rest_keys_container) 
            else:
                if rest_keys:
                    return x.__getitem__(rest_keys_valid) 
                else:
                    return x  
        if key is ...:
            if isinstance(idxes, list): 
                return Array([_get_all(target[i]) for i in idxes])
            else:
                return Array({k:_get_all(target[i]) for k,i in idxes.items()})

        # single item
        if isinstance(idxes, (int, str)):
            if rest_keys:
                return target[idxes].__getitem__(rest_keys_valid)
            else:
                return target[idxes] 
        # multiple items, return a subset (Array)
        else:
            if rest_keys:  
                if isinstance(idxes, list):
                    return Array([target[i].__getitem__(rest_keys_valid) for i in idxes])
                else:
                    return Array({k:target[i].__getitem__(rest_keys_valid) for k,i in idxes.items()})
            else:
                if isinstance(idxes, list):
                    return Array([target[i] for i in idxes])
                else:
                    return Array({k:target[i] for k,i in idxes.items()})
        

    def __setitem__(self, keys, value):
        """
        a['port_x'] = UInt()
        a[0] = UInt()
        
        broadcast input value according to keys 
        a[:] = 0
        a[3:6] = [1,2,3]
        """ 
        # deal with multiple slice
        if isinstance(keys, tuple):       
            key = keys[0]
            rest_keys = keys[1:]
        else:
            key = keys
            rest_keys = tuple()

        target, idxes = self._format_slice(key) 

        rest_keys_valid = rest_keys[0] if len(rest_keys) == 1 else rest_keys
        rest_keys_container = (..., *rest_keys)

        # ... means set all, always broadcast
        def _set_all(target, i, v): 
            if key is ...:
                x = target[i]
                if isinstance(x, Container):
                    x.__setitem__(rest_keys_container, v)  
                else: 
                    if rest_keys:
                        x.__setitem__(rest_keys_valid, v) 
                    else:
                        target[i] = v
            else:
                if rest_keys:
                    target[i].__setitem__(rest_keys_valid, v)
                else:
                    target[i] = v 


        # single assign
        if isinstance(idxes, (int, str)):
            if rest_keys:
                target[idxes].__setitem__(rest_keys_valid, value)
            else:
                target[idxes] = value
        # multi assign
        else:
            if isinstance(idxes, dict):
                idxes = list(idxes.values())
            # broadcast single value
            if not isinstance(value, Array): 
                for i in idxes:
                    _set_all(target, i, value)
            # broadcast dim width == 1
            elif isinstance(value, Array) and len(value) == 1: 
                v = value[0]
                for i in idxes:
                    _set_all(target, i, v)
            # one-to-one map
            else:
                assert len(idxes) == len(value), f'cannot assign len {len(value)} to len {len(idxes)}'
                for i, v in zip(idxes, value):
                    _set_all(target, i, v)  


    def __partial_assign__(self, a, b, value, keys):
        """ similar as __setitem__, but for hardware connection 

        if keys[0] is SignalKey, insert a ...; if no key rest, insert a SignalKey
        """   
        # deal with multiple slice
        if isinstance(keys, tuple):       
            key = keys[0]
            rest_keys = keys[1:]
        else:
            key = keys
            rest_keys = tuple()  
        if isinstance(key, SignalKey):
            rest_keys = (key, *rest_keys)
            key = ... 

        # must have rest keys
        if not rest_keys:
            rest_keys = (SignalKey(),)

        rest_keys_valid = rest_keys[0] if len(rest_keys) == 1 else rest_keys 
        rest_keys_container = (..., *rest_keys)

        target, idxes = self._format_slice(key) 

        # ... means set all, always broadcast
        def _set_all(x, v): 
            if key is ...: 
                if isinstance(x, Container):
                    x.__partial_assign__(a, b, v, rest_keys_container) 
                else:
                    x.__partial_assign__(a, b, v, rest_keys_valid)
            else:
                x.__partial_assign__(a, b, v, rest_keys_valid)


        # single assign
        if isinstance(idxes, (int, str)):
            _set_all(target[idxes], value)
        # multi assign
        else:
            if isinstance(idxes, dict):
                idxes = list(idxes.values())
            # broadcast single value
            if not isinstance(value, Array):
                for i in idxes: 
                    _set_all(target[i], value)
            # broadcast dim width == 1
            elif isinstance(value, Array) and len(value) == 1:
                v = value[0]
                for i in idxes:
                    _set_all(target[i], v)
            # one-to-one map
            else:
                assert len(idxes) == len(value), f'cannot broadcast len {len(value)} to len {len(idxes)}' 
                for i, v in zip(idxes, value): 
                    _set_all(target[i], v)
        




_valid_name = re.compile(r'[a-zA-Z]\w*')

class Array(Container):
    """ N-dimensional tree-like Array, with/without name, variable shape
        
    - keep insert order
    - store reference
    - always return new array in slicing
    - return odd array in __getattribute__
    """ 

    __slots__ = '_list', '_dict'

    def __init__(self, array_like: Iterable = [], *, recursive: bool = False, atoms: List[type] = [dict, str]):
        """
        array-like:
            list, dict, tuple, Array 

        recursive:
            False: 1-d Array 
            True: n-d Array 

        atoms:
            non-array: int, str, float, bool, Logic, ...  
        """
        # one is empty
        self._list: List[Any] = []  # vector
        self._dict: Dict[Any] = {}  # named array

        if isinstance(array_like, dict):
            for k, v in array_like.items(): 
                if not _valid_name.match(k):
                    raise Exception(f"invalid name: {k} should not start with _")  
                if recursive and hasattr(v, '__iter__') and type(v) not in atoms:
                    self._dict[k] = Array(v, recursive=True, atoms=atoms) 
                else:
                    self._dict[k] = v
        else:
            for v in array_like: 
                if recursive and hasattr(v, '__iter__') and type(v) not in atoms:
                    self._list.append(Array(v, recursive=True, atoms=atoms))
                else:
                    self._list.append(v)

    def __call__(self, *args, **kwargs):
        return Map(lambda f: f(*args, **kwargs), self)

    #-----------------------------------
    # unary operators & binary operators
    #-----------------------------------
    # bitwise
    def __invert__(self):           return Not(self)                # ~a      
    def __or__(self, other):        return Or(self, other)          # a | b
    def __and__(self, other):       return And(self, other)         # a & b
    def __xor__(self, other):       return Xor(self, other)         # a ^ b
    def __lshift__(self, other):    return Lshift(self, other)      # a << b
    def __rshift__(self, other):    return Rshift(self, other)      # a >> b
    def __pow__(self, other):       return Pow(self, other)         # a ** b
    # comparison, return 1-bit signal 
    def __lt__(self, other):        return Lt(self, other)          # a < b
    def __gt__(self, other):        return Gt(self, other)          # a > b
    def __le__(self, other):        return Le(self, other)          # a <= b
    def __ge__(self, other):        return Ge(self, other)          # a >= b
    def __eq__(self, other):        return Eq(self, other)          # a == b
    def __ne__(self, other):        return Ne(self, other)          # a != b
    # calculate 
    def __pos__(self):              return Pos(self)                # +a    
    def __neg__(self):              return Neg(self)                # -a 
    def __add__(self, other):       return Add(self, other)         # a + b
    def __sub__(self, other):       return Sub(self, other)         # a - b
    def __mul__(self, other):       return Mul(self, other)         # a * b
    def __matmul__(self, other):    return Matmul(self, other)      # a @ b
    def __truediv__(self, other):   return Div(self, other)         # a / b
    def __mod__(self, other):       return Mod(self, other)         # a % b
    def __floordiv__(self, other):  return Floordiv(self, other)    # a // b
    #----------- 
    # assignment    
    #----------- 
    def __ilshift__(self, other):   raise NotImplementedError()     # <<=
    def __iadd__(self, other):      raise NotImplementedError()     # +=
    def __isub__(self, other):      raise NotImplementedError()     # -=
    def __imul__(self, other):      raise NotImplementedError()     # *=  
    def __imatmul__(self, other):   raise NotImplementedError()     # @=           
    def __ifloordiv__(self, other): raise NotImplementedError()     # //=
    def __itruediv__(self, other):  raise NotImplementedError()     # /=
    def __imod__(self, other):      raise NotImplementedError()     # %= 
    def __ipow__(self,other):       raise NotImplementedError()     # **=
    def __irshift__(self, other):   raise NotImplementedError()     # >>=
    def __iand__(self, other):      raise NotImplementedError()     # &=
    def __ior__(self, other):       raise NotImplementedError()     # |=
    def __ixor__(self, other):      raise NotImplementedError()     # ^=

    def __hash__(self):             return id(self)

    def __contains__(self, v) -> bool:
        """ value in array
        """
        if self._dict:
            return v in self._dict 
        else:
            return v in self._list
        
    def __bool__(self) -> bool:
        if self._dict or self._list:
            return True 
        else:
            return False

    def __len__(self) -> int: 
        if self._dict:
            return len(self._dict)
        else:
            return len(self._list)

    def __str__(self):
        if self._dict:
            name = 'Bundle'
            body = ''.join([f'{k}: {v}\n' for k, v in self._dict.items()]) 
            body = '  ' + '  '.join(body.splitlines(keepends=True))
        elif self._list:
            name = 'Vec'
            body = ''.join([f'{i}\n' for i in self._list])            
            body = '  ' + '  '.join(body.splitlines(keepends=True)) 
        else:
            return 'Vec{}'
        return f'{name}{{\n{body}}}' 
            
    def __iter__(self): 
        """ return values, not keys
        """
        if self._dict:
            return iter(self._dict.values())
        else:
            return iter(self._list)

    
    def __getattribute__(self, name:str)->Any:
        """ 
        ex. array.a, array.b, array.a.x, ...

        notice: return reference
        """
        if name[0] == '_':
            return object.__getattribute__(self, name) 
        else: 
            return self._dict[name]

    def __setattr__(self, name: str, value: Any):
        if name[0] == '_':
            object.__setattr__(self, name, value)
        else:  
            if not _valid_name.match(name):
                raise Exception(f'invalid attribute name {name}') 
            if self._list:
                raise Exception('__setattr__ only valid for named array') 
            self._dict[name] = value
    
    def _insert(self, idx:int, value) -> None: 
        """ only for vector
        """ 
        if self._dict:
            raise Exception('cannot insert item to named array')
        self._list.insert(idx, value) 
    
    def _append(self, value) -> None: 
        """ (k, v) for named array, v for vector
        """
        if self._dict:
            k, v = value 
            assert _valid_name.match(k), f'invalid key {k}' 
            assert k not in self._dict, f'duplicated key {k}'
            self._dict[k] = v  
        else:
            self._list.append(value)
        
    def _extend(self, value: Iterable) -> None: 
        """ only for vector
        """ 
        if self._dict:
            raise Exception('cannot extend named array') 
        self._list.extend(value)
        
    def _pop(self, *key: Union[int, str]) -> Any: 
        if self._dict:
            return self._dict.pop(key[0]) 
        else: 
            return self._list.pop(*key)

    def _keys(self) -> Iterable:
        return self._dict.keys()
    
    def _values(self) -> Iterable:
        return self._dict._values()
    
    def _items(self) -> Iterable:
        return self._dict.items()

    def _get(self, key):
        return self._dict.get(key)   
        
    def _update(self, map: Union[dict, Array]):
        """ only for named array
        """ 
        if self._list:
            raise AttributeError('cannot update vector')
        if isinstance(map, Array):
            for k, v in map._items():
                assert _valid_name.match(k), f'invalid key {k}'
                self._dict[k] = v
        else:
            for k, v in map.items():
                assert _valid_name.match(k), f'invalid key {k}'
                self._dict[k] = v
                

    def _format_slice(self, key: Union[int, str, list]) -> Tuple[Union[list, dict], Union[int, str, list, dict]]:
        """ 
        array['port_a'] -> array._dict, 'port_a' 
        array[:2] -> array._list, [0,1]
        array[::-1] -> array._list, [-1,-2,...]
        
        key:
            None: insert a new dim
            ...: rest dim
            int/slice/str: single slice or multi slice 
            list: multi slice 
            dict: multi slice

        return: 
            (target, idx|idxes) 
            idxes: list|dict 
                list: list of int|str 
                dict: dict of str:int|str
        """
        # single item, ex. array['port_a']
        if isinstance(key, str):
            return self._dict, key
        # single item, ex. array[-1]
        elif hasattr(key, '__int__'):
            key = int(key) 
            if self._dict:
                return self._dict, list(self._dict.keys())[key]
            else:
                return self._list, key 
        # new axis 
        elif key is None:
            return [self], [0]
        # multi items, ex. array[1:], array[...]
        elif key is ...:
            if self._dict:
                return self._dict, {k:k for k in list(self._dict.keys())} 
            else:
                return self._list, list(range(len(self._list)))
        elif isinstance(key, slice):
            if self._dict: 
                return self._dict, {k:k for k in list(self._dict.keys())[key]}
            else:
                return self._list, list(range(len(self._list)))[key]
        # multi items, ex. array[[1,2,3]], array[['a','b', 'c']]
        elif isinstance(key, list): 
            if self._dict:
                idxes = {}
                for i in key:
                    idx = self._format_slice(i)[-1] 
                    assert isinstance(idx, str)  
                    idxes[idx] = idx
                return self._dict, idxes
            else:
                idxes = [] 
                for i in key:
                    idx = self._format_slice(i)[-1]
                    assert isinstance(idx, int) 
                    idxes.append(idx) 
                return self._list, idxes 
        # ex. array[{'a':0, 'b':1}]
        elif isinstance(key, dict):
            idxes = {} 
            if self._dict: 
                for name, i in key.items():
                    idx = self._format_slice(i)[-1]
                    assert isinstance(idx, str) 
                    idxes[name] = i 
                return self._dict, idxes 
            else:
                for name, i in key.items():
                    idx = self._format_slice(i)[-1]
                    assert isinstance(idx, int) 
                    idxes[name] = i 
                return self._list, idxes 
        else:
            raise KeyError(key) 


    @property
    def _flat(self) -> Array:
        """ dfs flat
        """
        ret = Array()
        if self._dict:
            items = self._dict.values()
        else:
            items = self._list 

        for i in items:
            if isinstance(i, Array):
                ret._extend(i._flat)
            else:
                ret._append(i)
        return ret

    def _to_dict(self) -> Union[list, dict]:
        if self._dict: 
            ret = {}
            for k, v in self._dict.items():
                if isinstance(v, Array):
                    ret[k] = v._to_dict() 
                else:
                    ret[k] = v 
            return ret
        else:
            ret = []
            for i in self._list:
                if isinstance(i, Array):
                    ret.append(i._to_dict())
                else:
                    ret.append(i)
            return ret


    @property
    def _shape(self) -> Optional[tuple]:
        """ shape of vector. for named array, return None 
        """
        if not self._list and not self._dict:
            return (0,)
        else:
            if self._dict:
                target = self._dict 
            else:
                target = self._list 
            # 1-d array
            if all([not isinstance(i, Array) for i in target]):
                return (len(self),) 
            # n-d array
            else:
                shapes = [] 
                for i in target:
                    if not isinstance(i, Array) or (s:=i._shape) is None:
                        return None 
                    shapes.append(s)
                for i in shapes: 
                    if i != shapes[0]:
                        return None 
                return (len(shapes), *shapes[0])
            
    @property 
    def _ndim(self) -> Optional[int]:
        if (shape:=self._shape) is not None:
            return len(shape)
        else:
            return None 
        

    @property 
    def _size(self) -> int: 
        return len(self._flat)


    def _reshape(self, *args: Union[Tuple[int], int]) -> Array: 
        """ return a new array 
        """
        if len(args) == 1 and isinstance(args[0], tuple):
            args = args[0] 
            
        new_shape = list(args)
        flat = self._flat 
        n_total = len(flat)
        assert n_total > 0, 'cannot reshape empty array'
        
        n_new = 1 
        idx_uncertain = None
        for i,n in enumerate(new_shape):
            assert isinstance(n, int)
            if n > 0:
                n_new *= n
            elif n == 0:
                raise ValueError(f'cannot reshape {self._shape} into {new_shape}') 
            else: 
                assert idx_uncertain is None, 'can only specify one unknown dimension'
                idx_uncertain = i 
        
        if idx_uncertain is None:
            assert n_new == n_total, f'cannot reshape {self._shape} into {new_shape}'
        else:
            assert n_total % n_new == 0, f'cannot reshape {self._shape} into {new_shape}' 
            new_shape[idx_uncertain] = n_total // n_new 
        
        ret = Array.full(tuple(new_shape), value = 0)
        it = iter(flat)  
        return Map(lambda x: next(it), ret)
    

    def _fill(self, v: Iterable) -> Array: 
        """ turn a 1-d iterable into self shape
        """
        v = iter(v)
        return Map(lambda _: next(v), self)
    
    def _full(self, v: Any):
        """ return Array with same shape, filled with v
        """ 
        return Map(lambda _: v, self) 
    
    def full(shape: Union[Tuple[int], Array, int], value: Any) -> Array:
        """ generate a n-d array

        Args:
            value : value to fill

        Examples:
            Array.full((2,3,4), 'a')
            Array.full(2,3,4, value='b')
        """ 
        if isinstance(shape, int):
            shape = (shape,)
        else:
            shape = tuple(shape)
        return Map(lambda _: value, Array(np.zeros(shape), recursive=True))

    def ones(*shape) -> Array:
        if isinstance(shape[0], int):  # zeros(2,3,4)
            shape = shape 
        else:                           # zeros((2,3,4))
            assert len(shape) == 1 
            shape = shape[0]
        return Map(lambda _: 1, Array(np.zeros(shape), recursive=True))
        
    def zeros(*shape) -> Array:
        if isinstance(shape[0], int):  # zeros(2,3,4)
            shape = shape 
        else:                           # zeros((2,3,4))
            assert len(shape) == 1 
            shape = shape[0]
        return Map(lambda _: 0, Array(np.zeros(shape), recursive=True))
    

def ToArray(x) -> Any: 
    """ turn iterable (except dict & str) into array, recursively
    """
    if isinstance(x, (dict, str, Array)) or not hasattr(x, '__iter__'):
        return x
    else:
        return Array(x, recursive=True)


def Bundle(*args, **kwargs) -> Array:
    """ Bundle(x = 1, y = 2)
        Bundle(1,2,3,4)
    """
    if args:
        assert not kwargs 
        return Array(args)
    else:
        return Array(kwargs) 

def bundle(f: Callable) -> Array:
    return Array(f())



    


def Map(f: Callable, /, *args, **kwargs): 
    """ map function on arrays

    Args:
        f: 
            Callable takes atoms as input 
        args: 
            positional arguments of function, lenth >= 1
            if Array, expand 
            if atom or len=1, broadcast 
        kwargs: 
            optional, not expand, but broadcast to all

    notice: other iterable regard as atom
    """
    arrays = [arg for arg in args if isinstance(arg, Array)]
    if not arrays:
        return f(*args, **kwargs)

    # check length 
    length = [len(arg) for arg in arrays]  # 1,0,1,3,3,1
    max_len = max(length)
    for i in length:
        if i == 0:
            return Array([])  # map function on empty array return empty array
        if i != 1 and i != max_len:
            raise Exception(f'cannot broadcast {i} to {max_len}')  

    keys = []                   # if named-array, use keys of first
    for arg in arrays:          # 1,1,3,3,1
        if len(arg) == max_len and arg._dict and not keys:
            keys = arg._dict.keys() 
 
    elements = []
    for i in range(max_len):
        arg_list = []
        for arg in args: 
            if isinstance(arg, Array): 
                if len(arg) == 1:
                    arg_list.append(arg[0])
                else:
                    arg_list.append(arg[i])
            else:
                arg_list.append(arg)
        elements.append(Map(f, *arg_list, **kwargs)) 

    if keys:
        return Array({k:v for k,v in zip(keys, elements)})
    else:
        return Array(elements) 



def MapFirst(f: Callable, /, *args: Array, **kwargs): 
    """ broadcast to first shape 

    ex. MapFirst(f, [1,2], [[3,4], [5,6]])  -> f(1, [3,4])
    """  
    assert args
    if not isinstance(args[0], Array):
        return f(*args, **kwargs) 
    
    first = args[0]
    if first._dict:
        keys = first._dict.keys()
    else:
        keys = [] 
    
    elements = []
    for i in range(len(first)):
        arg_list = []
        for arg in args:
            if isinstance(arg, Array):
                if len(arg) == 1:
                    arg_list.append(arg[0])
                else: 
                    arg_list.append(arg[i])
            else:
                arg_list.append(arg)
        elements.append(MapFirst(f, *arg_list, **kwargs))
    
    if keys:
        return Array({k:v for k,v in zip(keys, elements)})
    else:
        return Array(elements)



#-------------------------------------------
# decorators
#-------------------------------------------

def vectorize(f):
    """ apply f on positional array-like arguments
    """
    assert callable(f) 
    def vectorized(*args, **kwargs):
        # turn array-like into array
        args = [ToArray(i) for i in args]
        return Map(f, *args, **kwargs)
    vectorized.__hgl_wrapped__ = f
    return vectorized  

def vectorize_atom(f):
    """ will not turn iterable like list into Array
    """
    assert callable(f) 
    def vectorize_atom(*args, **kwargs):
        return Map(f, *args, **kwargs)
    vectorize_atom.__hgl_wrapped__ = f
    return vectorize_atom   

def vectorize_first(f):
    assert callable(f) 
    def vectorize_first(*args, **kwargs): 
        args = [ToArray(i) for i in args]
        return MapFirst(f, *args, **kwargs)
    vectorize_first.__hgl_wrapped__ = f
    return vectorize_first   


"""
default:
    Add(a,b,c, axis=0) === Add([a,b,c], axis=0) === a+b+c 
axis=1:
    Add(a,b,c, axis=1) === Add([a,b,c], axis=1) ===[sum(a), sum(b), sum(c)]
"""

def vectorize_axis(f):
    """ 
    f: 
        Callable takes 1 or more atoms as positional args 
    return :
        function take one array_like as positional input and apply f on all axis
        if multiple positional args, map f on args
    keyword argument 'axis':
        `...` or `None` means all axis 
        int, list[int] means selected axes
    """
    assert callable(f)
    
    def vectorized_axis(*args, **kwargs) -> Any: 
        assert args, 'no args'
        # regard multiple positional args as array
        args = args[0] if len(args) == 1 else args
        # turn array-like into array
        args = ToArray(args) 
        # get axis
        if 'axis' in kwargs:
            axis = kwargs.pop('axis')
        else:
            axis = 0    # axis is 0 by default 
        if axis is ...:
            axis = None # None means all axes 

        # single atom
        if not isinstance(args, Array):   
            return f(args, **kwargs) 
        # reduce all axes 
        if axis is None:    
            return f(*args._flat, **kwargs) 
        # sum columns
        elif axis == 0:
            return Map(f, *args, **kwargs)
        # sum rows
        elif axis == 1:
            # keep axis 0
            ret = []
            for i in args:
                # reduce axis 1
                if not isinstance(i, Array):
                    ret.append(f(i, **kwargs))
                else:
                    ret.append(Map(f, *i, **kwargs))
            return Array(ret) 
        # multiple axis or > 1
        else:
            value = args 
            shape = value._shape 
            assert shape is not None and 0 not in shape, 'not array-like'  
            if isinstance(axis, int): 
                axis = (axis,) 
            # axis
            keeped_axis: List[int] = []
            removed_axis: List[int] = []
            for i in axis:
                if i < 0:
                    i += len(shape)
                assert 0 <= i < len(shape), 'axis out of range'
                removed_axis.append(i)    
            for i in range(len(shape)):
                if i not in removed_axis:
                    keeped_axis.append(i)
            removed_axis.sort()
            
            # reduce all axis
            if not keeped_axis:
                return f(*value._flat, **kwargs)
            
            ret = []
            ret_shape = tuple(shape[i] for i in keeped_axis)
            for idxes in np.ndindex(ret_shape):
                _slice = list(shape)
                for axis, idx in zip(keeped_axis, idxes):
                    _slice[axis] = idx 
                for i in removed_axis:
                    _slice[i] = slice(None) 
                ret.append(f( *value.__getitem__(tuple(_slice))._flat, **kwargs))
            return Array(ret)._reshape(ret_shape)


        # if len(args) > 1: 
        #     if 'axis' in kwargs:
        #         raise TypeError('"axis" is not valid kwarg when more than one positional args')
        #     return Map(f, *args, **kwargs)
        # elif len(args) == 1:
        #     value = args[0]
        #     # single atom
        #     if not isinstance(value, Array):
        #         return f(value, **kwargs)
        #     # get axis
        #     if 'axis' not in kwargs:
        #         return Map(f, value, **kwargs)
        #     else:
        #         axis = kwargs.pop('axis') 
        #     if axis is ...:
        #         axis = None
        #     # default apply on all and return single value
        #     if axis is None:
        #         return f(*value._flat, **kwargs)
            
        #     shape = value._shape 
        #     assert shape is not None and 0 not in shape, 'not array-like'  
        #     if isinstance(axis, int): 
        #         axis = (axis,) 
        #     # axis
        #     keeped_axis: List[int] = []
        #     removed_axis: List[int] = []
        #     for i in axis:
        #         if i < 0:
        #             i += len(shape)
        #         assert 0 <= i < len(shape), 'axis out of range'
        #         removed_axis.append(i)    
        #     for i in range(len(shape)):
        #         if i not in removed_axis:
        #             keeped_axis.append(i)
        #     removed_axis.sort()
            
        #     # reduce all axis
        #     if not keeped_axis:
        #         return f(*value._flat, **kwargs)
            
        #     ret = []
        #     ret_shape = tuple(shape[i] for i in keeped_axis)
        #     for idxes in np.ndindex(ret_shape):
        #         _slice = list(shape)
        #         for axis, idx in zip(keeped_axis, idxes):
        #             _slice[axis] = idx 
        #         for i in removed_axis:
        #             _slice[i] = slice(None) 
        #         ret.append(f( *value.__getitem__(tuple(_slice))._flat, **kwargs))
        #     return Array(ret)._reshape(ret_shape)
        # else:
        #     raise ValueError(f'{f} takes at least one positional arg')

    vectorized_axis.__hgl_wrapped__ = f 
    return vectorized_axis





#-------------------------------------------
# functions
#-------------------------------------------

@vectorize 
def Signal(x):
    """ convert python internal type to Signal, may not generate new gate
    
    iterable except str -> Array
    int, str, gmpy2.mpz -> UInt
    """
    return HGL._sess.module._conf.dispatcher.call('Signal', x)

#--------
# bitwise
#--------

# ~a
@vectorize 
def Not(a, **kwargs):
    """ a: Signal or Immd
    """
    return HGL._sess.module._conf.dispatcher.call('Not', Signal(a), **kwargs)

# a | b
@vectorize_axis
def Or(*args, **kwargs):
    """ args: Signal or Immd
    """
    args = (Signal(i) for i in args)
    return HGL._sess.module._conf.dispatcher.call('Or', *args, **kwargs) 

# a & b
@vectorize_axis 
def And(*args, **kwargs):
    """ args: Signal or Immd
    """
    args = (Signal(i) for i in args)
    return HGL._sess.module._conf.dispatcher.call('And', *args, **kwargs) 

# a ^ b 
@vectorize_axis 
def Xor(*args, **kwargs):
    """ args: Signal or Immd
    """
    args = (Signal(i) for i in args)
    return HGL._sess.module._conf.dispatcher.call('Xor', *args, **kwargs) 


#------- 
# shift
#------- 
# a << b 
@vectorize
def Lshift(a, b, **kwargs):
    """ a: Signal, b: Signal or Immd
    """
    return HGL._sess.module._conf.dispatcher.call('Lshift', Signal(a), b, **kwargs) 

# a >> b
@vectorize
def Rshift(a, b, **kwargs):
    """ a: Signal, b: Signal or Immd
    """
    return HGL._sess.module._conf.dispatcher.call('Rshift', Signal(a), b, **kwargs) 

#--------
# compare
#--------
# a == b
@vectorize
def Eq(a, b, **kwargs):
    """ a: Signal, b: Signal or Immd or Any
    """
    return HGL._sess.module._conf.dispatcher.call('Eq', Signal(a), b, **kwargs) 

# a != b
@vectorize
def Ne(a, b, **kwargs):
    """ a: Signal, b: Signal or Immd or Any
    """
    return HGL._sess.module._conf.dispatcher.call('Ne', Signal(a), b, **kwargs) 


# a < b
@vectorize
def Lt(a, b, **kwargs):
    """ a: Signal, b: Signal or Immd 
    """
    return HGL._sess.module._conf.dispatcher.call('Lt', Signal(a), b, **kwargs) 

# a <= b
@vectorize
def Le(a, b, **kwargs):
    """ a: Signal, b: Signal or Immd 
    """
    return HGL._sess.module._conf.dispatcher.call('Le', Signal(a), b, **kwargs) 

# a > b
@vectorize
def Gt(a, b, **kwargs):
    """ a: Signal, b: Signal or Immd 
    """
    return HGL._sess.module._conf.dispatcher.call('Gt', Signal(a), b, **kwargs) 

# a >= b
@vectorize
def Ge(a, b, **kwargs):
    """ a: Signal, b: Signal or Immd 
    """
    return HGL._sess.module._conf.dispatcher.call('Ge', Signal(a), b, **kwargs) 



#-----------
# arithmetic
#-----------
# +a
@vectorize 
def Pos(a, **kwargs):
    """ a: Signal or Immd
    """
    return HGL._sess.module._conf.dispatcher.call('Pos', Signal(a), **kwargs)

# -a
@vectorize 
def Neg(a, **kwargs):
    """ a: Signal or Immd
    """
    return HGL._sess.module._conf.dispatcher.call('Neg', Signal(a), **kwargs)

# a + b + c
@vectorize_axis 
def Add(*args, **kwargs):
    """ args: Signal or Immd
    """
    args = (Signal(i) for i in args)
    return HGL._sess.module._conf.dispatcher.call('Add', *args, **kwargs) 

# a - b - c
@vectorize_axis 
def Sub(*args, **kwargs):
    """ args: Signal or Immd
    """
    args = (Signal(i) for i in args)
    return HGL._sess.module._conf.dispatcher.call('Sub', *args, **kwargs) 

# a * b * c
@vectorize_axis 
def Mul(*args, **kwargs):
    """ args: Signal or Immd
    """
    args = (Signal(i) for i in args)
    return HGL._sess.module._conf.dispatcher.call('Mul', *args, **kwargs) 

# a @ b
def Matmul(a, b, **kwargs):
    """ a: Signal or Array, b: Signal or Array or Immd
    """
    return HGL._sess.module._conf.dispatcher.call('Matmul', a, b, **kwargs) 

# a / b
@vectorize
def Div(a, b, **kwargs):
    """ a: Signal, b: Signal or Immd 
    """
    return HGL._sess.module._conf.dispatcher.call('Div', Signal(a), Signal(b), **kwargs) 

# a % b
@vectorize
def Mod(a, b, **kwargs):
    """ a: Signal, b: Signal or Immd 
    """
    return HGL._sess.module._conf.dispatcher.call('Mod', Signal(a), Signal(b), **kwargs) 

# a // b
@vectorize
def Floordiv(a, b, **kwargs):
    """ a: Signal, b: Signal or Immd 
    """
    return HGL._sess.module._conf.dispatcher.call('Floordiv', Signal(a), Signal(b), **kwargs) 

# a ** b
@vectorize
def Pow(a, b, **kwargs): 
    """ a: Signal, b: int
    
    Cat signal a b times
    """
    return HGL._sess.module._conf.dispatcher.call('Pow', Signal(a), int(b), **kwargs) 




# TODO reduce binary op on array axis
def Reduce(f, x: Iterable, axis=None, **kwargs): ...
    



