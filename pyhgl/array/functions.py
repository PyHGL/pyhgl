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


from ._hgl import HGL
from .dispatch import singleton, dispatch, Dispatcher, default_dispatcher, register_builtins
from .array import *



class HGLPattern(HGL):
    """ hardware assertion
    """
    @property 
    def __hgl_type__(self):
        return HGLPattern 

class HGLFunction(HGL):
    
    @property 
    def __hgl_type__(self):
        return HGLFunction 
        

#--------
# bitwise
#--------
@vectorize_axis
def Nand(*args, **kwargs):
    """ args: Signal or Immd
    """
    args = (Signal(i) for i in args)
    return HGL._sess.module._conf.dispatcher.call('Nand', *args, **kwargs)  

@vectorize_axis
def Nor(*args, **kwargs):
    """ args: Signal or Immd
    """
    args = (Signal(i) for i in args)
    return HGL._sess.module._conf.dispatcher.call('Nor', *args, **kwargs) 

@vectorize_axis
def Nxor(*args, **kwargs):
    """ args: Signal or Immd
    """
    args = (Signal(i) for i in args)
    return HGL._sess.module._conf.dispatcher.call('Nxor', *args, **kwargs) 

@vectorize 
def AndR(a, **kwargs):
    """ a: Signal or Immd
    """
    return HGL._sess.module._conf.dispatcher.call('AndR', Signal(a), **kwargs) 

@vectorize 
def OrR(a, **kwargs):
    """ a: Signal or Immd
    """
    return HGL._sess.module._conf.dispatcher.call('OrR', Signal(a), **kwargs) 

@vectorize 
def XorR(a, **kwargs):
    """ a: Signal or Immd
    """
    return HGL._sess.module._conf.dispatcher.call('XorR', Signal(a), **kwargs) 
 
@vectorize
def AddFull(a, b, **kwargs):
    """ a: Signal, b: Signal or Immd 
    """
    return HGL._sess.module._conf.dispatcher.call('AddFull', Signal(a), Signal(b), **kwargs) 

@vectorize
def MulFull(a, b, **kwargs):
    """ a: Signal, b: Signal or Immd 
    """
    return HGL._sess.module._conf.dispatcher.call('MulFull', Signal(a), Signal(b), **kwargs) 
 

#-------
# logic
#-------
@vectorize 
def Bool(a, **kwargs):
    """ a: Signal or Immd

    return a 1-bit uint signal, may not generate new gate
    """
    return HGL._sess.module._conf.dispatcher.call('Bool', Signal(a), **kwargs)
# !a
@vectorize 
def LogicNot(a, **kwargs):
    """ a: Signal or Immd
    """
    return HGL._sess.module._conf.dispatcher.call('LogicNot', Signal(a), **kwargs)

# a && b
@vectorize_axis
def LogicAnd(*args, **kwargs):
    """ args: Signal or Immd
    """
    args = (Signal(i) for i in args)
    return HGL._sess.module._conf.dispatcher.call('LogicAnd', *args, **kwargs) 


# a || b
@vectorize_axis
def LogicOr(*args, **kwargs):
    """ args: Signal or Immd
    """
    args = (Signal(i) for i in args)
    return HGL._sess.module._conf.dispatcher.call('LogicOr', *args, **kwargs) 


#--------
# slicing
#--------
@vectorize_axis
def Cat(*args, **kwargs):
    """ args: Signal or Immd
    """
    args = (Signal(i) for i in args)
    return HGL._sess.module._conf.dispatcher.call('Cat', *args, **kwargs) 
    
@vectorize_first 
def Slice(x, **kwargs):
    """ x: Signal or Immd, key = ...
    """
    return HGL._sess.module._conf.dispatcher.call('Slice', Signal(x), **kwargs)

#-----------
# converting
#-----------
@vectorize 
def Convert(a, b, **kwargs):
    """ a: Signal, b: SignalType
    """
    return HGL._sess.module._conf.dispatcher.call('Convert', a, b, **kwargs)


#--------
# netlist
#-------- 
@vectorize
def Wire(x, **kwargs):
    return HGL._sess.module._conf.dispatcher.call('Wire', Signal(x), **kwargs) 

@vectorize
def WireNext(x, **kwargs):
    return HGL._sess.module._conf.dispatcher.call('WireNext', Signal(x), **kwargs) 

@vectorize
def Reg(x, **kwargs):
    return HGL._sess.module._conf.dispatcher.call('Reg', Signal(x), **kwargs) 

@vectorize
def RegNext(x, **kwargs):
    return HGL._sess.module._conf.dispatcher.call('RegNext', Signal(x), **kwargs) 

@vectorize
def Latch(x, **kwargs):
    return HGL._sess.module._conf.dispatcher.call('Latch', Signal(x), **kwargs) 

@vectorize
def Wtri(x, **kwargs):
    return HGL._sess.module._conf.dispatcher.call('Wtri', Signal(x), **kwargs) 

@vectorize
def Wor(x, **kwargs):
    return HGL._sess.module._conf.dispatcher.call('Wor', Signal(x), **kwargs) 

@vectorize
def Wand(x, **kwargs):
    return HGL._sess.module._conf.dispatcher.call('Wand', Signal(x), **kwargs) 



#-------------------------------- 
# extended hgl operators: ! && ||  
#-------------------------------- 

    
@singleton 
class __hgl_logicnot__(HGL):
    """ PyHGL operator ! 
    """
    def __call__(self, a):
        if isinstance(a, HGLPattern):
            return HGL._sess.module._conf.dispatcher.call('Assert_Not', a)
        else:
            return LogicNot(a)    
    
    
@singleton 
class __hgl_logicand__(HGL):
    """ PyHGL operator && 
    """
    def __call__(self, a, b):
        return LogicAnd(a,b)    
    
    
@singleton 
class __hgl_logicor__(HGL):
    """ PyHGL operator ||
    """
    def __call__(self, a, b):
        if isinstance(a, HGLPattern) or isinstance(b, HGLPattern):
            return HGL._sess.module._conf.dispatcher.call('Assert_Or', a, b)
        else:
            return LogicOr(a,b)     
    

@singleton 
class __hgl_unit__(HGL):
    """ unit: 1.2`m/s`
    """
    def __call__(self, v, f: callable):
        # f()
        pass