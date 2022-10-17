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




class _Session:
    def __init__(self):                 self.sess: object = None 
    def __set__(self, obj, sess):       self.sess = sess
    def __get__(self, obj, cls=None):   return self.sess   


class HGL:

    __slots__ = () 
    _sess = _Session()

    @property  
    def __hgl_version__(self):      
        return (0,0,1)
    
    @property 
    def __hgl_type__(self):
        return type(self) 
    
    def __copy__(self):             
        return self      
    
    def __hash__(self):             
        return id(self)
    
    def __bool__(self):             
        raise NotImplementedError(self.__class__) 
    
    def __eq__(self, x):            
        raise NotImplementedError(self.__class__) 

    def __str__(self):              
        return object.__repr__(self) 

    def __repr__(self):             
        return str(self)



