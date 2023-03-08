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

class Value(HGL):
    def __init__(self) -> None:
        self.v64 = 0 
        self.x64 = 0 
        self.len = 64
        self.idx = 0


class Node(HGL):

    def __init__(self) -> None:
        self.iports: List[Value] = []
        self.oports: List[Value] = []
        self.gates: List[hgl_basic.Gate] = []