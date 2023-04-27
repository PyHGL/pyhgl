import io
import inspect
import traceback 
import re 
import os

_pyhgl_exec = re.compile(r'PyHGL Traceback:')

def format_hgl_stack(n_ignored: int = 2, max_n: int = 100) -> str:
    """ return part of traced frame
    
    filename:lineno \n code \n
    """
    stack = inspect.stack() 
    i = 0 
    for frame in stack:
        if (lines:=frame.code_context) is None:
            break
        else:
            line = lines[0] 
        if _pyhgl_exec.search(line):
            break 
        i += 1 
        if i >= n_ignored+max_n:
            break

    useful_frames = stack[n_ignored:i]
    ret = []
    for frame in useful_frames:
        if frame.code_context:
            code = frame.code_context[0] 
        else:
            code = ' \n'
        ret.append(f'  {frame.filename}:{frame.lineno}\n    {code.lstrip()}')
    return ''.join(ret)

def relative_path(path: str, level: int = 1, check_exist: bool = False) -> str:
    """ 
    path: 
        path of dir/file, relative to caller's directory/path. ex. ../a, ./b/, c/
    level: 
        nth caller 
    check_exist: 
        whether check the existance of path 

    return:
        absolute path
    """
    if os.path.isabs(path):
        ret = path 
    else:
        a = os.path.dirname(inspect.stack()[level].filename)
        ret = os.path.abspath(os.path.join(a, path)) 
    if check_exist:
        assert os.path.exists(ret) 
    return re.sub(r'\\', '/', ret)  
