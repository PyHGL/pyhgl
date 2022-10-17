import io
import inspect
import traceback 
import re 

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

    useful_frames = stack[n_ignored:min(i, n_ignored+max_n)]
    ret = []
    for i in useful_frames:
        if i.code_context:
            code = i.code_context[0] 
        else:
            code = ' \n'
        ret.append(f'  {i.filename}:{i.lineno}\n    {code.lstrip()}')
    return ''.join(reversed(ret))