import os
import re
import sys
import shutil 
import subprocess 
import time
import inspect

def _red(x: str) -> str:        return f'\033[0;31m{x}\033[0m'
def _green(x: str) -> str:      return f'\033[0;32m{x}\033[0m'
def _yellow(x: str) -> str:     return f'\033[0;33m{x}\033[0m'
def _blue(x: str) -> str:       return f'\033[0;34m{x}\033[0m'
def _purple(x: str) -> str:     return f'\033[0;35m{x}\033[0m'

def _fill_terminal(x: str, char: str, pos: str = 'center') -> str:
    try:
        terminal_width = shutil.get_terminal_size().columns
    except:
        terminal_width = 80
    n = terminal_width - len(x) - 1
    if n < 2:
        return x 
    else:
        if pos == 'center':
            left = n // 2 
            right = n - left
            return f'{left*char}{x}{right*char}'
        elif pos == 'left':
            return f'{x}{char*n}'
        elif pos == 'right':
            return f'{char*n}{x}'
        else:
            return x 
        
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

def caller_filename(level: int = 1):
    ret = inspect.stack()[level+1].filename
    return re.sub(r'\\', '/', ret) 


def run_python(cmd: str):
    python = sys.executable
    subprocess.run(f"{python} {cmd}" )  
    
    
