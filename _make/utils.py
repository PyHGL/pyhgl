import os 
import sys 
import subprocess
import inspect
import re 

def python(cmd: str):
    python = sys.executable
    subprocess.run(f"{python} {cmd}" )  

def relative_path(path: str, level: int = 1, check_exist: bool = False) -> str:
    """ 
    path: 
        directory/filename that relative to caller's directory/path. ex. ../a, ./b/, c/
    level: 
        nth caller 
    check_exist: 
        whether check the existance of path 

    return:
        an absolute path
    """
    if os.path.isabs(path):
        ret = path 
    else:
        a = os.path.dirname(inspect.stack()[level].filename)
        ret = os.path.abspath(os.path.join(a, path)) 
    if check_exist:
        assert os.path.exists(ret), f'path {ret} does not exit.'
    return re.sub(r'\\', '/', ret)  

def yellow(x: str) -> str:     
    return f'\033[0;33m{x}\033[0m' 

def finish(level: int = 1):
    """ match task names and then execute
    """
    f_locals = sys._getframe(level).f_locals
    tasks = [(k, v) for k,v in f_locals.items() if k.startswith('make_') and inspect.isfunction(v)]
    del f_locals 
    argv = '|'.join(sys.argv[1:])
    if not argv: 
        for f_name, f in tasks:
            print(yellow(f"python -m _make {f_name[5:]:<10}"), f.__doc__) 
        print(yellow(f"python -m _make {'.*':<11}"), 'run all tasks inorder')
        sys.exit(0)
    filter = f"make_({argv})"  
    for f_name, f in tasks:
        if re.match(filter, f_name):
            print(yellow(f_name))
            f() 
    sys.exit(0)