from .convert import * 
from .unit import *
from .trace import * 
from .integer import * 


def get_dll_close():
    import sys, platform, ctypes
    OS = platform.system()
    if OS == "Windows":  # pragma: Windows
        dll_close = ctypes.windll.kernel32.FreeLibrary
    elif OS == "Darwin":
        try:
            try:
                # macOS 11 (Big Sur). Possibly also later macOS 10s.
                stdlib = ctypes.CDLL("libc.dylib")
            except OSError:
                stdlib = ctypes.CDLL("libSystem")
        except OSError:
            # Older macOSs. Not only is the name inconsistent but it's
            # not even in PATH.
            stdlib = ctypes.CDLL("/usr/lib/system/libsystem_c.dylib")
        dll_close = stdlib.dlclose
    elif OS == "Linux":
        try:
            stdlib = ctypes.CDLL("")
        except OSError:
            # Alpine Linux.
            stdlib = ctypes.CDLL("libc.so")
        dll_close = stdlib.dlclose
    elif sys.platform == "msys":
        # msys can also use `ctypes.CDLL("kernel32.dll").FreeLibrary()`. Not sure
        # if or what the difference is.
        stdlib = ctypes.CDLL("msys-2.0.dll")
        dll_close = stdlib.dlclose
    elif sys.platform == "cygwin":
        stdlib = ctypes.CDLL("cygwin1.dll")
        dll_close = stdlib.dlclose
    elif OS == "FreeBSD":
        # FreeBSD uses `/usr/lib/libc.so.7` where `7` is another version number.
        # It is not in PATH but using its name instead of its path is somehow the
        # only way to open it. The name must include the .so.7 suffix.
        stdlib = ctypes.CDLL("libc.so.7")
        dll_close = stdlib.close
    else:
        raise NotImplementedError("Unknown platform.")

    dll_close.argtypes = [ctypes.c_void_p]
    return dll_close