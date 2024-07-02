"""Caches expensive function calls in pickled bytes on disk.

Code bases are from https://gist.github.com/soaxelbrooke/97b4ac9f829ade33510eadb1ca97de1e
"""

import base64
import hashlib
import os
import shutil
import subprocess
from functools import wraps

import pandas as pd
from termcolor import cprint

try:
    import dill as pickle
except ModuleNotFoundError:
    import pickle


def clear_caches():
    """ Delete all cache directories created by fscache """
    for dirname in filter(lambda s: s.startswith('.fscache'), os.listdir('./')):
        shutil.rmtree(dirname)


def clear_cache(dir_path: str):
    """ Delete cache directory for this context """
    try:
        shutil.rmtree(dir_path)
    except FileNotFoundError:
        pass


def get(dir_path: str, key: str, verbose: bool = False):
    """ Get object from cache if present, return None otherwise """
    if dir_path not in os.listdir('./'):
        subprocess.call(['mkdir', '-p', dir_path])
    if key in os.listdir(dir_path + '/'):
        full_path = '{}/{}'.format(dir_path, key)
        if key.endswith("df.csv"):
            loaded = pd.read_csv(full_path)
        else:
            with open(full_path, 'rb') as infile:
                loaded = pickle.load(infile)
        if verbose:
            cprint(f"Loaded from {os.path.abspath(full_path)}", "green")
        return loaded


def put(dir_path: str, key: str, obj, strict_exceptions=False, verbose: bool = False):
    """ Put object to file system cache, maybe die on invalid key name """
    try:
        full_path = '{}/{}'.format(dir_path, key)
        if type(obj) == pd.DataFrame and key.endswith("df.csv"):
            obj.to_csv(full_path, index=False)
        else:
            with open(full_path, 'wb') as outfile:
                pickle.dump(obj, outfile)
        if verbose:
            cprint(f"Saved at {os.path.abspath(full_path)}", "blue")
    except Exception as e:
        if strict_exceptions:
            raise e
        else:
            print(e)


def default_key_fn(*args, **kwargs) -> str:
    kw_pairs = tuple('{}={}'.format(k, v) for k, v in kwargs.items())
    hasher = hashlib.md5()
    hasher.update(':'.join(map(str, args + kw_pairs)).encode('utf-8'))
    return base64.standard_b64encode(hasher.digest()).decode('utf-8').replace('/', '+')


def repr_kv_fn(*args, **kwargs) -> str:
    kvs = [str(arg) for arg in args]
    for k, v in kwargs.items():
        kvs.append(f"{k}={v}")
    return f'{"+".join(kvs)}'


def repr_short_kv_fn(*args, **kwargs) -> str:
    SHORT = 25
    short_kvs = []
    long_args, long_kwargs = [], {}
    for arg in args:
        if len(str(arg)) <= SHORT:
            short_kvs.append(str(arg))
        else:
            long_args.append(arg)
    for k, v in kwargs.items():
        if len(str(v)) <= SHORT:
            short_kvs.append(f"{k}={v}")
        else:
            long_kwargs[k] = v
    hashed = default_key_fn(*long_args, **long_kwargs)
    return f'{"+".join(short_kvs)}_{hashed[:5]}'


def repr_short_kwargs_kv_fn(*args, **kwargs) -> str:
    SHORT = 25
    short_kvs = []
    long_args, long_kwargs = [], {}
    for arg in args:
        short_kvs.append(str(arg))
    for k, v in kwargs.items():
        if len(str(v)) <= SHORT:
            short_kvs.append(f"{k}={v}")
        else:
            long_kwargs[k] = v
    hashed = default_key_fn(*long_args, **long_kwargs)
    return f'{"+".join(short_kvs)}_{hashed[:5]}'


def fscaches(path: str = None,
             path_attrname_in_kwargs: str = None,
             key_fn=repr_short_kv_fn,
             keys_to_exclude=None,
             extension="pickle",
             verbose=False):
    """
    :param path: absolute or relative path
    :param path_attrname_in_kwargs: attribute name of the path in kwargs
    :param key_fn:
    :return:
    """
    assert path is not None or path_attrname_in_kwargs is not None

    def outer_wrapper(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):

            if path_attrname_in_kwargs is not None:
                _path = kwargs[path_attrname_in_kwargs]
                _key_kwargs = {k: v for k, v in kwargs.items() if k != path_attrname_in_kwargs}
            else:
                _path = path
                _key_kwargs = kwargs

            if keys_to_exclude is not None:
                _key_kwargs = {k: v for k, v in _key_kwargs.items() if k not in keys_to_exclude}

            if _path is None:
                if verbose:
                    cprint(f"Path in kwargs is None.", "green")
                return fn(*args, **kwargs)
            else:
                _path = os.path.join(_path, fn.__name__)
                key = f"{key_fn(*args, **_key_kwargs)}" + f".{extension}"
                maybe_loaded = get(_path, key, verbose=verbose)
                if maybe_loaded is not None:
                    return maybe_loaded
                result = fn(*args, **kwargs)
                put(_path, key, result, verbose=verbose)
                return result

        return wrapper

    return outer_wrapper


if __name__ == '__main__':
    @fscaches(path='../fscaches')
    def inc(num_1, num_2):
        return num_1 + 1


    print(inc(1, num_2=1))
