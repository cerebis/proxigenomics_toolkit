import bz2
import gzip
import io
import json
import logging
import pickle
from typing import IO, List, Optional, TypeVar

import yaml

logger = logging.getLogger(__name__)

# default buffer for incremental read/write
DEF_BUFFER = 16384

T = TypeVar('T')


def save_object(file_name: str, obj: object) -> None:
    """
    Serialize an object to a file with gzip compression. .gz will automatically be
    added if missing.

    :param file_name: output file name
    :param obj: object to serialize
    """
    with open_output(file_name, compress='gzip', mode='wb') as out_h:
        pickle.dump(obj, out_h)


def load_object(file_name: str) -> T:
    """
    Deserialize an object from a file with automatic support for compression.

    :param file_name: input file name
    :return: deserialzied object
    """
    with open_input(file_name) as in_h:
        try:
            return pickle.load(in_h)
        except UnicodeError:
            in_h.seek(0)
            logger.warning('Error loading pickled object {}. '
                           'Retrying assuming it was created under Python 2'.format(file_name))
            return pickle.load(in_h, encoding='latin1')


def open_input(file_name: str, mode: str='rb') -> IO:
    """
    Open a text file for input. The filename is used to indicate if it has been
    compressed. Recognising gzip and bz2.

    :param file_name: the name of the input file
    :param mode: file access mode
    :return: open file handle, possibly wrapped in a decompressor
    """
    suffix = file_name.split('.')[-1].lower()
    if suffix == 'bz2':
        return bz2.open(file_name, mode)
    elif suffix == 'gz':
        return gzip.open(file_name, mode)
    return open(file_name, mode)


def open_output(file_name: str,
                append: bool=False,
                compress: Optional[str]=None,
                mode: str='w') -> IO:
    """
    Open a text stream for reading or writing. Compression can be enabled
    with either 'bzip2' or 'gzip'. Additional option for gzip compression
    level. Compressed filenames are only appended with suffix if not included.

    :param file_name: file name of output
    :param append: append to any existing file
    :param compress: gzip, bzip2
    :param mode: file access mode, default 'w' for writing, 'a' for appending.
    :return:
    """
    if append:
        if not mode.endswith('+'):
            mode += '+'
    else:
        if mode.endswith('+'):
            raise IOError(f'Append set to false while mode was {mode}')

    if compress == 'bzip2':
        if not file_name.endswith('.bz2'):
            file_name += '.bz2'
        # bz2 missing method to be wrapped by BufferedWriter. Just directly
        # supply a buffer size
        return bz2.open(file_name, mode)
    elif compress == 'gzip':
        if not file_name.endswith('.gz'):
            file_name += '.gz'
        return gzip.open(file_name, mode)
    else:
        out_h = io.BufferedWriter(io.FileIO(file_name, mode))
        if 'b' not in mode:
            return io.TextIOWrapper(out_h)
        return out_h


def multicopy_tostream(file_name: str,
                       ostreams: List[IO],
                       bufsize: Optional[int]=None,
                       binary_io: bool=False) -> None:
    """
    Copy an input file to multiple output streams.
    :param file_name: input file name
    :param ostreams: output streams
    :param bufsize: buffer size
    :param binary_io: use binary I/O for output files. Default: False.
    :return:
    """
    with open(file_name, 'rb' if binary_io else 'r') as in_h:
        done = False
        while not done:
            buf = in_h.read(bufsize)
            if not buf:
                done = True
            for oi in ostreams:
                oi.write(buf)


def multicopy_tofile(file_name: str,
                     output_names: List[str],
                     bufsize: Optional[int]=None,
                     binary_io: bool=False,
                     compress: Optional[str]=None) -> None:
    """
    Copy an input file to multiple output files.
    :param file_name: input file name
    :param output_names: output file names
    :param bufsize: buffer size
    :param binary_io: use binary I/O for output files. Default: False.
    :param compress: gzip, bzip2, None
    :return:
    """
    assert compress is None or compress in ['gzip', 'bzip2'], 'compress must be \"gzip\" or \"bzip2\"'
    if binary_io:
        read_mode = 'rb'
        write_mode = 'wb'
    else:
        read_mode = 'r'
        write_mode = 'w'

    out_h = None
    try:
        print(read_mode,write_mode)
        in_h = open(file_name, read_mode)
        out_h = [open_output(_name, compress=compress, mode=write_mode) for _name in output_names]

        done = False
        while not done:
            buf = in_h.read(bufsize)
            if not buf:
                done = True
            for _hndl in out_h:
                _hndl.write(buf)
    finally:
        if out_h:
            for _hndl in out_h:
                if _hndl:
                    _hndl.close()

def write_to_stream(stream: IO, data: object, fmt: str= 'plain') -> None:
    """
    Write an object out to a stream, possibly using a serialization format
    different to default string representation.

    :param stream: open stream to twrite
    :param data: object instance
    :param fmt: plain, json or yaml
    """
    if fmt == 'yaml':
        yaml.dump(data, stream, default_flow_style=False)
    elif fmt == 'json':
        json.dump(data, stream, indent=1)
    elif fmt == 'plain':
        stream.write('{0}\n'.format(data))
    else:
        raise ValueError('Unsupported format: {0}'.format(fmt))


def read_from_stream(stream: IO, fmt: str='yaml') -> object:
    """
    Load an object instance from a serialized format. How, in terms of classes
    the object is represented will depend on the serialized information. For
    generic serialized formats, this is more than likely to involve dictionaries
    for classes with properties.

    :param stream: open stream to read
    :param fmt: yaml or json
    :return: loaded object
    """
    if fmt == 'yaml':
        return yaml.safe_load(stream)
    elif fmt == 'json':
        return json.load(stream)
    else:
        raise ValueError('Unsupported format: {0}'.format(fmt))
