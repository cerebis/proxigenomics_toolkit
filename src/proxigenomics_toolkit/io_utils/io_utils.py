import bz2
import gzip
import io
import json
import logging
import pickle
from collections.abc import Iterable
from copy import deepcopy
from typing import IO, List, Optional, TypeVar

import numpy as np
import toml
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
        # this produces a spurious warning in PyCharm
        # noinspection PyTypeChecker
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
        # this produces a spurious warning in PyCharm
        # noinspection PyTypeChecker
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


class InlineDumper(yaml.SafeDumper):
    """
    Custom YAML dumper class designed for inline representation of lists.

    The class customizes YAML output to represent lists in a compact, inline
    format. It extends the functionality of `yaml.SafeDumper` by adding a
    custom representer for lists.

    It is not expected to handle non-intrinics collections, such as
    from Numpy.

    :ivar stream: The output stream to which the YAML is dumped.
    :type stream: Protocol[str | bytes]
    """
    @staticmethod
    def _represent_list_inline(dumper: yaml.Dumper, data: Iterable) -> yaml.SequenceNode:
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
    def __init__(self, stream):
        super().__init__(stream, indent=4, sort_keys=True, default_flow_style=False)
        self.add_representer(list, self._represent_list_inline)


class CompactJSONEncoder(json.JSONEncoder):
    """A JSON Encoder that puts small containers on single lines."""

    CONTAINER_TYPES = (list, tuple, dict)
    """Container datatypes include primitives or other containers."""

    MAX_WIDTH = 200
    """Maximum width of a container that might be put on a single line."""

    MAX_ITEMS = 50
    """Maximum number of items in container that might be put on single line."""

    def __init__(self, *args, **kwargs):
        # using this class without indentation is pointless
        if kwargs.get("indent") is None:
            kwargs["indent"] = 4
        super().__init__(*args, **kwargs)
        self.indentation_level = 0

    def encode(self, o):
        """Encode JSON object *o* with respect to single line lists."""
        if isinstance(o, (list, tuple)):
            return self._encode_list(o)
        if isinstance(o, dict):
            return self._encode_object(o)
        return json.dumps(
            o,
            skipkeys=self.skipkeys,
            ensure_ascii=self.ensure_ascii,
            check_circular=self.check_circular,
            allow_nan=self.allow_nan,
            sort_keys=self.sort_keys,
            indent=self.indent,
            separators=(self.item_separator, self.key_separator),
            default=self.default if hasattr(self, "default") else None,
        )

    def _encode_list(self, o):
        if self._put_on_single_line(o):
            return "[" + ", ".join(self.encode(el) for el in o) + "]"
        self.indentation_level += 1
        output = [self.indent_str + self.encode(el) for el in o]
        self.indentation_level -= 1
        return "[\n" + ",\n".join(output) + "\n" + self.indent_str + "]"

    def _encode_object(self, o):
        if not o:
            return "{}"

        # ensure keys are converted to strings
        o = {str(k) if k is not None else "null": v for k, v in o.items()}

        if self.sort_keys:
            o = dict(sorted(o.items(), key=lambda x: x[0]))

        if self._put_on_single_line(o):
            return (
                "{ "
                + ", ".join(
                    f"{self.encode(k)}: {self.encode(el)}" for k, el in o.items()
                )
                + " }"
            )

        self.indentation_level += 1
        output = [
            f"{self.indent_str}{self.encode(k)}: {self.encode(v)}" for k, v in o.items()
        ]
        self.indentation_level -= 1

        return "{\n" + ",\n".join(output) + "\n" + self.indent_str + "}"

    def iterencode(self, o, **kwargs):
        """Required to also work with `json.dump`."""
        return self.encode(o)

    def _put_on_single_line(self, o):
        return (
            self._primitives_only(o)
            and len(o) <= self.MAX_ITEMS
            and len(str(o)) - 2 <= self.MAX_WIDTH
        )

    def _primitives_only(self, o: list | tuple | dict):
        if isinstance(o, (list, tuple)):
            return not any(isinstance(el, self.CONTAINER_TYPES) for el in o)
        elif isinstance(o, dict):
            return not any(isinstance(el, self.CONTAINER_TYPES) for el in o.values())

    @property
    def indent_str(self) -> str:
        if isinstance(self.indent, int):
            return " " * (self.indentation_level * self.indent)
        elif isinstance(self.indent, str):
            return self.indentation_level * self.indent
        else:
            raise ValueError(
                f"indent must either be of type int or str (is: {type(self.indent)})"
            )


def round_floats_in_obj(obj: T, precision: int) -> T:
    """
    Rounds all float or numpy floating numbers in the given object to the specified
    precision. The function recursively traverses the structures like dictionaries
    and iterables to round all floats. Iterables are returned as Lists, therefore
    it is expected that this function is used only in preparation for writing a
    serialization format, such as YAML, JSON, or TOML.

    :param obj: An object containing float, numpy floating numbers, dictionaries,
        or iterables that may contain such elements.
    :param precision: The number of decimal places to which float or numpy floating
        values will be rounded.
    :return: The same type as the input object with all float or numpy floating
        numbers rounded to the specified precision.
    """
    if isinstance(obj, (float, np.floating)):
        return float(round(obj, precision))
    if isinstance(obj, dict):
        return {k: round_floats_in_obj(v, precision) for k, v in obj.items()}
    if isinstance(obj, Iterable) and not isinstance(obj, str):
        return [round_floats_in_obj(x, precision) for x in obj]
    return obj


def serialize_simple_object(filename: str, obj: object, fmt: str='json', float_precision: int=4) -> None:

    def _write_json(obj: object, f_out) -> None:
        json.dump(obj, f_out, indent=4, sort_keys=True, cls=CompactJSONEncoder)

    def _write_yaml(obj: object, f_out) -> None:
        yaml.dump(obj, f_out, default_flow_style=False)

    def _write_toml(obj: object, f_out) -> None:
        toml.dump(obj, f_out)

    serializers = {'json': _write_json,
                   'yaml': _write_yaml,
                   'toml': _write_toml,}
    try:
        extension = filename.split('.')[-1].lower()
        if (fmt == 'json' or fmt == 'toml') and not extension == fmt:
            logger.warning(f'Filename {filename} does not end with .{fmt}')
        elif fmt == 'yaml' and not (extension == fmt or extension == '.yml'):
            logger.warning(f'Filename {filename} does not end with .{fmt} or .yml')

        with open(filename, 'wt') as f_out:
            serializers[fmt](round_floats_in_obj(deepcopy(obj), float_precision), f_out)
    except KeyError:
        raise ValueError(f'Unsupported serialization format: {fmt}')

