import gzip
import os
import io
import pickle
import json
import yaml

import pytest
from src.proxigenomics_toolkit.io_utils.io_utils import save_object, open_output, load_object, multicopy_tostream, \
    multicopy_tofile, write_to_stream, read_from_stream


def test_save_object_creates_file():
    obj = {"key": "value"}
    file_name = "test_save_object.gz"

    save_object(file_name, obj)

    assert os.path.isfile(file_name), "File was not created."
    os.remove(file_name)

def test_write_to_stream_with_plain_format():
    data = "test string"
    buffer = io.StringIO()

    write_to_stream(buffer, data, fmt="plain")
    assert buffer.getvalue() == "test string\n", "Plain format output does not match expected."


def test_read_from_stream_with_yaml_format():
    data = {"key": "value"}
    buffer = io.StringIO()
    yaml.dump(data, buffer)
    buffer.seek(0)

    result = read_from_stream(buffer, fmt="yaml")
    assert result == data, "YAML input does not match expected dictionary."


def test_read_from_stream_with_json_format():
    data = {"key": "value"}
    buffer = io.StringIO()
    json.dump(data, buffer)
    buffer.seek(0)

    result = read_from_stream(buffer, fmt="json")
    assert result == data, "JSON input does not match expected dictionary."


def test_read_from_stream_unsupported_format():
    data = "test string"
    buffer = io.StringIO(data)

    with pytest.raises(ValueError, match="Unsupported format.*"):
        read_from_stream(buffer, fmt="unsupported")

def test_write_to_stream_with_json_format():
    data = {"key": "value"}
    buffer = io.StringIO()

    write_to_stream(buffer, data, fmt="json")
    expected_output = json.dumps(data, indent=1)
    assert buffer.getvalue() == expected_output, "JSON format output does not match expected."


def test_write_to_stream_with_yaml_format():
    data = {"key": "value"}
    buffer = io.StringIO()

    write_to_stream(buffer, data, fmt="yaml")
    expected_output = yaml.dump(data, default_flow_style=False)
    assert buffer.getvalue() == expected_output, "YAML format output does not match expected."


def test_write_to_stream_unsupported_format():
    data = "test string"
    buffer = io.StringIO()

    with pytest.raises(ValueError, match="Unsupported format.*"):
        write_to_stream(buffer, data, fmt="unsupported")

def test_load_object_gz_file():
    obj = {"key": "value"}
    file_name = "test_load_object.gz"

    with gzip.open(file_name, 'wb') as f:
        pickle.dump(obj, f)

    loaded_obj = load_object(file_name)

    assert loaded_obj == obj, "Loaded object content does not match original."
    os.remove(file_name)


def test_save_object_correct_content():
    obj = {"key": "value"}
    file_name = "test_save_object.gz"

    save_object(file_name, obj)

    with gzip.open(file_name, 'rb') as f:
        loaded_obj = pickle.load(f)

    assert loaded_obj == obj, "Saved object content does not match original."
    os.remove(file_name)


def test_save_object_with_non_gz_extension():
    obj = {"key": "value"}
    file_name = "test_save_object"

    save_object(file_name, obj)

    expected_file_name = file_name + ".gz"
    assert os.path.isfile(expected_file_name), "File with .gz extension was not created."
    os.remove(expected_file_name)


def test_open_output_with_gzip_compression():
    file_name = "test_open_output.gz"

    with open_output(file_name, compress='gzip') as out_h:
        out_h.write(b"test data")

    with gzip.open(file_name, 'rb') as f:
        content = f.read()

    assert content == b"test data", "Content written to file does not match expected."
    os.remove(file_name)


def test_load_object_txt_file():
    obj = {"key": "value"}
    file_name = "test_load_object.txt"

    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)

    loaded_obj = load_object(file_name)

    assert loaded_obj == obj, "Loaded object content does not match original."
    os.remove(file_name)


def test_open_output_without_compression():
    file_name = "test_open_output.txt"

    with open_output(file_name, mode='wb') as out_h:
        out_h.write(b"plain text data")

    with open(file_name, 'rb') as f:
        content = f.read()

    assert content == b"plain text data", "Content written to file does not match expected for uncompressed file."
    os.remove(file_name)

@pytest.fixture
def text_data(tmp_path):
    input_file = os.path.join(tmp_path, "test_multicopy_input.txt")
    output_file1 = os.path.join(tmp_path, "test_multicopy_output1.txt")
    output_file2 = os.path.join(tmp_path, "test_multicopy_output2.txt")
    data = "Test data for multicopy_tofile"
    with open(input_file, 'w') as f:
        f.write(data)
    yield {'data': data,
           'rmode': 'r', 'wmode': 'w',
           'in': input_file,
           'out1': output_file1,
           'out2': output_file2
           }
    os.remove(input_file)
    os.remove(output_file1)
    os.remove(output_file2)


@pytest.fixture
def binary_data(tmp_path):
    input_file = os.path.join(tmp_path, "test_multicopy_input.bin")
    output_file1 = os.path.join(tmp_path, "test_multicopy_output1.bin")
    output_file2 = os.path.join(tmp_path, "test_multicopy_output2.bin")
    data = b"Test data for multicopy_tofile"
    with open(input_file, 'wb') as f:
        f.write(data)
    yield {'data': data,
           'rmode': 'rb', 'wmode': 'wb',
           'in': input_file,
           'out1': output_file1,
           'out2': output_file2
           }
    os.remove(input_file)
    os.remove(output_file1)
    os.remove(output_file2)


def test_multicopy_tostream_writes_to_multiple_streams2(text_data, binary_data):
    for _d, _isbin in [(text_data, False), (binary_data, True)]:
        # Create output file streams and write data
        with open(_d['out1'], _d['wmode']) as out1, open(_d['out2'], _d['wmode']) as out2:
            multicopy_tostream(_d['in'], out1, out2, binary_io=_isbin)
        # Verify _d in both output files
        with open(_d['out1'], _d['rmode']) as out1, open(_d['out2'], _d['rmode']) as out2:
            assert out1.read() == _d['data'], "Data mismatch in output_file1."
            assert out2.read() == _d['data'], "Data mismatch in output_file2."


def test_multicopy_tofile_copies_to_multiple_files(text_data, binary_data):
    for _d, _isbin in [(text_data, False), (binary_data, True)]:
        multicopy_tofile(_d['in'], _d['out1'], _d['out2'], binary_io=_isbin)
        with open(_d['out1'], _d['rmode']) as out1, open(_d['out2'], _d['rmode']) as out2:
            assert out1.read() == _d['data'], "Data mismatch in output_file1."
            assert out2.read() == _d['data'], "Data mismatch in output_file2."
