import argparse
import struct
from functools import reduce


def read_1_byte(file_object) -> bytes:
    assert file_object.mode == "rb"
    return file_object.read(1)


def read_2_byte(file_object) -> bytes:
    assert file_object.mode == "rb"
    return file_object.read(2)


def read_4_byte(file_object) -> bytes:
    assert file_object.mode == "rb"
    return file_object.read(4)


def read_8_byte(file_object) -> bytes:
    assert file_object.mode == "rb"
    return file_object.read(8)


def struct_unpack_char(data):
    assert len(data) == 1
    return struct.unpack("c", data)[0]


def struct_unpack_uint8(data):
    assert len(data) == 1
    return struct.unpack("B", data)[0]


def struct_unpack_int8(data):
    assert len(data) == 1
    return struct.unpack("b", data)[0]


def struct_unpack_uint16(data):
    assert len(data) == 2
    return struct.unpack("H", data)[0]


def struct_unpack_int16(data):
    assert len(data) == 2
    return struct.unpack("h", data)[0]


def struct_unpack_uint32(data):
    assert len(data) == 4
    return struct.unpack("I", data)[0]


def struct_unpack_int32(data):
    assert len(data) == 4
    return struct.unpack("i", data)[0]


def struct_unpack_uint64(data):
    assert len(data) == 8
    return struct.unpack("Q", data)[0]


def struct_unpack_int64(data):
    assert len(data) == 8
    return struct.unpack("q", data)[0]


def struct_unpack_float32(data):
    assert len(data) == 4
    return struct.unpack("f", data)[0]


def struct_unpack_float64(data):
    assert len(data) == 8
    return struct.unpack("d", data)[0]


def struct_unpack_bool(data):
    assert len(data) == 1
    return struct.unpack("?", data)[0]


class GGUFMetadataValue:
    GGUF_METADATA_VALUE_TYPE_UINT8 = 0
    GGUF_METADATA_VALUE_TYPE_INT8 = 1
    GGUF_METADATA_VALUE_TYPE_UINT16 = 2
    GGUF_METADATA_VALUE_TYPE_INT16 = 3
    GGUF_METADATA_VALUE_TYPE_UINT32 = 4
    GGUF_METADATA_VALUE_TYPE_INT32 = 5
    GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6
    GGUF_METADATA_VALUE_TYPE_BOOL = 7
    GGUF_METADATA_VALUE_TYPE_STRING = 8
    GGUF_METADATA_VALUE_TYPE_ARRAY = 9
    GGUF_METADATA_VALUE_TYPE_UINT64 = 10
    GGUF_METADATA_VALUE_TYPE_INT64 = 11
    GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12

    BYTES_VALUE_TYPE = GGUF_METADATA_VALUE_TYPE_UINT32

    @classmethod
    def read_value(cls, f, value_type):
        if value_type == cls.GGUF_METADATA_VALUE_TYPE_UINT8:
            return struct_unpack_uint8(read_1_byte(f))
        elif value_type == cls.GGUF_METADATA_VALUE_TYPE_INT8:
            return struct_unpack_int8(read_1_byte(f))
        elif value_type == cls.GGUF_METADATA_VALUE_TYPE_UINT16:
            return struct_unpack_uint16(read_2_byte(f))
        elif value_type == cls.GGUF_METADATA_VALUE_TYPE_INT16:
            return struct_unpack_int16(read_2_byte(f))
        elif value_type == cls.GGUF_METADATA_VALUE_TYPE_UINT32:
            return struct_unpack_uint32(read_4_byte(f))
        elif value_type == cls.GGUF_METADATA_VALUE_TYPE_INT32:
            return struct_unpack_int32(read_4_byte(f))
        elif value_type == cls.GGUF_METADATA_VALUE_TYPE_UINT64:
            return struct_unpack_uint64(read_8_byte(f))
        elif value_type == cls.GGUF_METADATA_VALUE_TYPE_INT64:
            return struct_unpack_int64(read_8_byte(f))
        elif value_type == cls.GGUF_METADATA_VALUE_TYPE_FLOAT32:
            return struct_unpack_float32(read_4_byte(f))
        elif value_type == cls.GGUF_METADATA_VALUE_TYPE_FLOAT64:
            return struct_unpack_float64(read_8_byte(f))
        elif value_type == cls.GGUF_METADATA_VALUE_TYPE_BOOL:
            return struct_unpack_bool(read_1_byte(f))
        elif value_type == cls.GGUF_METADATA_VALUE_TYPE_STRING:
            return GGUFString.read(f)
        elif value_type == cls.GGUF_METADATA_VALUE_TYPE_ARRAY:
            return GGUFArray.read(f)
        else:
            raise RuntimeError(f"No such value type : {value_type}")


class GGUFType:
    GGML_TYPE_F32 = 0
    GGML_TYPE_F16 = 1
    GGML_TYPE_Q4_0 = 2
    GGML_TYPE_Q4_1 = 3
    # GGML_TYPE_Q4_2 = 4
    # GGML_TYPE_Q4_3 = 5
    GGML_TYPE_Q5_0 = 6
    GGML_TYPE_Q5_1 = 7
    GGML_TYPE_Q8_0 = 8
    GGML_TYPE_Q8_1 = 9
    GGML_TYPE_Q2_K = 10
    GGML_TYPE_Q3_K = 11
    GGML_TYPE_Q4_K = 12
    GGML_TYPE_Q5_K = 13
    GGML_TYPE_Q6_K = 14
    GGML_TYPE_Q8_K = 15
    GGML_TYPE_I8 = 16
    GGML_TYPE_I16 = 17
    GGML_TYPE_I32 = 18
    GGML_TYPE_COUNT = 19

    @classmethod
    def to_string(cls, ggml_type):
        if ggml_type == cls.GGML_TYPE_F32:
            return "GGML_TYPE_FP32"
        elif ggml_type == cls.GGML_TYPE_F16:
            return "GGML_TYPE_FP16"
        elif ggml_type == cls.GGML_TYPE_Q4_0:
            return "GGML_TYPE_Q4_0"
        elif ggml_type == cls.GGML_TYPE_Q4_1:
            return "GGML_TYPE_Q4_1"
        elif ggml_type == cls.GGML_TYPE_Q5_0:
            return "GGML_TYPE_Q5_0"
        elif ggml_type == cls.GGML_TYPE_Q5_1:
            return "GGML_TYPE_Q5_1"
        elif ggml_type == cls.GGML_TYPE_Q8_0:
            return "GGML_TYPE_Q8_1"
        elif ggml_type == cls.GGML_TYPE_Q8_1:
            return "GGML_TYPE_Q8_1"
        elif ggml_type == cls.GGML_TYPE_Q2_K:
            return "GGML_TYPE_Q2_K"
        elif ggml_type == cls.GGML_TYPE_Q3_K:
            return "GGML_TYPE_Q3_K"
        elif ggml_type == cls.GGML_TYPE_Q4_K:
            return "GGML_TYPE_Q4_K"
        elif ggml_type == cls.GGML_TYPE_Q5_K:
            return "GGML_TYPE_Q5_K"
        elif ggml_type == cls.GGML_TYPE_Q6_K:
            return "GGML_TYPE_Q6_K"
        elif ggml_type == cls.GGML_TYPE_Q8_K:
            return "GGML_TYPE_Q8_K"
        elif ggml_type == cls.GGML_TYPE_I8:
            return "GGML_TYPE_I8"
        elif ggml_type == cls.GGML_TYPE_I16:
            return "GGML_TYPE_I16"
        elif ggml_type == cls.GGML_TYPE_I32:
            return "GGML_TYPE_I32"
        elif ggml_type == cls.GGML_TYPE_COUNT:
            return "GGML_TYPE_COUNT"
        else:
            raise RuntimeError(f"No such ggml type : {ggml_type}")


class GGUFString:
    GGUF_STRING_LEN_TYPE = GGUFMetadataValue.GGUF_METADATA_VALUE_TYPE_UINT64  # uint64_t

    @classmethod
    def read(cls, f):
        length = GGUFMetadataValue.read_value(f, GGUFString.GGUF_STRING_LEN_TYPE)

        data = f.read(length)
        value = data.decode("utf-8")
        return value


class GGUFArray:
    GGUF_ARRAY_LEN_TYPE = GGUFMetadataValue.GGUF_METADATA_VALUE_TYPE_UINT64  # uint64_t

    @classmethod
    def read(cls, f):
        # value_type
        value_type = GGUFMetadataValue.read_value(f, GGUFMetadataValue.BYTES_VALUE_TYPE)

        # len
        length = GGUFMetadataValue.read_value(f, GGUFArray.GGUF_ARRAY_LEN_TYPE)

        # value
        array = []
        for i in range(length):
            array.append(GGUFMetadataValue.read_value(f, value_type))
        return array


def pretty_print(kv, max_width=50):
    def truncate_if_overflow(string):
        if len(string) > max_width:
            return string[: max_width - 3] + "..."
        else:
            return string

    lengths = map(len, kv.keys())
    max_length = reduce(lambda x, y: x if x > y else y, lengths)
    for k, v in kv.items():
        print(f"{k:<{max_length+2}}=  {truncate_if_overflow(str(v))}")


def read_metadata_kv(f):
    # metadata key
    key = GGUFString.read(f)

    # value_type
    value_type = GGUFMetadataValue.read_value(f, GGUFMetadataValue.BYTES_VALUE_TYPE)

    # value
    value = GGUFMetadataValue.read_value(f, value_type)
    return key, value


def read_header(f):
    result = {}
    # magic + version + tensor_count + metadata_kv_count
    data = f.read(4 + 4 + 8 + 8)

    format_string = "IIQQ"
    values = struct.unpack(format_string, data)
    magic = hex(values[0])
    if magic != "0x46554747":
        raise RuntimeError(
            f"The magic number({magic}) is not correct. This file seems not .gguf file."
        )
    version = values[1]
    tensor_count = values[2]
    metadata_kv_count = values[3]

    result["magic"] = magic
    result["version"] = version
    result["tensor_count"] = tensor_count
    result["metadata_kv_count"] = metadata_kv_count

    for i in range(metadata_kv_count):
        key, value = read_metadata_kv(f)
        result[key] = value

    pretty_print(result)

    return result


def read_tensor_infos(f, tensor_count):
    if tensor_count == 0:
        print("No tensor stored in this gguf file.")
    for _ in range(tensor_count):
        result = {}

        name = GGUFMetadataValue.read_value(
            f, GGUFMetadataValue.GGUF_METADATA_VALUE_TYPE_STRING
        )

        n_dimensions = GGUFMetadataValue.read_value(
            f, GGUFMetadataValue.GGUF_METADATA_VALUE_TYPE_UINT32
        )

        shape = []
        for __ in range(n_dimensions):
            dim = GGUFMetadataValue.read_value(
                f, GGUFMetadataValue.GGUF_METADATA_VALUE_TYPE_UINT64
            )
            shape.append(dim)

        ggml_type = GGUFType.to_string(
            GGUFMetadataValue.read_value(
                f, GGUFMetadataValue.GGUF_METADATA_VALUE_TYPE_UINT32
            )
        )
        offset = GGUFMetadataValue.read_value(
            f, GGUFMetadataValue.GGUF_METADATA_VALUE_TYPE_UINT64
        )

        result["name"] = name
        result["n_dimensions"] = n_dimensions
        result["shape"] = shape
        result["ggml_type"] = ggml_type
        result["offset"] = offset
        pretty_print(result, max_width=50)
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Parse & Print GGUF header & tensor_infos"
    )
    parser.add_argument("filename", help=".gguf file")
    args = parser.parse_args()

    with open(args.filename, "rb") as f:
        header_kv = read_header(f)
        tensor_infos_kv = read_tensor_infos(f, header_kv["tensor_count"])


if __name__ == "__main__":
    main()
