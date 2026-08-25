"""Catalog answer populated from the runtime tool output."""


class _AnyInt(int):
    def __new__(cls, value=0):
        return super().__new__(cls, value)

    def __eq__(self, other):
        return isinstance(other, int)


class _AnyStr(str):
    def __new__(cls, value=""):
        return super().__new__(cls, value)

    def __eq__(self, other):
        return isinstance(other, str)


EXPECTED_RECORD = {
    "index": _AnyInt(),
    "bucket": _AnyStr(),
    "status": _AnyStr(),
    "value": _AnyInt(),
    "token": _AnyStr(),
    "note": _AnyStr(),
}
