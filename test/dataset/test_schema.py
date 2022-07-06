from gluonts.dataset.schema import (
    PyArrayField,
    Schema
)

def test_infer_schema():
    expected_schema = Schema(
        {
            "x": int,
            "y": float,
            "z": str,
            "w": PyArrayField([3,], float),
            "m": PyArrayField([3,], str)
        }
    )
    data = {
        "x": 6,
        "y": 3.4,
        "z": "str",
        "w": [2.3, 4.5, 9.2],
        "m": ["2022-3-3", "2022-3-4", "2022-3-5"]
    }

    infer_schema = Schema.infer(data)
    assert(expected_schema == infer_schema)