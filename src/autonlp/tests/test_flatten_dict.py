from autonlp.utils import flatten_dict


def test_flatten_dict():
    d = {
        "a": 1,
        "c": {"a": 2, "b": {"x": 5, "y": 10}},
        "d": [1, 2, 3],
    }

    flat_dict = flatten_dict(d, max_depth=0)
    assert flat_dict == d

    flat_dict = flatten_dict(d, max_depth=5)
    assert flat_dict == {"a": 1, "c.a": 2, "c.b.x": 5, "c.b.y": 10, "d": [1, 2, 3]}

    flat_dict = flatten_dict(d, max_depth=1)
    assert flat_dict == {"a": 1, "c.a": 2, "c.b": {"x": 5, "y": 10}, "d": [1, 2, 3]}
