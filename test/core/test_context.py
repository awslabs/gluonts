from gluonts.core.context import Context, let, inject


class MyContext(Context):
    foo: str = "bar"


ctx = MyContext()


def test_functional():
    ctx = Context()
    ctx._declare("foo", str, default="bar")

    assert ctx.foo == "bar"


def test_declarative():
    assert ctx.foo == "bar"


def test_get():
    with ctx._let(foo="hello"):
        assert ctx["foo"] == "hello"
        assert ctx.foo == "hello"
        assert ctx._get("foo") == "hello"


def test_let():
    assert ctx.foo == "bar"

    with let(ctx, foo="hello"):
        assert ctx.foo == "hello"

    assert ctx.foo == "bar"

    with ctx._let(foo="hello"):
        assert ctx.foo == "hello"

    assert ctx.foo == "bar"


def test_inject():
    @ctx._inject("a")
    def x(a, b):
        return a, b

    with ctx._let(a=42):
        assert x(b=0) == (42, 0)
        assert x(1, 2) == (1, 2)

    @ctx._inject("b")
    def x(a, b):
        return a, b

    with ctx._let(b=42):
        assert x(0) == (0, 42)
        assert x(1, 2) == (1, 2)
