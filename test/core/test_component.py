# Standard library imports
import random
from textwrap import dedent
from typing import Dict, List

# First-party imports
from gluonts.core.component import validated
from gluonts.core.serde import dump_code, dump_json, load_code, load_json


class Complex:
    @validated()
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

        assert type(self.x) == float
        assert type(self.y) == float

    def __eq__(self, that):
        return self.x == that.x and self.y == that.y

    def __hash__(self):
        return hash((self.x, self.y))


class Foo:
    @validated()
    def __init__(self, a: int, c: Complex, **kwargs) -> None:
        self.a = a
        self.b = kwargs['b']
        self.c = c

        assert type(self.a) == int
        assert type(self.b) == float
        assert type(self.c) == Complex

    def __eq__(self, that):
        return self.a == that.a and self.b == that.b and self.c == that.c

    def __hash__(self):
        return hash((self.a, self.b, self.c))


class Baz(Foo):
    @validated()
    def __init__(self, a: int, b: float, c: Complex, d: int) -> None:
        super().__init__(a=a, c=c, b=b)
        self.d = d


class Bar:
    @validated()
    def __init__(
        self,
        x_list: List[Foo],
        x_dict: Dict[int, Foo],
        input_fields: List[Foo],
    ) -> None:
        self.x_list = x_list
        self.x_dict = x_dict
        self.input_fields = input_fields


# define test.test_components.X as alias of X within the scope of this
# file without modifying test/__init__.py
# noinspection PyPep8Naming
class test:
    class test_components:
        Complex = Complex
        Bar = Bar
        Foo = Foo


# noinspection PyTypeChecker
def test_component_ctor():
    random.seed(5_432_671_244)

    A = 100
    B = 200
    C = 300

    x_list = [
        Foo(
            str(random.randint(0, A)),
            Complex(x=random.uniform(0, C), y=str(random.uniform(0, C))),
            b=random.uniform(0, B),
        )
        for i in range(4)
    ]
    fields = [
        Foo(
            a=str(random.randint(0, A)),
            b=random.uniform(0, B),
            c=Complex(x=str(random.uniform(0, C)), y=random.uniform(0, C)),
        )
        for i in range(5)
    ]
    x_dict = {
        i: Foo(
            b=random.uniform(0, B),
            a=str(random.randint(0, A)),
            c=Complex(
                x=str(random.uniform(0, C)), y=str(random.uniform(0, C))
            ),
        )
        for i in range(6)
    }

    bar01 = Bar(x_list, input_fields=fields, x_dict=x_dict)
    bar02 = load_code(dump_code(bar01))
    bar03 = load_json(dump_json(bar02))

    def compare_tpes(x, y, z, tpe):
        assert tpe == type(x) == type(y) == type(z)

    def compare_vals(x, y, z):
        assert x == y == z

    compare_tpes(bar02.x_list, bar02.x_list, bar03.x_list, tpe=list)
    compare_tpes(bar02.x_dict, bar02.x_dict, bar03.x_dict, tpe=dict)
    compare_tpes(
        bar02.input_fields, bar02.input_fields, bar03.input_fields, tpe=list
    )

    compare_vals(len(bar02.x_list), len(bar02.x_list), len(bar03.x_list))
    compare_vals(len(bar02.x_dict), len(bar02.x_dict), len(bar03.x_dict))
    compare_vals(
        len(bar02.input_fields),
        len(bar02.input_fields),
        len(bar03.input_fields),
    )

    compare_vals(bar02.x_list, bar02.x_list, bar03.x_list)
    compare_vals(bar02.x_dict, bar02.x_dict, bar03.x_dict)
    compare_vals(bar02.input_fields, bar02.input_fields, bar03.input_fields)

    baz01 = Baz(a="0", b="9", c=Complex(x="1", y="2"), d="42")
    baz02 = load_json(dump_json(baz01))

    assert type(baz01) == type(baz02)
    assert baz01 == baz02


def test_dynamic_loading():
    code = dedent(
        '''
        dict(
           trainer=gluonts.trainer.Trainer(
               ctx="cpu(0)",
               epochs=5,
               learning_rate=0.001,
               clip_gradient=10.0,
               weight_decay=1e-08,
               patience=5,
               batch_size=32,
               num_batches_per_epoch=10,
               hybridize=False,
           ),
           num_hidden_dimensions=[3],
           context_length=5,
           prediction_length=2,
           freq="1H",
           distr_output=gluonts.distribution.StudentTOutput(),
           batch_normalization=False,
           mean_scaling=True
        )
        '''
    )

    load_code(code)


def test_to_code():
    c1 = Complex(x=0.0, y=0.0)
    c2 = Complex(y=0.0, x=0.0)

    assert repr(c1) == repr(c2)
