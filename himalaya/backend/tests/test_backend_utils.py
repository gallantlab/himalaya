import pytest

from himalaya.backend import set_backend
from himalaya.backend import get_backend
from himalaya.backend import ALL_BACKENDS
from himalaya.backend import force_cpu_backend
from himalaya.backend._utils import MATCHING_CPU_BACKEND


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_set_backend_correct(backend):
    # test the change of backend
    module = set_backend(backend)
    assert module.__name__.split('.')[-1] == backend

    # test idempotence
    module = set_backend(set_backend(backend))
    assert module.__name__.split('.')[-1] == backend

    # test set and get
    module = set_backend(get_backend())
    assert module.__name__.split('.')[-1] == backend

    assert set_backend(backend)


def test_set_backend_incorrect():
    for backend in ["wrong", ["numpy"], True, None, 10]:
        with pytest.raises(ValueError):
            set_backend(backend)
        with pytest.raises(ValueError):
            set_backend(backend, on_error="raise")
        with pytest.warns(Warning):
            set_backend(backend, on_error="warn")
        with pytest.raises(ValueError):
            set_backend(backend, on_error="foo")


class ToyEstimator():
    def __init__(self, force_cpu):
        self.force_cpu = force_cpu

    @force_cpu_backend
    def get_backend_wrapped(self):
        return get_backend()


@pytest.mark.parametrize('backend', ALL_BACKENDS)
def test_force_cpu_backend(backend):
    backend = set_backend(backend)

    est = ToyEstimator(force_cpu=True)
    assert est.get_backend_wrapped().name == MATCHING_CPU_BACKEND[backend.name]

    est = ToyEstimator(force_cpu=False)
    assert est.get_backend_wrapped().name == backend.name
