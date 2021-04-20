import abc


class S7Main(abc.ABC):
    @property
    @abc.abstractproperty
    def X(self):
        raise NotImplementedError()


class S7SideCars(abc.ABC):
    @property
    @abc.abstractproperty
    def f(self):
        raise NotImplementedError()

    @property
    @abc.abstractproperty
    def o(self):
        raise NotImplementedError()


class S7(S7Main, S7SideCars):
    pass
