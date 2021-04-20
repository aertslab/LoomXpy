from abc import ABCMeta


class WithInitHook(ABCMeta):
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        instance._initialized = True
        return instance
