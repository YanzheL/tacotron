class TextEncoder(object):
    LANGUAGE = 'unknown'

    @classmethod
    def get_encoder(cls, language):
        for sub_cls in cls.__subclasses__():
            if language.lower() == sub_cls.LANGUAGE.lower():
                return sub_cls
        raise ValueError(
            "Cannot find a implemented encoder class for language <{}>".format(language)
        )

    def __init__(self):
        raise NotImplementedError('Encoders are static by default')

    @staticmethod
    def encode(text):
        raise NotImplementedError('This method should be implemented in subclass')

    @property
    def SYMBOLS_SIZE(self):
        raise NotImplementedError('Implementation class does not provide this property')
