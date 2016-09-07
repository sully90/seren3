class NoParticlesException(Exception):

    def __init__(self, message, fn_name):
        super(NoParticlesException, self).__init__(message)

        self.fn_name = fn_name

class UnknownFieldException(Exception):

    def __init__(self, field):
        super(UnknownFieldException, self).__init__("Unknown field: %s" % field)

        self.field = field

class CatalogueNotFoundException(Exception):

    def __init__(self, message):
        super(CatalogueNotFoundException, self).__init__(message)