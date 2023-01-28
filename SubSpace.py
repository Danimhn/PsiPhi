class SubSpace:

    def __init__(self):
        self._bases = []

    def add_basis(self, basis):
        self._bases.append(basis)

    def get_basis(self):
        return self._bases