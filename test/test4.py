class ABC:
    def __init__(self):
        self.a = 123

    @property
    def num(self):
        return self.a

    @num.setter
    def num(self, val):
        self.a = val


x = ABC()
print(x.num, x.a)
x.num = 1
print(x.num, x.a)
print(x.__dict__)