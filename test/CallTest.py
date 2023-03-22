class Person:
    def __call__(self, name):
        print("__call__" + "Hello, " + name)

    def hello(self, name):
        print("hello," + name)


person = Person()
person("midori")
person.hello("ares")

# __call__Hello, midori
# hello,ares

