from abc import ABC, abstractmethod
import random
import string
import re
import hashlib



class SimpleFlow(ABC):

    @abstractmethod
    def input(self, x):
        pass

    @abstractmethod
    def output(self, x):
        pass

    @abstractmethod
    def get_program_spec(self):
        pass

    @abstractmethod
    def get_next_param(self):
        pass

    @abstractmethod
    def get_name(self):
        pass


class SimpleFlow1:

    def __init__(self):
        self.program_spec = "name: c1c2c3 => person: c1c2c3"
        self.name = "simple_flow1"

    def input(self, x):
        return f"name: {x}"

    def output(self, x):
        return f"person: {x}"

    def get_program_spec(self):
        return self.program_spec

    def get_next_param(self):
        return random_string_generator(3)

    def get_name(self):
        return self.name


class SimpleFlow2:

    def __init__(self):
        self.program_spec = "person: c1c2c3 => SELECT * FROM db WHERE user = c1c2c3"
        self.name = "simple_flow2"

    def input(self, x):
        return f"person: {x}"

    def output(self, x):
        return f"SELECT * FROM db WHERE user = {x}"

    def get_program_spec(self):
        return self.program_spec

    def get_next_param(self):
        return random_string_generator(3)

    def get_name(self):
        return self.name


class SimpleFlow3:

    def __init__(self):
        self.long_string = random_string_generator(1233)
        self.program_spec = "person: c1c2c3 => [if special_char in c1c2c3 'failed' else 'success']"
        self.name = "simple_flow3"

    def input(self, x):
        return f"person: {x}"

    def output(self, x):
        return "failed_" if contains_special_chars(x) else "success"

    def get_program_spec(self):
        return self.program_spec

    def get_next_param(self):
        return random_string_generator(3)

    def get_name(self):
        return self.name


class SimpleHashFlow:
    def __init__(self):
        self.program_spec = "name: c1c2c3 => person: sha1hash"
        self.name = "simplehashflow"

    def input(self, x):
        return f"name: {x}"

    def output(self, x):
        return f"person: {hashlib.sha1(x.encode('utf-8')).hexdigest()}"

    def get_program_spec(self):
        return self.program_spec

    def get_next_param(self):
        return random_string_generator(3)

    def get_name(self):
        return self.name


class SimpleLogicFlow:
    def __init__(self):
        self.program_spec = "age: x name: {random string} => [if x > 1900 && x < 2020 'ok' else 'not ok'] (x is a 4-digit number)"
        self.name = "simplelogicflow"

    def input(self, x):
        return f"age: {x} name: {random_string_generator(4)}"

    def output(self, x):
        return "ok" if x > 1900 and x < 2020 else "not ok"

    def get_program_spec(self):
        return self.program_spec

    def get_next_param(self):
        return random.randint(1000, 3000)

    def get_name(self):
        return self.name


class ChainedSimpleFlow:

    def __init__(self, simple_flows):
        self.simple_flows = simple_flows
        self.name = "chained_flow"

    def input(self, x):
        return self.simple_flows[0].input(x)

    def output(self, x):
        return self.simple_flows[-1].output(x)

    def get_program_spec(self):
        return "<chained program spec>"

    def get_next_param(self):
        return self.simple_flows[0].get_next_param()

    def get_name(self):
        return self.name


def random_string_generator(bytes_length, letters_only=False):
    if letters_only:
        return ''.join(
            [random.choice(string.ascii_letters) for _ in
             range(0, bytes_length)])
    return ''.join([random.choice(string.ascii_letters + string.punctuation) for _ in
             range(0, bytes_length)])


def contains_special_chars(s):
    # this as argument in compile method
    regex = re.compile('[@_!#$%^&*()<>?/\|}{~:]')
    # Pass the string in search
    # method of regex object.
    if (regex.search(s) == None):
        return False
    return True


SimpleFlow.register(SimpleFlow1)
SimpleFlow.register(SimpleFlow2)
SimpleFlow.register(SimpleFlow3)
SimpleFlow.register(ChainedSimpleFlow)

