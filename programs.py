# programs.py
import json
import time
import random
import string

MAGIC_KEYS = ["magicKey1", "magicKey9n11!", "voodoo1234c1m"]



def first_byte_dependent(input):

    # a sample program whose output in fact depends only on the first
    # byte in the input array (copied directly from the paper)
    # ----------------------------------------------------
    # 1 x = input();
    # 2 a = x[0];
    # 3 b = x[1];
    # 4 c = a*a + b;
    # 5 z = c - b;
    # 6 print z

    x = input
    a = x[0]
    b = x[1]
    c = a * a + b
    z = c - b
    return z


def first_byte_value_dependent(input):

    # a program that returns 1 if the first byte is 7 else returns 0
    # this program depends both only on the first byte and the value of the
    # first byte

    x = input
    if x[0] == 7:
        return 1
    return 0



def key_swap(query_params):
    # a program json -> json where values may be passed from input to output
    # into new keys. the key mapping, if any, is not known.
    try:
        d = json.loads(query_params.strip())
        if ("age" in d and isinstance(d["age"], int) and d["age"] > 0) \
                and ("name" in d and isinstance(d["name"], str) and len(
            d["name"]) > 0):
            return json.dumps({"person": d["name"], "howOld": d["age"], "somethingElse": random.randint(0, 10)})
    except:
        return "Error"



def db_query(query_params):
    """
    :param query_params: a json str
    :return:
    """
    # a program that receives a json containing params
    # for a database query. the program checks the validity of
    # the params as correct values for the required keys before
    # passing making the database query call.
    special_names = ["fooPerson", "barKid"]
    try:
        d = json.loads(query_params.strip())
        if ("age" in d and isinstance(d["age"], int) and d["age"] > 0) \
            and ("name" in d and isinstance(d["name"], str) and len(d["name"]) > 0):
                # q = f"""
                #     SELECT
                #         *
                #     FROM
                #         fooTables
                #     WHERE
                #         name = {name} AND age = {age}"""
                name = d["name"]
                if name in special_names:
                    return f"Welcome {name}. You qualify for our special deal expiring on {time.ctime()} "
                return f"Welcome {name}. You're not special but you can still use our services."
        return "Incorrect payload"

    except Exception as ex:
        print(ex)
        return "Error"


def inverted_magic_keys(str):
    """
    program "<key> <key2: foo or bar> <random> |---> 0 or 1

    1 only of key in MAGIC_KEYS and key2 is foo
    """
    key, key2, random_str = str.split(" ")
    if key in MAGIC_KEYS and key2 == "foo":
        return 1
    return 0


def simple_flow0(x):
    # no random information
    input = f"bar {x}"
    #output = "foo" if x > 5 else "bar"
    output = f"{x}o!d bar"
    return input, output


def simple_flow1(x):
    input = f"foo {x} {random_string_generator(5)}"
    output = f"{x}  bar"
    return input, output


def random_string_generator(bytes_length):
    return ''.join([random.choice(string.ascii_letters) for _ in
             range(0, bytes_length)])


def magic_keys(a):
    # this problem may be hard because of high dimensionality input/output => expensive to compute the jacobian
    """
    A program f: A -> B, where A is an infinite discrete set
    whose values are observed rather than known beforehand.
    Case 1: f(a) = b0 for virtually all a in A, except for a few
    special values a_sp1, a_sp2, etc.

    this behavior can be detected by monitoring hashes of B values and
    recording which A values do not hash to hash(b0).

    Now suppose a modified, more realistic program f: A -> B,
    with set A as before but B also infinite discrete, and
    f(a) = random for virtually all 'a' values in A.  But for a
    small subset of A, f(a) = (random, g(a)), that is the output both has
    random information and deterministic information from a. In this case
    there is information flow but checking hashes of the entire output will
    not work because of the random data in the output.

    the neural network model should figure out the presence of information flow
    where hashing fails here.

    """
    if a in MAGIC_KEYS:
        return f"""
        Access granted for {a}. 
        Token: {random.randint(0, 10**9)}
        {''.join([random.choice(string.ascii_letters + string.digits) for n in range(0, 256)])}
        """
    else:
        return f"""
        Access denied.
        {''.join([random.choice(string.ascii_letters + string.digits) for n in range(random.randint(0, 256))])}
        """







