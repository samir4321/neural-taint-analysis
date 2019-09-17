
import unittest

import utils


class TestUtils(unittest.TestCase):

    def setUp(self):
        self.n1, self.b1 = 23, b'\x17'
        self.n2, self.b2 = 5952582390, b'\x01b\xcd2\xf6'

    def test_str_to_bytes(self):
        s = "foo"
        enc_s = utils.str_to_bytes(s)
        assert(isinstance(enc_s, bytes))

    def test_bytes_to_num(self):
        assert utils.byte_to_num(self.b1) == self.n1
        assert utils.byte_to_num(self.b2) == self.n2

    def test_num_to_bytes(self):
        assert utils.num_to_byte(self.n1) == self.b1
        assert utils.num_to_byte(self.n2) == self.b2

    def test_normalize(self):
        assert (all(utils.denormalize(utils.normalize(i)) == i for i in range(0, 256)))

