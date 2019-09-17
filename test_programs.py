# test_programs.py

import unittest
import programs
import json



class TestPrograms(unittest.TestCase):

    def test_db_query(self):
        query_params = """
            {"name": "fooPerson", "age": 32}
        """
        resp = programs.db_query(query_params)
        expected_substring = "Welcome fooPerson. You qualify for our special deal expiring on"
        assert(expected_substring in resp)

    def test_first_byte_dependent(self):
        pass

    def test_first_byte_value_dependent(self):
        pass

    def test_key_swap(self):
        query_params = """
                    {"name": "fooPerson", "age": 32}
                """
        exp_sub_d = {'person': 'fooPerson', 'howOld': 32}
        d = json.loads((programs.key_swap(query_params)))
        for k, v in exp_sub_d.items():
            assert (k in d)
            assert (d[k] == v)

    def test_magic_keys(self):
        assert "Access denied" in programs.magic_keys("foo")
        assert "Access granted" in programs.magic_keys("magicKey1")


