# logic_flow_example.py
from simple_flow import SimpleLogicFlow
from chained_simple_flow import run_simple_flow_model

DESCRIPTION = """
This program returns a boolean based on syntax conditions on its 
byte sequence input.
"""

if __name__ == "__main__":
    print(DESCRIPTION)
    print('-' * 90)
    run_simple_flow_model(SimpleLogicFlow(), train=False)

