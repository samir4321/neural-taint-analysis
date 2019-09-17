# chained_flow_example.py
import simple_flow
from chained_simple_flow import chain_saliencies

if __name__ == "__main__":
    simple_flows = [simple_flow.SimpleFlow1(), simple_flow.SimpleFlow2()]
    chain_saliencies(simple_flows)


