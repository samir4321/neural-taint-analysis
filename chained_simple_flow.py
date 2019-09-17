# chained_simple_flow.py

import torch

import model_builder
import utils
import derivative_ops
import simple_flow

import numpy as np


def chain_saliencies(simple_flows):

    print("""
    This example computes the saliency maps for two programs chained together. 
    Input and output data are byte sequences.
    It then computes the chained saliency map, determining the chained information flow.
    """)
    print("-" * 90)
    print()
    print()
    models = [simple_flow_model(sf) for sf in simple_flows]
    N_predict = 1000
    x_preds = []

    csf = simple_flow.ChainedSimpleFlow(simple_flows)
    for i, mdl in enumerate(models):
        sf = simple_flows[i]
        x, _, _, _ = build_data(sf, N_predict)
        x_preds.append(x)
    for i, sf in enumerate(simple_flows):

        print(f"* PROGRAM {i} \n\t{sf.get_program_spec()}")

        mdl = models[i]
        print()
        print(f"TOP SALIENCIES S_ij FOR PROGRAM {i}")
        print("-" * 40)
        highlighted_top_saliencies(sf, mdl, 5)

    print()
    print()
    print(f"Chaining saliencies between {', '.join(['PROGRAM ' + str(i) for i in range(len(simple_flows))])}.")
    chained_program_spec = "name: c1c2c3 => SELECT * FROM db WHERE user = c1c2c3"
    print(f"* CHAINED PROGRAM \n\t{chained_program_spec}")
    mn_saliencies = [derivative_ops.mean_saliency_map(mdl, x_preds[i]) for i, mdl in enumerate(models)]
    rev = mn_saliencies[::-1]
    chained_sm = rev[0]
    for i in range(len(rev) - 1):
        m = rev[i + 1]
        chained_sm = torch.mm(chained_sm, m)

    print()
    print()
    print("TOP CHAINED SALIENCIES S_ij...")
    print("-" * 40)
    highlighted_top_saliencies2(csf, chained_sm, 5)


def simple_flow_model(sf):

    x = sf.get_next_param()
    D_in = len(sf.input(x))
    D_out = len(sf.output(x))
    # build model from simple flow
    mdl = model_builder.neural_taint_model(D_in, D_out,
                                           output_activation=torch.nn.Tanh())
    return mdl


def build_data(sf, N):

    inputs = []
    outputs = []
    for i in range(N):
        x = sf.get_next_param()
        input_str, output_str = sf.input(x), sf.output(x)
        inputs.append(input_str)
        outputs.append(output_str)

    x, y = texts_to_torch_arrs(inputs, outputs)
    return x, y, inputs, outputs


def run_simple_flow_model(sf, train=True):

    N_training = 10**5
    nepochs = 5000
    batch_size = 100
    save_path = f"./models/{sf.get_name()}.pt"

    print_freq = 10
    save_freq = 10
    learning_rate = 1e-4
    top_k = 25

    x, y, _, _ = build_data(sf, N_training)
    mdl = simple_flow_model(sf)

    if train:
        print(f"training model {sf.get_name()} ...")
        mdl = model_builder.train_model(mdl, x, y, learning_rate=learning_rate, batch_size=batch_size,
                                        nepochs=nepochs, print_freq=print_freq, save_freq=save_freq, save_path=save_path)

    show_top_mean_saliencies(sf, mdl, top_k)


def show_top_mean_saliencies(sf, mdl, top_k):


    # show top k mean saliency values and corresponding indices in mean S_ij matrix
    print()
    print()
    print(f"* PROGRAM\n\t{sf.get_program_spec()}")
    print()
    print()
    print("TOP SALIENCIES S_ij FOR PROGRAM")
    print('-' * 50)
    highlighted_top_saliencies(sf, mdl, top_k)


def highlighted_top_saliencies2(sf, mn_saliencies, top_k):
    N = 1000
    x, y, inputs, outputs = build_data(sf, N)
    sample_input, sample_output = inputs[0], outputs[0]
    a = mn_saliencies.data.numpy()
    top_indices = tuple(
        zip(*np.unravel_index(a.argsort(axis=None), dims=a.shape)))[::-1][
                  :top_k]
    res = [tuple(list(ix) + [a[ix]]) for ix in top_indices]
    for i, j, saliency in res:
        print(f"""input byte: {j} output byte: {i} mean saliency: {saliency}
    {highlight_bytes(sample_input, j, j+1)} => {highlight_bytes(sample_output, i, i+1)}
    """)


def highlighted_top_saliencies(sf, mdl, top_k):

    mdl_path = f"./models/{sf.get_name()}.pt"
    mdl.load_state_dict(torch.load(mdl_path))
    mdl.eval()
    N = 1000
    x, y, inputs, outputs = build_data(sf, N)
    sample_input, sample_output = inputs[0], outputs[0]
    top_mn_saliencies = derivative_ops.top_mean_saliencies(mdl, x, top_k=top_k)
    for i, j, saliency in top_mn_saliencies:
        print(f"input byte: {j} output byte: {i} mean saliency: {saliency}")
        print(f"{highlight_bytes(sample_input, j, j+1)} => {highlight_bytes(sample_output, i, i+1)}")



def highlight_bytes(s, i1, i2):
    sub_s = s[i1:i2]
    sub = f'\033[91m{sub_s}\033[0m'
    return "".join([s[:i1], sub, s[i2:]])


def texts_to_torch_arrs(inputs, outputs):

    assert len(inputs) == len(outputs)
    D_in = max(len(e) for e in inputs)
    D_out = max(len(e) for e in outputs)
    N = len(inputs)

    input_arrs = [str_to_arr(s, D_in) for s in inputs]
    output_arrs = [str_to_arr(s, D_out) for s in outputs]

    x = utils.map_tensor(torch.FloatTensor(input_arrs).reshape(N, D_in), utils.normalize)
    y = utils.map_tensor(torch.FloatTensor(output_arrs).reshape(N, D_out), utils.normalize)

    return x, y


def str_to_arr(s, bytes_length):
    arr = [ord(c) for c in s]
    arr += [0] * (bytes_length - len(s))
    return arr

def logic_flow_example():
    pass

def chained_flow_example():
    pass



if __name__ == "__main__":
    pass


