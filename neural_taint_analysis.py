# neural_taint_analysis.py
"""
    based on https://arxiv.org/pdf/1907.03756.pdf
"""
import torch

import model_builder
import programs
import utils
import derivative_ops
import simple_flow

import random
import string
import numpy as np


def chain_saliencies(simple_flows):

    print("CHAINING SALIENCIES S_ij")
    print("-" * 25)
    models = [simple_flow_model(sf) for sf in simple_flows]
    N_predict = 1000
    x_preds = []

    csf = simple_flow.ChainedSimpleFlow(simple_flows)
    for i, mdl in enumerate(models):
        sf = simple_flows[i]
        x, _, _, _ = build_data(sf, N_predict)
        x_preds.append(x)
    for i, sf in enumerate(simple_flows):
        print()
        print(f"* PROGRAM {i} \n\t{sf.get_program_spec()}")
        mdl = models[i]
        print()
        print(f"TOP SALIENCIES S_ij FOR PROGRAM {i}")
        print()
        highlighted_top_saliencies(sf, mdl, 5)

    print()
    print()
    print(f"chaining saliencies between {', '.join(['PROGRAM ' + str(i) for i in range(len(simple_flows))])} ...")
    mn_saliencies = [derivative_ops.mean_saliency_map(mdl, x_preds[i]) for i, mdl in enumerate(models)]
    rev = mn_saliencies[::-1]
    chained_sm = rev[0]
    for i in range(len(rev) - 1):
        m = rev[i + 1]
        chained_sm = torch.mm(chained_sm, m)

    print()
    print()
    print("TOP CHAINED SALIENCIES S_ij...")
    print()
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

    print(f"learning {sf.get_program_spec()}")

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

    x_pred = x[:1000]
    show_top_mean_saliencies(sf, mdl, x_pred, top_k)



def show_top_mean_saliencies(sf, mdl, x_pred, top_k):

    save_path = f"./models/{sf.get_name()}.pt"

    print(f'loading model {sf.get_name()}...')
    mdl.load_state_dict(torch.load(save_path))
    mdl.eval()  # to be run before prediction to
    # "set dropout and batch normalization layers to evaluation mode before running inference. "

    # show top k mean saliency values and corresponding indices in mean S_ij matrix
    print()
    print()
    print("top k mean saliences ...")
    print(f"PROGRAM SPEC: {sf.get_program_spec()}")
    print('-' * 50)

    top_mn_saliencies = derivative_ops.top_mean_saliencies(mdl, x_pred,
                                                           top_k=top_k)
    for i, j, saliency in top_mn_saliencies:
        print(f'output byte: {i} input byte: {j} mean saliency: {saliency}')



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
        print(f"""input byte: {j} output byte: {i} mean saliency: {saliency}
{highlight_bytes(sample_input, j, j+1)} => {highlight_bytes(sample_output, i, i+1)}
""")


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


def magic_keys_model(program, input_bytes_length, output_bytes_length, train=True):

    input_generator = lambda: (''.join([random.choice(string.ascii_letters + string.digits) for _ in range(5, 20)])
                               if random.uniform(0, 1) > 0.1 else random.choice(programs.MAGIC_KEYS))

    N = 10**3
    D_in, D_out = input_bytes_length, output_bytes_length

    inputs = []
    outputs = []
    magic_key_indices = []
    non_magic_key_indices = []
    for i in range(N):
        input_str = input_generator()
        if input_str in programs.MAGIC_KEYS:
            magic_key_indices.append(i)
        else:
            non_magic_key_indices.append(i)
        input_arr = str_to_arr(input_str, input_bytes_length)
        assert len(input_arr) == input_bytes_length
        output_str = program(input_str)
        output_arr = str_to_arr(output_str, output_bytes_length)
        assert len(output_arr) == output_bytes_length
        inputs.append(input_arr)
        outputs.append(output_arr)

    x = utils.map_tensor(torch.FloatTensor(inputs).reshape(N, input_bytes_length), utils.normalize)
    y = utils.map_tensor(torch.FloatTensor(outputs).reshape(N, output_bytes_length), utils.normalize)

    assert (x.shape == (N, input_bytes_length))
    assert (y.shape == (N, output_bytes_length))

    mdl = model_builder.neural_taint_model(D_in, D_out,
                                           output_activation=None)
    nepochs = 10**4
    batch_size = 100
    save_path = "./models/magic_keys.pt"

    if train:
        print("training model ...")
        mdl = model_builder.train_model(mdl, x, y, batch_size=batch_size,
                                        nepochs=nepochs, print_freq=50)
        print('saving model ...')
        torch.save(mdl.state_dict(), save_path)

    print('loading model ...')
    mdl.load_state_dict(torch.load(save_path))
    mdl.eval() # to be run before prediction to
    # "set dropout and batch normalization layers to evaluation mode before running inference. "

    x_magic_keys = x.index_select(0, torch.LongTensor(magic_key_indices))
    x_non_magic_keys = x.index_select(0, torch.LongTensor(non_magic_key_indices))

    # sensitivity = derivative_ops.sensitivity(mdl, x)
    # print('-' * 50)
    # for k in range(N):
    #     print(f'index: {k} sensitivity: {sensitivity[k]} is_magic_key: {k in magic_key_indices}')
    #
    # print('-' * 50)
    #
    # mn_sensitivity_magic = derivative_ops.mean_sensitivity(mdl, x_magic_keys)
    # print(f'mean sensitivity on magic keys: {mn_sensitivity_magic}')
    #
    # mn_sensitivity_non_magic = derivative_ops.mean_sensitivity(mdl, x_non_magic_keys)
    # print(f'mean sensitivity on non-magic keys: {mn_sensitivity_non_magic}')

    #saliency_mp = derivative_ops.saliency_map(mdl, x)
    # top_saliencies = derivative_ops.top_saliencies(mdl, x, 100)
    # for k, i, j, saliency in top_saliencies:
    #     print(f'index: {k} is_magic_key: {k in magic_key_indices} output byte: {i} input byte: {j} saliency: {saliency}')

    top_mn_saliencies = derivative_ops.top_mean_saliencies(mdl, x, 100)
    for i, j, saliency in top_mn_saliencies:
        print(f'output byte: {i} input byte: {j} mean saliency: {saliency}')



if __name__ == "__main__":
    #run_simple_flow_model(simple_flow.SimpleFlow3(), train=False)
    # simple_flows = [simple_flow.SimpleFlow1(), simple_flow.SimpleFlow2()]
    # chain_saliencies(simple_flows)
    #print(highlight_bytes("foo bar", 1, 3))
    run_simple_flow_model(simple_flow.SimpleLogicFlow(), train=False)


