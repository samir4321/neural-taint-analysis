
import torch

import model_builder
import programs
import utils
import derivative_ops

import random
import string

import simple_flow

def first_byte_dependent_model():

    N, D_in, D_out = 10**5, 4, 1

    mdl = model_builder.neural_taint_model(D_in, D_out,
                                           output_activation=torch.nn.ReLU())
    x = torch.randn(N, D_in)
    z = utils.map_tensor(x, programs.first_byte_dependent).reshape(N, 1)

    nepochs = 10**4
    batch_size = 100
    mdl = model_builder.train_model(mdl, x, z, batch_size=batch_size,
                                    nepochs=nepochs)

    N_support = 10 ** 4
    x_support = torch.randn(N_support, D_in)
    jac_batch_avgnorm = derivative_ops.mean_saliency_map(mdl, x_support)

    print()
    print()
    print(f'Model derivatives matrix D_ij:\n')
    print('-' * 20)
    utils.matprint(jac_batch_avgnorm.cpu().detach().numpy())

    batch_norm_sum = jac_batch_avgnorm.sum()
    for i, component in enumerate(jac_batch_avgnorm):
        if (batch_norm_sum - component) * 50. < component:
            print()
            print('---->')
            print(f'\tcomponent D_{i}0 has outsized influence ({component}); deduced that the {i}th byte is singularly responsible for the output.')

