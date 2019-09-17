from torch import __init__

import torch



def neural_taint_model(D_in, D_out, output_activation=torch.nn.ReLU()):

    #H = int(D_in / 2)
    H = 5

    if output_activation is None:
        return torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H),
            torch.nn.ReLU(),
        torch.nn.Linear(H, D_out)
    )
    return torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
        output_activation
    )



def train_model(model, x, z, learning_rate=1e-3, batch_size=100, nepochs=10**4, print_freq=500, save_freq=None,
                save_path=None):

    loss_fn = torch.nn.MSELoss(reduction='sum')
    learning_rate = learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("Training ...")

    N = x.shape[0]
    for t in range(nepochs):
        rperm = torch.randperm(N)
        x_perm = x[rperm]
        z_perm = z[rperm]
        for k in range(int(N / batch_size)):
            i = k * batch_size
            xb = x_perm[i:i+batch_size]
            zb = z_perm[i:i+batch_size]
            z_pred = model(xb)
            loss = loss_fn(z_pred, zb)
            if k == 0 and t % print_freq == 0:
                print(f'epoch: {t}, epoch loss: {loss.item()}')
            if k == 0 and save_freq is not None and save_path is not None and t % save_freq == 0:
                print('saving model')
                torch.save(model.state_dict(), save_path)
            model.zero_grad()
            loss.backward()
            optimizer.step()

    return model



