
def eval_lats(pred, target, res=1000):
    errs = np.zeros((2, res))
    for i in range(2):
        for i_s, s in enumerate(np.linspace(0, 2 * np.pi, res)):
            newpred = 2 * (0.5 - i) * pred + s
            errs[i, i_s] = np.mean(np.arccos(np.cos(newpred - target)))

    i, s = np.unravel_index(errs.argmin(), errs.shape)

    return np.amin(errs), 2 * (0.5 - i), np.linspace(0, 2 * np.pi, res)[s]


### Run experiments on Peyrache data

# Load modules
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from scipy.stats import pearsonr
import pickle
import os
import sys

sys.path.append(os.path.join(os.getcwd(), 'NeuralLVM'))
from NeuralLVM.training import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print('Running on', device)


def run_peyrache(
        model_type='VAE', #{'VAE'}
        feature_type='bump', #{'bump', 'shared/shared_flex', 'separate/separate_flex'}
        vae_do_inference=True #{True, False}
):
    num_neuron_train = 23
    num_neuron_test = 3
    num_ensemble = 1
    latent_dim = 1
    global_seed = 42
    num_rep=20

    file_name = os.path.join(os.getcwd(), 'Mouse28_train_test_data.pkl')
    with open(file_name, 'rb') as handle:
        mouse_data = pickle.load(handle)

    y_train = mouse_data['y_train'][0]
    z_train = mouse_data['z_train'][0]
    y_test = mouse_data['y_test'][0]
    z_test = mouse_data['z_test'][0]
    rf = mouse_data['rf'][0]
    neurons_train_ind = mouse_data['neurons_train_ind'][0]
    #rf_matr = mouse28_data['rf_matr']

    if model_type == 'VAE':

        stats = {
            'corr_rates': [],
            'llh_train_train': [],
            'kld_loss': [],
            'poisson_loss': [],
            'slowness_loss': [],
            'z_test': [],
            'z_pred': [],
            'rf': [],
            'rf_pred': [],
            'neurons_test_ind': [],
            'y_test': [],
            'y_pred': [],
            'mu': [],
            'logvar': []
        }

        for index in range(num_rep):
            print("Rep. number ", index+1, " out of ", num_rep, "\n")
            t0 = time.time()

            model = LatentVariableModel(
                  num_neuron_train=num_neuron_train * num_ensemble,
                  num_neuron_test=num_neuron_test * num_ensemble,
                  num_hidden=64,
                  num_ensemble=num_ensemble,
                  latent_dim=latent_dim,
                  seed=global_seed + index,
                  tuning_width=10.0,
                  nonlinearity='exp',
                  kernel_size=9,
                  feature_type=feature_type,
                  num_feature_basis=16,
              ).to(device)
            trainer = Trainer(
                  model=model,
                  data_train=y_train,
                  data_test=y_test,
                  neurons_train_ind=neurons_train_ind,
                  mode='full',
                  z_train=None,
                  z_test=None,
                  num_steps=50000,
                  batch_size=128, #1024,128,64
                  seed=global_seed + index,
                  learning_rate=1e-3,#3e-3
              )
            trainer.train()

            data_train = torch.tensor(y_train, dtype=torch.float32, device=device)
            y_, _, z_, mu, logvar = model(data_train[trainer.neurons_train_ind], z=None)

            # LLH of train data/train neuron
            llh_train_train = compute_poisson_loss(
                data_train[trainer.neurons_train_ind],
                y_
            )

            data_test = torch.tensor(y_test, dtype=torch.float32, device=device)
            _, y_, z_, mu, logvar = model(data_test[trainer.neurons_train_ind], z=None)

            if vae_do_inference:
                z_ = inference(model,
                              data_test[trainer.neurons_train_ind],
                              data_test[trainer.neurons_test_ind]
                              )
                _, y_, _, mu, logvar = model(data_test[trainer.neurons_train_ind], z=z_)

            z_ = z_.view(z_test.shape)
            #z_angle = vector2angle(mu)
            poisson_loss = compute_poisson_loss(
                  data_test[trainer.neurons_test_ind],
                  y_
              )
            kld_loss = compute_kld_to_normal(mu, logvar)
            slowness_loss = compute_slowness_loss(mu)
            rate_corr = []
            for j in range(num_neuron_test):
                rate_corr.append(pearsonr((y_[j]).detach().cpu().numpy(),
                                        data_test[trainer.neurons_test_ind][j].detach().cpu().numpy())[0])

            z_corr, _ ,_ = eval_lats(
                  z_test.flatten(),
                  z_.detach().cpu().numpy().flatten()
              )

            stats['corr_rates'].append(np.nanmean(rate_corr))
            stats['llh_train_train'].append(llh_train_train.item())
            stats['kld_loss'].append(kld_loss.item())
            stats['poisson_loss'].append(poisson_loss.item())
            stats['slowness_loss'].append(slowness_loss.item())
            stats['z_test'].append(z_test.flatten())
            stats['z_pred'].append(z_.detach().cpu().numpy().flatten())
            stats['rf'].append(rf)
            stats['rf_pred'].append(model.receptive_fields_test.detach().cpu().numpy())
            stats['neurons_test_ind'].append(trainer.neurons_test_ind)
            stats['y_test'].append(data_test[trainer.neurons_test_ind].detach().cpu().numpy())
            stats['y_pred'].append(y_.detach().cpu().numpy())
            stats['mu'].append(mu.detach().cpu().numpy())
            stats['logvar'].append(logvar.view(z_test.shape).detach().cpu().numpy())
            print("Rep. number ", index+1, ' took', time.time() - t0, '\n')

            file_name = os.path.join(os.getcwd(), 'results', 'fig4_stats_peyrache_%s_%s_doinf_%s.pkl' % (model_type, feature_type, vae_do_inference))
            with open(file_name, 'wb') as handle:
                pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)


run_peyrache('VAE', 'bump', True)
