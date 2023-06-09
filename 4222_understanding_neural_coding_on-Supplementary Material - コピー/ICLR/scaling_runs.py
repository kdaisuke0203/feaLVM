
### Run experiments in scaling

# Load modules
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from scipy.stats import pearsonr
import pickle
import os
import sys
import cmath
import scipy.stats as stats

sys.path.append(os.path.join(os.getcwd(), 'NeuralLVM'))

from NeuralLVM.training import *
from NeuralLVM.data import get_data


device = "cuda" if torch.cuda.is_available() else "cpu"
print('Running on', device)


def run_scaling(
    scaling_type='N', #{'N', 'T'}
    model_type='VAE', #{'VAE'}
    feature_type='bump', #{'bump', 'shared/shared_flex', 'separate/separate_flex'}
    vae_do_inference=True
):
    if scaling_type == 'N':
        num_neuron_train_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        num_train_list = [1000]
    if scaling_type == 'T':
        num_neuron_train_list = [30]
        num_train_list = [75, 100, 150, 200, 300, 400, 500, 750, 1000, 2500]

    num_neuron_test = 30
    latent_dim = 1
    num_ensemble = 1
    num_test = 1000
    global_seed = 42
    num_rep = 20

    # Should prob. parallelize this..
    for num_train in num_train_list:
        for num_neuron_train in num_neuron_train_list:

            if model_type == 'VAE':
                if num_train < 128:
                    train_batch = 64
                else:
                    train_batch = 128

                stats = {
                    'corr_rates': [],
                    'corr_latents': [],
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
                    y_train, z_train, y_test, z_test, rf, neurons_train_ind = get_data(
                        num_neuron_train=num_neuron_train,
                        num_neuron_test=num_neuron_test,
                        len_data_train=num_train,
                        len_data_test=num_test,
                        index=index,
                        global_seed=global_seed
                      )
                    model = LatentVariableModel(
                          num_neuron_train=num_neuron_train * num_ensemble,
                          num_neuron_test=num_neuron_test * num_ensemble,
                          num_hidden=64,
                          num_ensemble=num_ensemble,
                          latent_dim=latent_dim,
                          seed=global_seed + index,
                          tuning_width=10.0,
                          nonlinearity='exp',
                          kernel_size=1,
                          feature_type=feature_type,
                          num_feature_basis=10,
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
                          batch_size=train_batch,#1024,128,64
                          seed=global_seed + index,
                          learning_rate=1e-3,#3e-3
                      )
                    trainer.train()

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
                        rate_corr.append(pearsonr(y_[j].detach().cpu().numpy(),
                                                data_test[trainer.neurons_test_ind][j].detach().cpu().numpy())[0])
                    z_corr = corrcoef(
                          z_test.flatten(),
                          z_.detach().cpu().numpy().flatten(),
                          deg=False,
                          test=False
                      )
                    stats['corr_rates'].append(np.nanmean(rate_corr)) #not in use any more
                    stats['corr_latents'].append(0) # not in use any more
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

                    if scaling_type == 'N':
                        num_iter = num_neuron_train
                    if scaling_type == 'T':
                        num_iter = num_train
                    file_name = os.path.join(os.getcwd(), 'results', 'fig2_stats_peakrate_0.5_%s_%s_inf_%s_%s_%d.pkl' % (model_type, feature_type, vae_do_inference, scaling_type, num_iter))
                    with open(file_name, 'wb') as handle:
                        pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

run_scaling('N', 'VAE', 'bump', True)
