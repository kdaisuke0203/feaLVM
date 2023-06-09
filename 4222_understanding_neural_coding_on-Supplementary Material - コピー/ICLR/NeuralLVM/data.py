import numpy as np
from scipy.spatial import distance
import os
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from scipy import stats
import sys
import time

class data_generation:
    def __init__(
            self,
            num_neuron=50,
            len_data=1000,
            num_ensemble=1,
            dim=1,
            periodicity=True,
            kernel_type='exp',
            kernel_sdev=5.0,
            kernel_scale=50.0,
            bump_placement='random',
            peak_firing=0.5,
            back_firing=0.005,
            tuning_width=1.2,
            snr_scale=1.0,
            poisson_noise=False,
            seed=1337
    ):
        self.len_data = len_data
        self.kernel_type = kernel_type
        self.kernel_sdev = kernel_sdev
        self.kernel_scale = kernel_scale
        self.dim = dim
        self.periodicity = periodicity

        self.num_neuron = num_neuron
        self.num_ensemble = num_ensemble
        self.bump_placement = bump_placement
        self.peak_firing = peak_firing
        self.back_firing = back_firing
        self.tuning_width = tuning_width
        self.snr_scale = snr_scale
        self.poisson_noise = poisson_noise

        self.seed = seed

        self.ensemble_weights = np.zeros((self.num_neuron * self.num_ensemble, self.num_ensemble))
        for i in range(self.num_ensemble):
            self.ensemble_weights[i * self.num_neuron:(i + 1) * self.num_neuron, i] = 1

    def generate_z(self):
        np.random.seed(self.seed)

        if self.kernel_type == 'exp':
            dist = distance.cdist(np.linspace(1, self.len_data,self.len_data).reshape((self.len_data,1)),
                                  np.linspace(1, self.len_data,self.len_data).reshape((self.len_data,1)),
                                  'euclidean')
            K_t = self.kernel_sdev * np.exp(-dist / self.kernel_scale)

        elif self.kernel_type != 'exp':
            # might want different kernels
            print("Not fixed yet")
        z = []
        for i in range(self.dim * self.num_ensemble):
            z.append(np.random.multivariate_normal(np.zeros(self.len_data), K_t))
        z = np.asarray(z).T
        if self.periodicity:
            z = z % (2 * np.pi)

        return z

    def generate_receptive_fields(self, z):
        np.random.seed(self.seed)

        if self.periodicity:
            if self.bump_placement == 'random':
                rf_location = 2 * np.pi * np.random.rand(self.num_neuron * self.num_ensemble, self.dim)
            elif self.bump_placement == 'uniform':
                rf_location = np.tile(np.array([[(i + 0.5) / self.num_neuron * (2 * np.pi)
                                                 for i in range(self.num_neuron)]
                                                for j in range(self.dim)]).T, (self.num_ensemble, 1))
        else:
            min_z = np.min(z, axis=0)
            max_z = np.max(z, axis=0)

            if self.bump_placement == 'random':
                rf_location = np.random.rand(self.num_neuron * self.num_ensemble, self.dim)
                rf_location = min_z + rf_location * (max_z - min_z)
            elif self.bump_placement == 'uniform':
                rf_location = np.tile(np.array([min_z + [(i + 0.5) / self.num_neuron * (max_z - min_z)
                                                         for i in range(self.num_neuron)]
                                                for j in range(self.dim)]).T, (self.num_ensemble, 1))

        return rf_location

    def generate_spikes(self, z, rf_location):
        np.random.seed(self.seed)

        selector = np.stack([np.eye(self.dim) for i in range(self.num_ensemble)], 0)
        selector = self.ensemble_weights[..., None, None] * selector[None]
        selector = np.concatenate(np.split(selector, self.num_ensemble, axis=1), axis=3).reshape(
            self.num_neuron * self.num_ensemble, self.dim, self.num_ensemble * self.dim)
        selected = np.matmul(selector, z.T)

        dist = (rf_location[..., None] - selected)
        if self.periodicity:
            dist = np.abs(dist)
            dist[dist > np.pi] = 2 * np.pi - dist[dist > np.pi]
        dist = dist**2
        dist = np.sum(dist, axis=1)

        response = np.log(self.back_firing) + (np.log(self.peak_firing / self.back_firing)) * np.exp(-dist / (2 * self.tuning_width))
        response = np.exp(response) * self.snr_scale
        if self.poisson_noise:
            response = np.random.poisson(response)
        response = response / self.snr_scale

        return response

def get_data(
    num_neuron_train,
    num_neuron_test,
    len_data_train,
    len_data_test,
    index,
    global_seed,
    num_ensemble=1,
    dim=1,
    data_type='simulated', #{'simulated', 'peyrache', 'gardner'}
    start_t=0,
    tuning_selection=0,
    downsample_factor=1,
    data_dir=None
):
    if data_type == 'simulated':
        num_neuron = num_neuron_train + num_neuron_test
        data = data_generation(
            len_data=len_data_train + len_data_test,
            dim=1,
            num_neuron=num_neuron,
            poisson_noise=True,
            bump_placement='random',
            seed=global_seed + index
        )

        print("Generating latents\n")
        z = data.generate_z()
        z_train = z[:len_data_train, :]
        z_test = z[len_data_train:, :]

        print("Generating receptive fields\n")
        rf = data.generate_receptive_fields(z)

        print("Generating spikes")
        y_train = data.generate_spikes(z_train, rf)
        data.poisson_noise = False
        y_test = data.generate_spikes(z_test, rf)

    elif data_type == 'peyrache':
        # num_neuron={18,21,60,26,30,59} for case{1,2,3,4,5,6}
        z, y, num_neuron, rfs, rf_matr = prepare_peyrache(
            len_data_train + len_data_test,
            start_t=0,
            downsample_factor=downsample_factor,
            tuning_selection=tuning_selection
        )
        z_train = z[:len_data_train]
        z_test = z[len_data_train:]
        y_train = y[:, :len_data_train]
        y_test = y[:, len_data_train:(len_data_train + len_data_test)]

        rf = {'rf' : rfs, 'rf_matr': rf_matr}
        xs=0
        ys=0

    elif data_type == 'gardner':
        # num_neuron=387 {mod1:93, mod2:149, mod3:145}
        _, y, num_neuron, _, num_mod1, num_mod2, num_mod3, y1, y2, y3, xs, ys= prepare_gardner(
            len_data_train + len_data_test,
            downsample_factor=1,
            even_ensemble_number=False,
            k_mod1=90, #51 mod1, 90 mod2
            k_mod2=149,
            k_mod3=145
        )

        #Use only 2nd and 3rd module
        y = np.concatenate((y2, y3), axis = 0)
        num_neuron = num_mod2 + num_mod3

        y_train = y[:, :len_data_train]
        y_test = y[:, len_data_train:(len_data_train + len_data_test)]
        z_train = 0
        z_test = 0
        rf = 0

    # select training and test neurons
    np.random.seed(global_seed + index)
    neurons_train_ind = np.zeros(num_neuron, dtype=bool)
    if data_type == 'gardner':
        if num_neuron_test == 1:
          ind = np.random.choice(num_neuron, num_neuron_train, replace=False)
        elif num_ensemble == 2:
          ind_1 = np.random.choice(num_mod2, num_neuron_train // num_ensemble, replace=False)
          ind_2 = np.random.choice(num_mod3, num_neuron_train // num_ensemble, replace=False)
          ind = np.concatenate((ind_1, ind_2 + num_mod2))
        elif num_ensemble == 3:
          ind_1 = np.random.choice(num_mod1, num_neuron_train // num_ensemble, replace=False)
          ind_2 = np.random.choice(num_mod2, num_neuron_train // num_ensemble, replace=False)
          ind_3 = np.random.choice(num_mod3, num_neuron_train // num_ensemble, replace=False)
          ind = np.concatenate((ind_1, ind_2 + num_mod1, ind_3 + num_mod1 + num_mod2))
    else:
        ind = np.random.choice(num_neuron, num_neuron_train, replace=False)
    neurons_train_ind[ind] = True

    #return y_train, z_train, y_test, z_test, rf, neurons_train_ind, rf_matr, y, xs, ys
    return y_train, z_train, y_test, z_test, rf, neurons_train_ind

def prepare_peyrache(
    len_data,
    data_dir=os.getcwd(),
    mat_name = 'Mouse28-140313_simple_awakedata.mat', #{'Mouse28-140313_simple_awakedata.mat','Mouse12-120806_simple_awakedata.mat'}
    start_t=0,
    downsample_factor=4, # 1=25ms bins, 2=50ms bins, 4=100ms bins, 20=500ms bins...
    tuning_selection=4 #{1:tuned and strong, 2:tuned and also weak, 3:all active, 4,5,6: same for mouse28}
):
    print("Loading file\n")
    file_name = os.path.join(data_dir, mat_name)
    mat = loadmat(file_name)
    headangle = np.ravel(np.array(mat['headangle'])) # Observed head direction
    cellspikes = np.array(mat['cellspikes']) # Observed spike time points
    cellnames = np.array(mat['cellnames']) # Alphanumeric identifiers for cells
    trackingtimes = np.ravel(np.array(mat['trackingtimes'])) # Time stamps of head direction observations
    positions = np.array(mat['position']) # Animal positions

    print("Preparing latent\n")
    z = headangle
    t_max = len(z)
    if start_t + (len_data) * downsample_factor > t_max:
        sys.exit('Combination of start point, downsampling and data length places the end of data outside t_max. Choose lower data length, starting point or downsampling factor.')
    whiches = np.isnan(z)
    z = z[~whiches]
    z = z % (2 * np.pi)

    print('Binning spikes\n')
    starttime = min(trackingtimes)
    tracking_interval = np.mean(trackingtimes[1:] - trackingtimes[:(-1)])
    binsize = tracking_interval
    nbins = len(trackingtimes)
    binnedspikes = np.zeros((len(cellnames), nbins))
    for i in range(len(cellnames)):
        spikes = np.ravel((cellspikes[0])[i])
        for j in range(len(spikes)):
            timebin = int(np.floor((spikes[j] - starttime)/float(binsize)))
            if(timebin > nbins - 1 or timebin < 0):
                continue
            binnedspikes[i,timebin] += 1
    binnedspikes = binnedspikes[:, ~whiches]


    binsize = downsample_factor * tracking_interval
    nbins = (binnedspikes.shape[1]) // downsample_factor
    if downsample_factor != 1:
        #print('Bin size after downsampling: {:.2f}ms'.format(binsize))
        #print('Number of bins for entire interval:', nbins)
        #print('Number of bins when using downsampled binsize:', nbins/downsample_factor)
        print('Downsampling binned spikes\n')
        downsampled_binnedspikes = np.zeros((len(cellnames), nbins))
        for i in range(len(cellnames)):
            for j in range(nbins):
                downsampled_binnedspikes[i, j] = np.sum(binnedspikes[i, downsample_factor * j:downsample_factor * (j + 1)])
        binnedspikes = downsampled_binnedspikes

    if downsample_factor != 1:
        print('Downsampling latent\n')
        downsampled_z = np.zeros(len(z) // downsample_factor)
        for i in range(len(z) // downsample_factor):
            cos_mean = np.mean(np.cos(z[downsample_factor * i:downsample_factor * (i + 1)]))
            sin_mean = np.mean(np.sin(z[downsample_factor * i:downsample_factor * (i + 1)]))
            downsampled_z[i] = np.arctan2(sin_mean, cos_mean) % (2 * np.pi)
        z = downsampled_z

    downsampled_start = start_t // downsample_factor
    z = z[downsampled_start:downsampled_start + (len_data)]
    binnedspikes = binnedspikes[:, downsampled_start:downsampled_start + (len_data)]

    print('Selecting neurons\n')
    ### Mouse12
    if tuning_selection == 1:
        tuned_neurons = [20,21,22,23,24,25,26,27,28,29,31,34,35,36,37,38,39,68] #tuned and quite active
    elif tuning_selection == 2:
        tuned_neurons = [17,18,20,21,22,23,24,25,26,27,28,29,31,32,34,35,36,37,38,39,68] #tuned and somewhat active
    elif tuning_selection == 3:
        tuned_neurons = [0,1,2,3,4,5,6,8,9,10,11,12,13,14,15,17,18,19,20,21,22,23,24,25,
                         26,27,28,29,31,32,33,34,35,36,37,38,39,43,44,45,47,49,50,51,52,
                         53,54,56,58,60,61,62,63,64,66,67,68,69,70,71] # all that are active in some form
    ### Mouse28
    elif tuning_selection == 4:
        tuned_neurons = [7,15,16,17,19,21,22,23,25,26,38,46,47,48,49,51,54,55,
                         57,58,60,61,62,65,66,67] # tuned and quite active
    elif tuning_selection == 5:
        tuned_neurons = [7,15,16,17,19,20,21,22,23,25,26,29,38,44,46,47,48,49,
                         51,53,54,55,57,58,60,61,62,65,66,67] # tuned and somewhat active
    elif tuning_selection == 6:
        tuned_neurons = [0,1,3,4,5,6,7,8,9,10,11,14,15,16,17,18,19,20,21,22,23,24,25,26,
                         27,29,31,32,33,34,36,37,38,39,40,41,43,44,46,47,48,49,50,51,52,
                         53,54,55,57,58,60,61,62,63,64,65,66,67,69] # all that are active in some form

    good_neurons = np.zeros(len(cellnames)) > 1
    good_neurons[tuned_neurons] = True
    binnedspikes = binnedspikes[good_neurons, :]

    print('Locating receptive field peak')
    bins = np.linspace(-0.000001, 2. * np.pi + 0.0000001, num = 30 + 1)
    x_grid = 0.5 * (bins[:(-1)] + bins[1:])
    observed_mean_spikes_in_bins = np.zeros((len(tuned_neurons), 30))
    for i in range(len(tuned_neurons)):
        for x in range(30):
            timesinbin = (z > bins[x]) * (z < bins[x + 1])
            if (np.sum(timesinbin) > 0):
                observed_mean_spikes_in_bins[i,x] = np.mean(binnedspikes[i, timesinbin])
    rfs = x_grid[np.argmax(observed_mean_spikes_in_bins, axis = 1)]

    return z, binnedspikes, len(tuned_neurons), rfs, observed_mean_spikes_in_bins

def prepare_gardner(
    len_data,
    data_dir=os.getcwd(),
    start_t=0,
    speed_threshold=2.5, #2.5, 5.0
    downsample_factor=1, #1=10ms bins, 2=20ms bins, 10=100ms bins, 20=200ms bins...
    tuning_selection=1, #{1:only grid cell tuned}
    session = 'open_field', #{'open_field', 'all_sessions'}
    even_ensemble_number=True,
    k_mod1=51,
    k_mod2=51,
    k_mod3=51
):
    if session == 'open_field':
        min_time_1 = 7457
        max_time_1 = 14778
        min_time_2 = 14890
        max_time_2 = 16045

    elif session == 'all_sessions':
        #open field
        min_time_1 = 7457
        max_time_1 = 14778
        min_time_2 = 14890
        max_time_2 = 16045
        #maze 1
        min_time_3 = 16925
        max_time_3 = 18026
        min_time_4 = 18183
        max_time_4 = 20704
        #maze 2
        min_time_5 = 20895
        max_time_5 = 21640

    print("Loading file\n")
    file = np.load(os.path.join(os.getcwd(),'rat_r_day1_grid_modules_1_2_3.npz'), allow_pickle=True)
    conj = np.load(os.path.join(os.getcwd(), 'is_conjunctive_all.npz'), allow_pickle=True)
    trackingtimes = file['t']
    x_pos = file['x']
    y_pos = file['y']
    spikes_mod1 = file['spikes_mod1']
    spikes_mod2 = file['spikes_mod2']
    spikes_mod3 = file['spikes_mod3']


    print("Identify correct intervals\n")
    # identify where either break in recording or rat is still
    if session == 'open_field':
        good_times = np.where(((trackingtimes >= min_time_1) & (trackingtimes <= max_time_1)) |
                            ((trackingtimes >= min_time_2) & (trackingtimes <= max_time_2)))[0]

    elif session == 'all_sessions':
        good_times = np.where(((trackingtimes >= min_time_1) & (trackingtimes <= max_time_1)) |
                            ((trackingtimes >= min_time_2) & (trackingtimes <= max_time_2)) |
                            ((trackingtimes >= min_time_3) & (trackingtimes <= max_time_3)) |
                            ((trackingtimes >= min_time_4) & (trackingtimes <= max_time_4)) |
                            ((trackingtimes >= min_time_5) & (trackingtimes <= max_time_5)))[0]

    xxs = gaussian_filter1d(x_pos - np.min(x_pos), sigma = 100)
    yys = gaussian_filter1d(y_pos - np.min(y_pos), sigma = 100)
    dx = (xxs[1:] - xxs[:-1]) * 100
    dy = (yys[1:] - yys[:-1]) * 100
    speed = np.divide(np.sqrt(dx**2 + dy**2), trackingtimes[1:] - trackingtimes[:-1])
    speed = np.concatenate(([speed[0]], speed))
    good_speed = np.where(speed > speed_threshold)[0]
    good_spikes = np.intersect1d(good_times, good_speed)

    print('Binning spikes\n')
    starttime = min(trackingtimes)
    tracking_interval = np.mean(trackingtimes[1:] - trackingtimes[:(-1)])
    binsize = tracking_interval
    nbins = len(trackingtimes)

    binnedspikes_mod1 = np.zeros((len(spikes_mod1.item().items()), nbins))
    binnedspikes_mod2 = np.zeros((len(spikes_mod2.item().items()), nbins))
    binnedspikes_mod3 = np.zeros((len(spikes_mod3.item().items()), nbins))

    t0 = time.time()
    for key in spikes_mod1.item():
        spikes = spikes_mod1.item()[key]
        for j in range(len(spikes)):
            timebin = int(np.floor((spikes[j] - starttime)/float(binsize)))
            if(timebin > nbins - 1 or timebin < 0):
                continue
            binnedspikes_mod1[key, timebin] += 1
    print('Binning module 1 took ', time.time() - t0, ' seconds\n')

    t0 = time.time()
    for key in spikes_mod2.item():
        spikes = spikes_mod2.item()[key]
        for j in range(len(spikes)):
            timebin = int(np.floor((spikes[j] - starttime)/float(binsize)))
            if(timebin > nbins - 1 or timebin < 0):
                continue
            binnedspikes_mod2[key, timebin] += 1
    print('Binning module 2 took ', time.time() - t0, ' seconds\n')

    t0 = time.time()
    for key in spikes_mod3.item():
        spikes = spikes_mod3.item()[key]
        for j in range(len(spikes)):
            timebin = int(np.floor((spikes[j] - starttime)/float(binsize)))
            if(timebin > nbins - 1 or timebin < 0):
                continue
            binnedspikes_mod3[key, timebin] += 1
    print('Binning module 3 took ', time.time() - t0, ' seconds\n')

    print('Selecting right intervals\n')
    trackingtimes = trackingtimes[good_spikes]
    x_pos = x_pos[good_spikes]
    y_pos = y_pos[good_spikes]
    binnedspikes_mod1 = binnedspikes_mod1[:,good_spikes]
    binnedspikes_mod2 = binnedspikes_mod2[:,good_spikes]
    binnedspikes_mod3 = binnedspikes_mod3[:,good_spikes]

    nbins = len(trackingtimes)
    binsize = downsample_factor * tracking_interval
    nbins = nbins // downsample_factor
    if downsample_factor != 1:
        #print('Bin size after downsampling: {:.2f}ms'.format(binsize))
        #print('Number of bins for entire interval:', nbins)
        #print('Number of bins when using downsampled binsize:', nbins/downsample_factor)
        print('Downsampling binned spikes for module 1\n')
        downsampled_binnedspikes_mod1 = np.zeros((len(spikes_mod1.item().items()), nbins))
        for i in range((len(spikes_mod1.item().items()))):
            for j in range(nbins):
                downsampled_binnedspikes_mod1[i, j] = np.sum(binnedspikes_mod1[i, downsample_factor * j:downsample_factor * (j + 1)])
        binnedspikes_mod1 = downsampled_binnedspikes_mod1

        print('Downsampling binned spikes for module 2\n')
        downsampled_binnedspikes_mod2 = np.zeros((len(spikes_mod2.item().items()), nbins))
        for i in range((len(spikes_mod2.item().items()))):
            for j in range(nbins):
                downsampled_binnedspikes_mod2[i, j] = np.sum(binnedspikes_mod2[i, downsample_factor * j:downsample_factor * (j + 1)])
        binnedspikes_mod2 = downsampled_binnedspikes_mod2

        print('Downsampling binned spikes for module 3\n')
        downsampled_binnedspikes_mod3 = np.zeros((len(spikes_mod3.item().items()), nbins))
        for i in range((len(spikes_mod3.item().items()))):
            for j in range(nbins):
                downsampled_binnedspikes_mod3[i, j] = np.sum(binnedspikes_mod3[i, downsample_factor * j:downsample_factor * (j + 1)])
        binnedspikes_mod3 = downsampled_binnedspikes_mod3

        x_pos_sub = np.zeros(nbins)
        y_pos_sub = np.zeros(nbins)
        for i in range(nbins):
                x_pos_sub[i] = np.mean(x_pos[downsample_factor * i:downsample_factor * (i + 1)])
                y_pos_sub[i] = np.mean(y_pos[downsample_factor * i:downsample_factor * (i + 1)])

        x_pos = x_pos_sub
        y_pos = y_pos_sub

    print('Remove conjunctive neurons\n')
    # remove neurons also tuned to hd
    mod1_select = ~conj['is_conj_R1_day1']
    mod2_select = ~conj['is_conj_R2_day1']
    mod3_select = ~conj['is_conj_R3_day1']

    binnedspikes_mod1 = binnedspikes_mod1[mod1_select]
    binnedspikes_mod2 = binnedspikes_mod2[mod2_select]
    binnedspikes_mod3 = binnedspikes_mod3[mod3_select]


    print('Selecting neurons\n')
    mod1_spinfo=np.load(os.path.join(os.getcwd(),'roger_mod1_box_si_inds.npz'), allow_pickle=True)['ind']
    mod2_spinfo=np.load(os.path.join(os.getcwd(),'roger_mod3_box_si_inds.npz'), allow_pickle=True)['ind']
    mod3_spinfo=np.load(os.path.join(os.getcwd(),'roger_mod4_box_si_inds.npz'), allow_pickle=True)['ind']

    # select k most informate neurons in each module
    sel_mod1 = np.flip(mod1_spinfo)[:k_mod1]
    sel_mod2 = np.flip(mod2_spinfo)[:k_mod2]
    sel_mod3 = np.flip(mod3_spinfo)[:k_mod3]

    if even_ensemble_number:
        k = np.min([k_mod1, k_mod2, k_mod3])
        sel_mod1 = sel_mod1[:k]
        sel_mod2 = sel_mod2[:k]
        sel_mod3 = sel_mod3[:k]
        binnedspikes_mod1 = binnedspikes_mod1[sel_mod1]
        binnedspikes_mod2 = binnedspikes_mod2[sel_mod2]
        binnedspikes_mod3 = binnedspikes_mod3[sel_mod3]
    else:
        binnedspikes_mod1 = binnedspikes_mod1[sel_mod1]
        binnedspikes_mod2 = binnedspikes_mod2[sel_mod2]
        binnedspikes_mod3 = binnedspikes_mod3[sel_mod3]

    print('Num neuron mod1: ',len(mod1_select), 'Only grid tuned: ', np.sum(mod1_select))
    print('Num neuron mod2: ',len(mod2_select), 'Only grid tuned: ', np.sum(mod2_select))
    print('Num neuron mod3: ',len(mod3_select), 'Only grid tuned: ', np.sum(mod3_select), ' \n')

    print('Num selected mod1: ', binnedspikes_mod1.shape[0])
    print('Num selected mod2: ', binnedspikes_mod2.shape[0])
    print('Num selected mod3: ', binnedspikes_mod3.shape[0], ' \n')
    binnedspikes = np.concatenate((binnedspikes_mod1, binnedspikes_mod2, binnedspikes_mod3), axis = 0)
    print('Done\n')

    return 0, binnedspikes, binnedspikes.shape[0], 0, binnedspikes_mod1.shape[0], binnedspikes_mod2.shape[0], binnedspikes_mod3.shape[0], binnedspikes_mod1, binnedspikes_mod2, binnedspikes_mod3, x_pos, y_pos
