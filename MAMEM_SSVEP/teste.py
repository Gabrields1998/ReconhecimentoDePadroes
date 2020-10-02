import numpy as np
import json
import mne
import scipy
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from scipy.signal import stft
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

data = np.load('files/data.npy')
labels = np.load('files/labels.npy')

desc_file = open('files/descriptor.json')
deor = json.loads(desc_file.read())
desc_file.close()

print('Estruturas => dados', data.shape, 'labels', labels.shape)
print(labels)

data = data[:, :256, :]

trial_duration = 5
sampling_frequency = data.shape[-1] / trial_duration
montage = mne.channels.make_standard_montage('EGI_256')
ch_names = data.shape[1]
ch_types = 'eeg'

# primeiramente devemos criar o objeto info
info = mne.create_info(montage.ch_names, sampling_frequency, ch_types)

# set experiments montage
info.set_montage(montage)

# por fim a criação do EpochsArray
events = np.array([[index, 0, event] for index, event in enumerate(labels)])
# objeto MNE epoch
#'E128', 'E129', 'E138', 'E139', 'E140', 'E141'
epoch = mne.EpochsArray(data, info, events)
f = open("result2.txt", "w")
# for i in range(100, 200):
#channels = ['E' + str(i)]
channels = ['E101', 'E118', 'E119',
            'E126', 'E127' , 'E128', 'E129',
            'E137','E138', 'E139', 
            'E140', 'E141', 'E148', 'E149',
            'E150','E151', 'E152', 'E158', 'E159',
            'E160']

filtered_epoch = epoch.copy().pick_channels(channels)
filtered_epoch.filter(l_freq=5.0, h_freq=14.0)

# CAR
# mne.set_eeg_reference(filtered_epoch, ref_channels=channels)

X, _ = mne.time_frequency.psd_multitaper(filtered_epoch, fmin=5.0, fmax=14.0)
print('Shape dos dados:', X.shape)

X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
print('Shape dos dados:', X.shape)

y = np.load('files/labels.npy')
print('Shape original dos labels', y.shape)

size = int(X.shape[0] / y.shape[0])
y = np.concatenate([y for i in range(size)])
print('Shape final dos labels', y.shape)
i = 0
for count in range(50):
    for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
        for gamma in [10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]:
            for C in [0.01, 0.1, 1, 10, 100, 1000]:
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
                clf = SVC(gamma=gamma, kernel=kernel, C=C)
                clf.fit(X_train, y_train)
                res = clf.predict(X_test)
                tot_hit = sum([1 for i in range(len(res)) if res[i] == y_test[i]])
                accuracy = tot_hit / X_test.shape[0] * 100
                if accuracy > 50:
                    i += 1
                    print('Kernel:{} | Gamma:{} e C:{} | Accuracy: {:.2f}% | channel: {} | count: {}'.format(
                        kernel, gamma, C, tot_hit / X_test.shape[0] * 100, channels, i)
                    )
                    f.write('Kernel:{} | Gamma:{} e C:{} | Accuracy: {:.2f}% | channel: {}| count: {}\n'.format(
                        kernel, gamma, C, tot_hit / X_test.shape[0] * 100, channels, i)
                    )
f.write('\n')
print('Finish SVM')
f.close()
    # for count in range(20):
        # for n_estimators in [75, 95, 100, 105, 125, 150, 175, 200, 225, 250, 275, 300]:
            # X_train, X_test, y_train, y_test = train_test_split(
                # X, y, train_size=0.7, shuffle=True)
            # regressor = RandomForestClassifier(n_estimators=n_estimators, random_state=25)
            # regressor = regressor.fit(X_train, y_train)
            # y_pred = regressor.predict(X_test)
            # if ( 100*metrics.accuracy_score(y_test, y_pred) >= 50.):
                # print(count)
                # print('Accuracy: n_estimators:{} - {:.2f}%'.format(
                    # n_estimators, 100*metrics.accuracy_score(y_test, y_pred))
                # )
# 
    # print('Finish RF - Random Forest')
