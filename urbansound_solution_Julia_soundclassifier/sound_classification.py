import torch
import pandas as pd
from sound_dataset import SoundDS
from k_fold import k_fold
from torch.utils.data import Subset
from model import SoundClassifier
from training import training
from validation import inference
from tqdm import tqdm

# laod data 
data_path = 'data/UrbanSound8K/audio'

# read metadata file
metadata_file = 'data/UrbanSound8K/metadata/UrbanSound8K.csv'
meta_data = pd.read_csv(metadata_file)
# Construct file path by concatenating fold and file name
meta_data['relative_path'] = '/fold' + meta_data['fold'].astype(str) + '/' + meta_data['slice_file_name'].astype(str)

# make list of indices for folds
folds = []
for fold in meta_data['fold'].unique():
    folds.append(list(meta_data[meta_data['fold'] == fold].index))

# Take relevant columns from meta-data
meta_data = meta_data[['relative_path', 'classID']]    

sound_dataset = SoundDS(meta_data, data_path)    

device = torch.device("mps")

epoch_count = 10
# For fold results
results = {}
results_tests = []
results_train = []
results_losses = []
# K-fold Cross Validation model evaluation
for val,test, train, fold in tqdm(k_fold(folds=folds)):
    # initialise and reset neuronael network
    #sound_classifier.reset_weights()

    # Create the model and put it on the GPU if available
    sound_classifier = SoundClassifier()
    sound_classifier = sound_classifier.to(device)
    # Check that it is on GPU
    next(sound_classifier.parameters()).device

    # print foldnumer
    print(f'FOLD {fold}')
    print('--------------------------------')

    # sample elements from given folder structure
    val_subset = Subset(sound_dataset, val)
    train_subset = Subset(sound_dataset, train)
    test_subset = Subset(sound_dataset, test)

    # Create training and validation data loaders
    val_dl = torch.utils.data.DataLoader(val_subset, batch_size=16, shuffle=False) 
    train_dl = torch.utils.data.DataLoader(train_subset, batch_size=16, shuffle=True)
    test_dl = torch.utils.data.DataLoader(test_subset, batch_size=16, shuffle=False)


    test_acc, train_acc, losses = training(sound_classifier, train_dl, epoch_count, device, fold, meta_data, test_dl)
    results_tests.append(test_acc)
    results_train.append(train_acc)
    results_losses.append(losses)

    sound_dataset.do_augment = False
    results[fold] = inference(sound_classifier, val_dl, device, fold, meta_data)
    sound_dataset.do_augment = True


print(f'K-FOLD CROSS VALIDATION RESULTS FOR {len(folds)} FOLDS')
print('--------------------------------')

all_average = []
for epoch in range(len(results_tests[0])):
    sum = 0.0
    for i in range(len(results_tests)):
        sum += results_tests[i][epoch]
    print(f'Average on Test Data in epoch{epoch+1} : {sum/len(results_tests)}')
    all_average.append(sum/len(results_tests))
print(all_average)

print('\n_________________________________\n')

print("test results")
print(results_tests)
print("train results")
print(results_train)
print("loss results")
print(results_losses)


print('\n_________________________________\n')
all_average = []
for epoch in range(len(results_tests[0])):
    sum = 0.0
    for i in range(len(results_tests)):
        sum += results_train[i][epoch]
    print(f'Average on Train Data in epoch{epoch+1} : {sum/len(results_tests)}')
    all_average.append(sum/len(results_tests))
print(all_average)

print('\n_________________________________\n')
all_average = []
for epoch in range(len(results_tests[0])):
    sum = 0.0
    for i in range(len(results_tests)):
        sum += results_losses[i][epoch]
    print(f'Average loss in epoch{epoch+1} : {sum/len(results_tests)}')
    all_average.append(sum/len(results_tests))
print(all_average)

print('\n_________________________________\n')
sum = 0.0
for key, value in results.items():
    print(f'Fold {key}: {value} %')
    sum += value
print(f'Average: {sum/len(results.items())} %')