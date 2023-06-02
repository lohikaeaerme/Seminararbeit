import torch
import pandas as pd
from sound_dataset import SoundDS
from torch.utils.data import Subset
from model import SoundClassifier
from training import training
from validation import inference
from torch.utils.data import DataLoader

data_path = ''

# read metadata file
metadata_file = 'data/AudioMNIST/audioMNIST_meta.csv'
meta_data = pd.read_csv(metadata_file)
# read metadata julia
meta_data_julia = pd.read_csv('MNIST_julia/mnist_julia.csv')

# filter some persons for validation
test_persons = [1,9,19,28,47]
val_persons = [44, 26, 33, 37]

train_ids = []
val_ids = []
test_ids = []
for index,person in enumerate(meta_data["person"]):
    if(person in val_persons):
        val_ids.append(index)
    elif(person in test_persons): 
        test_ids.append(index)
    else:
        train_ids.append(index)

# Take relevant columns from meta-data
meta_data = meta_data[['relative_path', 'label']]    

sound_dataset = SoundDS(meta_data, data_path) 
sound_dataset_julia = SoundDS(meta_data_julia, data_path)   

device = torch.device("mps")
# Create the model and put it on the GPU if available
sound_classifier = SoundClassifier()
sound_classifier = sound_classifier.to(device)
# Check that it is on GPU
next(sound_classifier.parameters()).device

epoch_count = 2

# sample elements from given folder structure
test_subset = Subset(sound_dataset, test_ids)
val_subset = Subset(sound_dataset, val_ids)
train_subset = Subset(sound_dataset, train_ids)

# Create training and validation data loaders
test_dl = DataLoader(test_subset, batch_size=16, shuffle=False)
val_dl = DataLoader(val_subset, batch_size=16, shuffle=False) 
train_dl = DataLoader(train_subset, batch_size=16, shuffle=True)
julia_dl = DataLoader(sound_dataset_julia, batch_size=16, shuffle=False, drop_last=False)

training(sound_classifier, train_dl, epoch_count, device, meta_data, test_dl)

sound_dataset.do_augment = False
inference(sound_classifier, val_dl, device, meta_data)

print('_____________________________')
print('on julias data')
inference(sound_classifier, julia_dl, device, meta_data_julia, True)
