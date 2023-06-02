from torch.utils.data import Dataset
from sound_util import SoundUtil

# ----------------------------
# Sound Dataset
# ----------------------------
class SoundDS(Dataset):
  def __init__(self, df, data_path, do_augment=True):
    self.df = df
    self.data_path = str(data_path)
    self.duration = 1000
    self.sr = 48000
    self.channel = 2
    self.shift_pct = 0.4
    self.do_augment = do_augment
            
  # ----------------------------
  # Number of items in dataset
  # ----------------------------
  def __len__(self):
    return len(self.df)    
    
  # ----------------------------
  # Get i'th item in dataset
  # ----------------------------
  def __getitem__(self, idx):
    # Absolute file path of the audio file - concatenate the audio directory with
    # the relative path
    audio_file = self.data_path + self.df.loc[idx, 'relative_path']
    # Get the Class ID
    class_id = self.df.loc[idx, 'label']

    aud = SoundUtil.open(audio_file)

    dur_aud = SoundUtil.pad_trunc(aud, self.duration)

    if(self.do_augment):
      shift_aud = SoundUtil.time_shift(dur_aud, self.shift_pct)
      sgram = SoundUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
      aug_sgram = SoundUtil.spectro_augment(sgram, max_mask_pct=0.025, n_freq_masks=1, n_time_masks=1)
      return aug_sgram, class_id, idx
    else :
      sgram = SoundUtil.spectro_gram(dur_aud, n_mels=64, n_fft=1024, hop_len=None)
      return sgram, class_id, idx
