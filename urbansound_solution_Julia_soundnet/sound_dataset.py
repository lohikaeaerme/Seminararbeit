from torch.utils.data import Dataset
from sound_util import SoundUtil
import torch

# ----------------------------
# Sound Dataset
# ----------------------------
class SoundDS(Dataset):
  def __init__(self, df, data_path, do_augment=True):
    self.df = df
    self.data_path = str(data_path)
    self.duration = 4000
    self.sr = 44100
    self.channel = 1
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
    class_id = self.df.loc[idx, 'classID']

    aud = SoundUtil.open(audio_file)
    # Some sounds have a higher sample rate, or fewer channels compared to the
    # majority. So make all sounds have the same number of channels and same 
    # sample rate. Unless the sample rate is the same, the pad_trunc will still
    # result in arrays of different lengths, even though the sound duration is
    # the same.
    reaud = SoundUtil.resample(aud, self.sr)
    rechan = SoundUtil.rechannel(reaud, self.channel)

    dur_aud = SoundUtil.pad_trunc(rechan, self.duration)

    if(self.do_augment):
      shift_aud = SoundUtil.time_shift(dur_aud, self.shift_pct)
      sgram = SoundUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
      aug_sgram = SoundUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)
      #aug_sgram.squeeze_()
      aug_sgram_1d = torch.reshape(aug_sgram, (-1,))
      #print(aug_sgram_1d.shape)
      aug_sgram_1d = aug_sgram_1d.unsqueeze(0)
      return aug_sgram_1d, class_id, idx
    else :
      sgram = SoundUtil.spectro_gram(dur_aud, n_mels=64, n_fft=1024, hop_len=None)
      sgram_1d = torch.reshape(sgram, (-1,))
      sgram_1d = sgram_1d.unsqueeze(0)
      return sgram_1d, class_id, idx
