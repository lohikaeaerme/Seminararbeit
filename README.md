# Seminararbeit

Code zu meiner Seminararbeit zum The deep learning sound classification. Baut auf dem Artikel [Audio Deep Learning Made Simple: Sound Classification, Step-by-Step](https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5)
von ketan Doshi auf. Sie beschäftigt sich mit der Frage wie datapreparation und augmentation für Soundaten funktioniert. Weiterhin implementiert sie zwei deep learning Modelle.
Eines aus dem Artikel von Ketan Doshi(soundclassifier) und eines von [paperswithcode](https://github.com/Alibaba-MIIL/AudioClassfication)(soundnet).

Weiterhin habe ich das gelernte dann auf den [AudioMNIST](https://www.kaggle.com/datasets/sripaadsrinivasan/audio-mnist) angewandt und eigene Audioaufnahmen dazu gesprochen. 

## How to use
1. Datensäze runterladen: [urbansound8k](https://urbansounddataset.weebly.com/urbansound8k.html) und [audioMNIST](https://www.kaggle.com/datasets/sripaadsrinivasan/audio-mnist)
2. audioMNIST label erstellen mit Hilfe von audioMNIST_make_label_df.ipybn
3. für die einzelnen Netze jeweils im passenden Ordner die Datei sound_classifier.py ausführen. (Voreingestellt mit 10 epochen)
