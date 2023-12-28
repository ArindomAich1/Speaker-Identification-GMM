# Speaker-Identification-GMM

## Work Process

- Uses Kaggle Dataset.
- Extracts the MFCC from the speech sounds.
- Makes one model for each speakers using the gmm package
- While testing , it carries out the log-likelyhood of the testfile to belong to each model and the highest score gets accepted(Maximization Problem).

## Packeges

- numpy
- scipy.io
- python_speech_features
- os
- matplotlib
