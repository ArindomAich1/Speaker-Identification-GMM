import numpy as np
from scipy.io import wavfile
from python_speech_features import mfcc
from gmm import GaussianMixtureModel
import os
import matplotlib.pyplot as plt

# Step 1: Feature Extraction
def extract_features(file_path):
    rate, audio = wavfile.read(file_path)
    features = mfcc(audio, rate)
    return features

# Step 2: Training the GMM Model
def train_model(folders):
    data = []
    for folder in folders:
        print("Folder: ", folder)
        files = [f for f in os.listdir(folder) if f.endswith('.wav')]
        for file in files:
            print("File: ", folder,"/",file)
            file_path = os.path.join(folder, file)
            features = extract_features(file_path)
            data.extend(features)
    data = np.array(data)
    print("gmm initiated...")
    gmm = GaussianMixtureModel(n_components=2, max_iterations=200, random_state=0)
    normal_dist = GaussianMixtureModel(n_components=1, max_iterations=1, random_state=0)
    gmm.fit(data)
    normal_dist.fit(data)
    return {
        "gmm": gmm,
        "normal_dist": normal_dist
    }

# Step 3: Testing the GMM Model
def test_model(gmm_models, test_file):
    features = extract_features(test_file)
    scores_gmm = [gmm["gmm"].score(features) for gmm in gmm_models]
    normal_dist_score = [normal_dist["normal_dist"].score(features) for normal_dist in gmm_models]
    return {
        "gmm": scores_gmm, 
        "normal_dist" :normal_dist_score
    } 

# main function
if __name__ == '__main__':
    folders = ["Benjamin_Netanyau", "Jens_Stoltenberg"]
    print("Training initiated...")
    gmm_models = [train_model([folder]) for folder in folders]

    file_names1 = []

    for i in range(601, 801):
        file_names1.append(f"Benjamin_Netanyau_testing/{i}.wav")

    file_names2 = []

    for i in range(601, 801):
        file_names2.append(f"Jens_Stoltenberg_testing/{i}.wav")
    
    k_gmm = 0.0
    k_normal_dist = 0.0
    count  = 0.0
    
    li = []

    for i in file_names1:
        test_scores = test_model(gmm_models, i) 

        predicted_speaker = folders[np.argmax(test_scores["gmm"])]
        li.append(predicted_speaker)
        if predicted_speaker == "Benjamin_Netanyau":
            print(predicted_speaker,"1")
            k_gmm = k_gmm + 1.0

        predicted_speaker = folders[np.argmax(test_scores["normal_dist"])]
        li.append(predicted_speaker)
        if predicted_speaker == "Benjamin_Netanyau":
            print(predicted_speaker,"1")
            k_normal_dist = k_normal_dist + 1.0
        count = count + 1.0

    for i in file_names2:
        test_scores = test_model(gmm_models, i) 
        predicted_speaker = folders[np.argmax(test_scores["gmm"])]
        li.append(predicted_speaker)
        if predicted_speaker == "Jens_Stoltenberg":
            print(predicted_speaker,"2")
            k_gmm = k_gmm + 1.0
               
        predicted_speaker = folders[np.argmax(test_scores["normal_dist"])]
        li.append(predicted_speaker)
        if predicted_speaker == "Jens_Stoltenberg":
            print(predicted_speaker,"2")
            k_normal_dist = k_normal_dist + 1.0 
        count = count + 1.0 

    acc_gmm = k_gmm/count
    acc_normal_dist = k_normal_dist/count

    print("\nTraining Information:")
    print('''
--------------------------------------------------------
|No. of files:\t\t|Training\t|Testing\t|
--------------------------------------------------------
|Benjamin_Netanyau:\t|1300\t\t|200\t\t|
--------------------------------------------------------
|Jens_Stoltenberg:\t|1300\t\t|200\t\t|
--------------------------------------------------------
                  ''')    
    print("\nAccuracy Table:")
    print(f'''
-------------------------------------------------
|Models\t\t|GMM\t|Normal Distribution\t|
-------------------------------------------------
|Accuracy\t|{acc_gmm}\t|{acc_normal_dist}\t\t\t|
-------------------------------------------------
                  ''')
    
    print(li)