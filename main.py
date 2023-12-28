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
    gmm = GaussianMixtureModel(n_components=3, max_iterations=200, random_state=0)
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
    test_file_1 = 'test_folder/Benjamin_Netanyau_testing.wav'
    test_file_2 = 'test_folder/Jens_Stoltenberg_testing.wav'

    while(True):
        print("-------------------------------------------------")
        check = input("Enter your choice:")
        if check == "1" :
            print("\nFor the test file ",test_file_1)
            test_scores = test_model(gmm_models, test_file_1)
        elif check== "2" :
            print("\nFor the test file ",test_file_2)
            test_scores = test_model(gmm_models, test_file_2)
        else:
            print("Program terminated!")
            break
        
        print("GMM Score for : ",test_scores["gmm"])
        print("Normal Distribution Score: ", test_scores["normal_dist"])
        predicted_speaker = folders[np.argmax(test_scores["gmm"])]
        print(f"\nPredicted Speaker: {predicted_speaker}")
    