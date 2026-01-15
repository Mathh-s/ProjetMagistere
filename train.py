import numpy as np
from reseau import reseauFactice, entrainer
from ReadingMnist import MnistDataloader
import pickle


loader = MnistDataloader()
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = loader.load_data()

X_train_clean = []
Y_train_clean = []


for i in range(len(x_train_raw)):
    img = np.array(x_train_raw[i])
    img_flat = img.reshape(-1)
    img_norm = img_flat / 255.0
    X_train_clean.append(img_norm)

    valeur_reelle = y_train_raw[i]
    if valeur_reelle == 3:
        target = np.array([1.0])
    else:
        target = np.array([0.0])
    Y_train_clean.append(target)

monreseau = reseauFactice(X_train_clean[0], 0, 3, [784, 32, 1])
entrainer(monreseau, X_train_clean, Y_train_clean, nb=100)

nom_fichier = "parametre.pkl"
print(f"Sauvegarde du réseau dans {nom_fichier}")

with open(nom_fichier, 'wb') as fichier:
    pickle.dump(monreseau, fichier)

'''
img_test = np.array(x_test_raw[0]).reshape(-1) / 255.0
label_test = y_test_raw[0]

resultat = monreseau.forward(img_test)
print(f"Chiffre réel : {label_test}")
print(f"Probabilité que ce soit un 3 selon le réseau : {resultat[0][0]:.4f}")
'''