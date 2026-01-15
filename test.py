import pickle
import numpy as np
from reseau import reseauFactice, relu, reluderiv
from ReadingMnist import MnistDataloader


def testperf():
    with open("parametre.pkl", 'rb') as fichier:
        mon_reseau = pickle.load(fichier)

    loader = MnistDataloader()
    z, (x_test, y_test) = loader.load_data()

    total_images = len(x_test)
    print(f"Nombre d'images à tester : {total_images}")

    bonnes_reponses = 0
    vrais_3_trouves = 0
    vrais_3_rates = 0
    faux_3_detectes = 0

    for i in range(total_images):
        img = np.array(x_test[i]).reshape(-1) / 255.0
        est3 = (y_test[i] == 3)

        sortie = mon_reseau.forward(img)
        predict3 = (sortie > 0.5)

        if predict3 == est3:
            bonnes_reponses += 1
            if est3:
                vrais_3_trouves += 1
        else:
            if est3:
                vrais_3_rates += 1
            else:
                faux_3_detectes += 1

    score = (bonnes_reponses / total_images) * 100
    print(f"Bonnes réponses : {bonnes_reponses} / {total_images}")
    print(f"Précision globale : {score:.2f}%")
    print(f"- Vrais 3 correctement trouvés : {vrais_3_trouves}")
    print(f"- Vrais 3 ratés (Faux Négatifs) : {vrais_3_rates}")
    print(f"- Faux 3 inventés (Faux Positifs) : {faux_3_detectes}")


if __name__ == "__main__":
    testperf()