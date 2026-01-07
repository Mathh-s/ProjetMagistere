import numpy as np


x=5
mat = np.random.randint(0, 256, (x, x))

def relu(x):
    return np.maximum(0, x)
def reluderiv(x):
    return np.where(x > 0, 1, 0)

class reseauFactice :

    def __init__(self, start, valeurpoids, nb_couche, nb_neuronnesparcouche):
        self.nb_couche = nb_couche
        self.nb_neuronnesparcouche = nb_neuronnesparcouche
        self.biais = []
        self.vectPoids = []
        self.entree = start.reshape(1, -1)
        for l in range(nb_couche - 1):
            nbentree = nb_neuronnesparcouche[l]
            nbsortie = nb_neuronnesparcouche[l + 1]
            poids = np.full((nbentree, nbsortie), valeurpoids, dtype = float)
            self.vectPoids.append(poids)
            self.biais.append(np.ones((1, nbsortie), dtype=float))
        self.avant = [] # avant RELU
        self.apres = [] # apres RELU



    def forward(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        self.avant = []
        a = x
        self.apres = [x]
        for i in range(len(self.vectPoids)):
            z = np.dot(a, self.vectPoids[i]) + self.biais[i]
            self.avant.append(z)
            a = relu(z)
            self.apres.append(a)
        return a


    def backward(self, y):  # sortie du neurone
        if y.ndim == 1:
            y = y.reshape(1, -1)
        erreur = self.apres[-1] - y
        for i in range(len(self.vectPoids) - 1, -1, -1):
            erreuravant = erreur * reluderiv(self.avant[i])
            changepoids = np.dot(self.apres[i].T, erreuravant)
            changebiais = erreuravant
            erreur = np.dot(erreuravant, self.vectPoids[i].T)
            self.vectPoids[i] -= changepoids*0.01
            self.biais[i] -= changebiais*0.01


def entrainer(reseau, xtrain, ytrain, nb=100):
    for num in range(nb):
        erreurtot = 0
        for i in range(len(xtrain)):
            x = xtrain[i]
            y = ytrain[i]
            sortie = reseau.forward(x)

            erreurtot += np.mean((sortie - y) ** 2)
            reseau.backward(y)

        if num % 10 == 0:
            print(f"Nombre {num}: Erreur moyenne : {erreurtot / len(xtrain):.5f}")


# 1. Données d'entraînement
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y = np.array([[0], [1], [1], [0]], dtype=float)

# 2. Création d'un réseau (2 entrées, 4 neurones cachés, 1 sortie)
monreseau = reseauFactice(X[0], 0.5, 3, [2, 4, 1])

# 3. Entraînement
entrainer(monreseau, X, Y, nb=500)

"""

start = np.array([1, 2], dtype=float)
valeur_poids = 1
nb_couche = 2
nb_neuronnes = [2, 2]

reseau = reseauFactice(start, valeur_poids, nb_couche, nb_neuronnes)


sortie = reseau.forward()
print(sortie)




start = np.array([1, 2], dtype=float)
valeur_poids = 1
nb_couche = 2
nb_neuronnes = [2, 2]

reseau = reseauFactice(start, valeur_poids, nb_couche, nb_neuronnes)

y_attendu = np.array([[0., 1.]])

for i in range(100):
    sortie = reseau.forward(start)
    reseau.backward(y_attendu)
    print(f"itération {i} -> sortie :", sortie)
"""