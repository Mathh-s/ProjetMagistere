import numpy as np


x=5
mat = np.random.randint(0, 256, (x, x))

def relu(x):
    return np.maximum(0, x)
def reluderiv(x):
    return np.where(x > 0, 1, 0)

class reseauFactice :

    def __init__(self, start, valeurpoids, biais, nb_couche, nb_neuronnesparcouche, excepted):
        self.nb_couche = nb_couche
        self.nb_neuronnesparcouche = nb_neuronnesparcouche
        self.entree = start
        self.excepted = excepted
        self.biais = np.full((1,nb_couche), 1)
        self.vectPoids = []
        for l in range(nb_couche - 1):
            nbentree = nb_neuronnesparcouche[l]
            nbsortie = nb_neuronnesparcouche[l + 1]
            poids = np.full((nbentree, nbsortie), valeurpoids, dtype = float)
            self.vectPoids.append(poids)
            self.biais.append(np.ones((1, nbsortie), dtype=float))
        self.avant = [] #avant RELU
        self.apres = [] # apres RELU



    def forward(self):
        a = self.entree
        self.avant = []
        self.apres = [a]
        for i in range(len(self.vectPoids)):
            poids = self.vectPoids[i]
            b = self.biais[i]
            z = np.dot(a, poids) + b
            self.avant.append(z)
            a = relu(z)
            self.apres.append(a)
        return a


    def erreur(self):
        if self.sortie != self.excepted:
            return 0
        return 1


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




"""

start = np.array([1, 2], dtype=float)
valeur_poids = 1
nb_couche = 2
nb_neuronnes = [2, 2]

reseau = reseauFactice(start, valeur_poids, nb_couche, nb_neuronnes)


sortie = reseau.forward()
print(sortie)
"""



start = np.array([1, 2], dtype=float)
valeur_poids = 1
nb_couche = 2
nb_neuronnes = [2, 2]

reseau = reseauFactice(start, valeur_poids, nb_couche, nb_neuronnes)

y_attendu = np.array([[0., 1.]])

for i in range(10):
    sortie = reseau.forward()
    reseau.backward(y_attendu)
    print(f"it {i} -> sortie :", sortie)