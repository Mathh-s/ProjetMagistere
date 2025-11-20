import numpy as np



x=5
mat = np.random.randint(0, 256, (x, x))



class reseauFactice :

    def __init__(self,start, valeurpoids, biais, nb_couche, nb_neuronnesparcouche):
        self.nb_couche = nb_couche
        self.nb_neuronnesparcouche = nb_neuronnesparcouche
        self.entree = start
        self.biais = np.full((1,nb_couche), 1)
        self.vectPoids = []
        for l in range(nb_couche - 1):
            nbentree = nb_neuronnesparcouche[l]
            nbsortie = nb_neuronnesparcouche[l + 1]
            poids = np.full((nbentree, nbsortie), valeurpoids, dtype = float)
            self.vectPoids.append(poids)


    def neurone(self):
        return sortie                       # sortie de dim 1



    def forward(self):
        a = self.entree
        for i in range(len(self.vectPoids)):
            poids = self.vectPoids[i]
            b = self.biais[i]
            a = np.dot(a, poids) + b
        return a



    def backward(self):                     # sortie du neurone
        calculgradient()
        self.vectPoids = newpoids

