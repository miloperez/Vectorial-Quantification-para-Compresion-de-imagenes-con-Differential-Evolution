import numpy as np
import math


class KNN:
    def __init__(self, K, X_tr, Y_tr):
        self.K = K
        # conjunto de entrenamiento comprimido en un solo arreglo
        self.Z = []
        for i in range(len(X_tr)):
            self.Z.append([X_tr[i], Y_tr[i]])

    def test(self, X_test, Y_test):
        aciertos = 0
        total = 0

        # revisar todos los elementos del conjunto de prueba
        for j in range(len(X_test)):
            # obtener una predicción según los conjuntos de entrenamiento
            tag = self.predict(X_test[j])

            # si la etiqueta corresponde, contar como acierto
            if tag == Y_test[j]:
                aciertos += 1
            total += 1

        # reportar porcentaje de aciertos
        return aciertos / total

    def predict(self, a):
        Dist = []
        # obtener las distancias de cada elemento con los del conjunto de entrenamiento
        # preservando sus etiquetas
        for i in range(len(self.Z)):
            Dist.append([[math.dist(a, self.Z[i][0])], [self.Z[i][1]]])
        Dist = sorted(Dist)
        Y = []
        # obtener un arreglo con las etiquetas
        for k in range(len(Dist)):
            Y.append(Dist[k][1])
        # obtener un arreglo con los diferentes valores de las etiquetas
        dif = np.unique(np.array(Y))
        # predicción
        tag = 0
        # revisar los K primeros elementos y votar por la etiqueta que más se repite
        for d in dif:
            x = [i for i in Y[:self.K] if i == d]
            if len(x) > tag:
                tag = d

        return tag
