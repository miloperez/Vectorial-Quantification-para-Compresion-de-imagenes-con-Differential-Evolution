import numpy as np
import DifferentialEvolution as DE
import KNN as myKNN
import math


class VectorialQuantification:
    def __init__(self, blockSide, n_codewords, img):
        self.blockSize = blockSide
        self.n_codewords = n_codewords
        self.X = self.__blockSeparation(img, blockSide)

        dim = n_codewords * pow(blockSide, 2)

        # def __init__(self, D, low, high, F, Cr, NP, MaxGen, fitness):
        self.codewords = DE.DiffEvo(dim, 0, 255, 0.8, 0.9, 30, 25, self.__distortion)
        self.opt = self.codewords.getBest()
        # print(self.opt)

    def getImage(self):
        tmp = self.__blockJoin(self.__engine(self.opt))
        return tmp

    # aka fitness
    def __distortion(self, codewords):
        X_1 = self.__engine(codewords)

        sum = 0
        for j in range(len(self.X)):
            sum += math.dist(self.X[j], X_1[j])

        return sum / self.n_codewords

    def __engine(self, codewords):
        X_1 = []

        a = self.n_codewords
        b = len(codewords)/self.n_codewords

        codewords = np.reshape(codewords, (int(a), int(b)))
        Y_tr = list(range(len(codewords)))

        classify = myKNN.KNN(1, codewords, Y_tr)
        for i in self.X:
            X_1.append(codewords[classify.predict(i)])
        return X_1

    def __blockSeparation(self, X, lado):
        X_t = []
        rows, cols = np.shape(X)
        for i in range(0,rows,lado):
            for j in range(0, cols, lado):
                sub_bloque = X[i:i+lado, j:j+lado]
                X_t.append(sub_bloque.ravel())
        return X_t

    def __blockJoin(self, X):
        lado = self.blockSize
        size = len(X) * len(X[0])
        side = int(pow(size, 0.5))
        X_t = np.zeros((side, side))

        lol = int(pow(len(X), 0.5))

        for I in range(lol):
            for J in range(lol):
                for i in range(lado):
                    for j in range(lado):
                        X_t[(I * lado) + i][(J * lado) + j] = X[(I * lol) + J][(i * lado) + j]

        return list(X_t)
