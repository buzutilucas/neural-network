from neuralnetwork import NeuralNetworkIris

if __name__ == '__main__':

    ai = NeuralNetworkIris()
    print(ai.predict([4.6, 3.6, 1, 0.2]))
    print(ai.predict([7, 4.6, 3, 0.2]))