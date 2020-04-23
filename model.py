import numpy as np

class Model():

    def __init__(self, input_dim = 4, output_dim = 1):
        
        init_coef = 0.01

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.w1 = np.random.rand(256, self.input_dim) * init_coef
        self.b1 = np.random.rand(256, 1) * init_coef
        self.w2 = np.random.rand(self.output_dim, 256) * init_coef
        self.b2 = np.random.rand(self.output_dim, 1) * init_coef
        
        self.score = 0

    def salt(self, spread):

        self.w1 += (np.random.rand(256, self.input_dim) - .5 ) * spread
        self.b1 += (np.random.rand(256, 1) - .5) * spread
        self.w2 += (np.random.rand(self.output_dim, 256) - .5 ) * spread
        self.b2 += (np.random.rand(self.output_dim, 1)  - .5) * spread

        #print(self.w1, end = "\r")

        return self

    def forward(self, x):
        
        x = np.reshape(x, (self.input_dim, 1))
        x = np.tanh(self.w1 @ x + self.b1)
        x = self.w2 @ x + self.b2
        x = float(np.squeeze(x))
        if x > 0:
            return 1
        else:
            return 0