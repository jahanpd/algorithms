import numpy as np
from sumtree import SumTree

class genetic_optimisation:
    def __init__(self,
                 model, # model needs to be a class, with methods build(self.x_train,self.y_train, hp1), train(epochs), evaluate(self.x_test, self.x_test)
                 x_train, # training data
                 y_train, # training labels
                 x_test, # test data
                 y_test, # test labels
                 param_one=None, param_two=None, # two hyperparameter ranges defined as [min,max]
                 epochs=None, generations=None, keep=None, size=None):

        self.model = model

        # set the training data and labels metrics
        self.x_train = x_train.astype(np.float32) # value of inputs
        self.y_train = y_train.astype(np.float32) # value of targets
        self.x_test = x_test.astype(np.float32) # value of inputs
        self.y_test = y_test.astype(np.float32) # value of targets

        if size is not None: # population of models will be 20 unless otherwise specified
            self.size = size
        else:
            self.size = 20

        if epochs is not None: # train each model for 100 epochs unless otherwise specified
            self.epochs = epochs
        else:
            self.epochs = 100

        if generations is not None: # carry over for 100 generations unless otherwise specified
            self.generations = generations
        else:
            self.generations = 100

        if keep is not None: # keep 20% each generation, and generate 80% random samples unless otherwise specified
            self.keep = np.floor(keep*self.size)
        else:
            self.keep = np.floor(0.2*self.size)

        if param_one is not None:
            self.param_one = param_one
            self.params = 1
        else:
            print("WARNING: no hyperparameters entered")
        if param_two is not None:
            self.param_two = param_two
            self.params = 2

    def train(self):
        if self.params == 1:
            top_performers = []
            for gen in range(self.generations):
                performance = SumTree(self.size)
                randoms = np.random.randint(low = self.param_one[0], high = self.param_one[1], size=(self.size - len(top_performers)))
                hps = np.append(np.array(top_performers), randoms)
                for hp in hps: # train all models and save performance
                    temp = self.model(self.x_train,self.y_train)
                    temp.build(hp)
                    temp.train(self.epochs)
                    metric = temp.evaluate(self.x_test, self.y_test)
                    performance.add(metric, np.array([hp]))
                keep_metrics = performance.tree[-int(self.keep):] # array of the highest performing metrics
                hyperparameters = [] # array to store the best n=self.keep performing hyperparameters
                for metric in keep_metrics: # note that the order of keep metrics is lowest to highest performance
                    _, __, hp_temp = performance.get(metric)
                    hyperparameters.append(hp_temp[0])
                mated = []
                for mate in range(len(hyperparameters)-2): # mating routine
                    mated.append(np.mean(hyperparameters[mate:mate+1]))
                top_performers = mated
                print("generation:", gen,
                      "  min performance (params, metric):", hyperparameters[0], keep_metrics[0],
                      "  max performance:", hyperparameters[-1], keep_metrics[-1])
            self.hyperparameters = hyperparameters
            self.keep_metrics = keep_metrics
            return(hyperparameters, keep_metrics)

        if self.params == 2:
            top_performers = []
            for gen in range(self.generations):
                performance = SumTree(self.size)
                hp1 = np.random.randint(low = self.param_one[0], high = self.param_one[1], size=(self.size - len(top_performers)))
                hp2 = np.random.randint(low = self.param_two[0], high = self.param_two[1], size=(self.size - len(top_performers)))
                hps = np.append(np.array(top_performers).reshape((-1,2)), np.dstack((hp1,hp2))).reshape((-1,2))
                for hp in hps: # train all models and save performance
                    temp= self.model(self.x_train,self.y_train)
                    temp.build(hp[0],hp[1])
                    temp.train(self.epochs)
                    metric = temp.evaluate(self.x_test, self.y_test)
                    performance.add(metric, np.array([hp]))
                keep_metrics = performance.tree[-int(self.keep):] # array of the highest performing metrics
                hyperparameters = [] # array to store the best n=self.keep performing hyperparameters
                for metric in keep_metrics: # note that the order of keep metrics is lowest to highest performance
                    _, __, hp_temp = performance.get(metric)
                    hyperparameters = np.append(np.array(hyperparameters), hp_temp)
                mated_hp1 = []
                mated_hp2 = []
                for mate in range(len(hyperparameters.reshape((-1,2))) - 2): # mating routine
                    mated_hp1.append(np.mean(hyperparameters[mate:mate+1,0]))
                    mated_hp2.append(np.mean(hyperparameters[mate:mate+1,1]))
                top_performers = np.dstack((np.array(hp1),np.array(hp2))).reshape((-1,2))
                print("generation:", gen,
                      "  min performance (params, metric):", hyperparameters, keep_metrics[0],
                      "  max performance:", hyperparameters, keep_metrics[-1])
            self.hyperparameters = hyperparameters
            self.keep_metrics = keep_metrics
            return(hyperparameters, keep_metrics)
