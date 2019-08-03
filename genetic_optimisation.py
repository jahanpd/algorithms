import os
import numpy as np
from sumtree import SumTree

class genetic_optimisation:
    def __init__(self,
                 model, # model needs to be a class, with methods build(self.x_train,self.y_train, hp1), train(epochs), evaluate(self.x_test, self.x_test)
                 x_train, # training data
                 y_train, # training labels
                 x_test, # test data
                 y_test, # test labels
                 param_one=None, param_two=None, param_three=None, param_four=None, param_five=None, # two hyperparameter ranges defined as [min,max]
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
        if param_three is not None:
            self.param_two = param_two
            self.params = 3
        if param_four is not None:
            self.param_two = param_two
            self.params = 4
        if param_five is not None:
            self.param_two = param_two
            self.params = 5

    def train(self):
        if self.params == 1:
            top_performers = []
            gen_performance = []
            for gen in range(self.generations):
                performance = SumTree(self.size)
                hp1 = np.random.randint(low = self.param_one[0], high = self.param_one[1], size=(self.size - len(top_performers))*2)
                hps = np.append(np.array(top_performers).reshape((-1,2)), hp1.reshape((-1,2)))
                for hp in hps: # train all models and save performance
                    print(hp)
                    temp= self.model(self.x_train,self.y_train)
                    temp.build(np.mean(np.array([hp[0],hp[1]])))
                    temp.train(self.epochs)
                    metric = temp.evaluate(self.x_test, self.y_test)
                    performance.add(metric, np.array([hp]))
                keep_metrics = np.sort(performance.p_array())[-int(self.keep):] # array of the highest performing metrics
                hyperparameters = [] # array to store the best n=self.keep performing hyperparameters
                for metric in keep_metrics: # note that the order of keep metrics is lowest to highest performance
                    _, __, hp_temp = performance.get(metric)
                    hyperparameters = np.append(np.array(hyperparameters), hp_temp)
                hyperparameters = hyperparameters.reshape((-1,4))
                mated_hp1 = []
                for mate in hyperparameters: # mating routine with mendellian inheritance from the two alleles
                    mated_hp1.append(np.random.choice(mate[np.array([0,1])]))
                    mated_hp1.append(np.random.choice(hyperparameters[np.random.randint(len(hyperparameters))]))
                top_performers = np.array(mated_hp1).reshape((-1,2))
                print("generation:", gen,
                      "  min performance (params, metric):", hyperparameters[0], keep_metrics[0],
                      "  max performance:", hyperparameters[-1], keep_metrics[-1])
                gen_performance.append(keep_metrics[0])
                gen_performance.append(keep_metrics[-1])
            self.hyperparameters = hyperparameters
            self.keep_metrics = keep_metrics
            return(hyperparameters, keep_metrics, np.array(gen_performance).reshape(-1,2))

        if self.params == 2:
            top_performers = []
            gen_performance = []
            os.mkdir("temp")
            for gen in range(self.generations):
                performance = SumTree(self.size)
                hp1 = np.random.randint(low = self.param_one[0], high = self.param_one[1], size=(self.size - len(top_performers))*2)
                hp2 = np.random.randint(low = self.param_two[0], high = self.param_two[1], size=(self.size - len(top_performers))*2)
                hps = np.append(np.array(top_performers).reshape((-1,4)), np.dstack((hp1,hp2))).reshape((-1,4))
                for hp in hps: # train all models and save performance
                    print(hp)
                    temp= self.model(self.x_train,self.y_train)
                    temp.build(np.mean(np.array([hp[0],hp[2]])),
                               np.mean(np.array([hp[1],hp[3]]))
                               )
                    temp.train(self.epochs)
                    metric = temp.evaluate(self.x_test, self.y_test)
                    performance.add(metric, np.array([hp]))
                keep_metrics = np.sort(performance.p_array())[-int(self.keep):] # array of the highest performing metrics
                hyperparameters = [] # array to store the best n=self.keep performing hyperparameters
                for metric in keep_metrics: # note that the order of keep metrics is lowest to highest performance
                    _, __, hp_temp = performance.get(metric)
                    hyperparameters = np.append(np.array(hyperparameters), hp_temp)
                hyperparameters = hyperparameters.reshape((-1,4))
                mated_hp1 = []
                mated_hp2 = []
                for mate in hyperparameters: # mating routine with mendellian inheritance from the two alleles
                    mated_hp1.append(np.random.choice(mate[np.array([0,2])]))
                    mated_hp1.append(np.random.choice(hyperparameters[np.random.randint(len(hyperparameters))][np.array([0,2])]))
                    mated_hp2.append(np.random.choice(mate[np.array([1,3])]))
                    mated_hp2.append(np.random.choice(hyperparameters[np.random.randint(len(hyperparameters))][np.array([1,3])]))
                top_performers = np.dstack((np.array(mated_hp1),
                                            np.array(mated_hp2))
                                            ).reshape((-1,4))
                print("generation:", gen,
                      "  min performance (params, metric):", hyperparameters[0], keep_metrics[0],
                      "  max performance:", hyperparameters[-1], keep_metrics[-1])
                gen_performance.append([keep_metrics[0],keep_metrics[-1]])
                np.savetxt("temp/gen_perf.csv",np.array(gen_performance).reshape(-1,2),delimiter=",")
            os.remove("temp/gen_perf.csv")
            os.rmdir("temp")
            self.hyperparameters = hyperparameters
            self.keep_metrics = keep_metrics
            return(hyperparameters, keep_metrics, np.array(gen_performance).reshape(-1,2))

        if self.params == 3:
            top_performers = []
            gen_performance = []
            for gen in range(self.generations):
                performance = SumTree(self.size)
                hp1 = np.random.randint(low = self.param_one[0], high = self.param_one[1], size=(self.size - len(top_performers))*2)
                hp2 = np.random.randint(low = self.param_two[0], high = self.param_two[1], size=(self.size - len(top_performers))*2)
                hp3 = np.random.randint(low = self.param_three[0], high = self.param_three[1], size=(self.size - len(top_performers))*2)
                hps = np.append(np.array(top_performers).reshape((-1,6)), np.dstack((hp1,hp2,hp3))).reshape((-1,6))
                for hp in hps: # train all models and save performance
                    print(hp)
                    temp= self.model(self.x_train,self.y_train)
                    temp.build(np.mean(np.array([hp[0],hp[3]])),
                               np.mean(np.array([hp[1],hp[4]])),
                               np.mean(np.array([hp[2],hp[5]]))
                               )
                    temp.train(self.epochs)
                    metric = temp.evaluate(self.x_test, self.y_test)
                    performance.add(metric, np.array([hp]))
                keep_metrics = np.sort(performance.p_array())[-int(self.keep):] # array of the highest performing metrics
                hyperparameters = [] # array to store the best n=self.keep performing hyperparameters
                for metric in keep_metrics: # note that the order of keep metrics is lowest to highest performance
                    _, __, hp_temp = performance.get(metric)
                    hyperparameters = np.append(np.array(hyperparameters), hp_temp)
                hyperparameters = hyperparameters.reshape((-1,6))
                mated_hp1 = []
                mated_hp2 = []
                mated_hp3 = []
                for mate in hyperparameters: # mating routine with mendellian inheritance from the two alleles
                    mated_hp1.append(np.random.choice(mate[np.array([0,3])]))
                    mated_hp1.append(np.random.choice(hyperparameters[np.random.randint(len(hyperparameters))][np.array([0,3])]))
                    mated_hp2.append(np.random.choice(mate[np.array([1,4])]))
                    mated_hp2.append(np.random.choice(hyperparameters[np.random.randint(len(hyperparameters))][np.array([1,4])]))
                    mated_hp3.append(np.random.choice(mate[np.array([2,5])]))
                    mated_hp3.append(np.random.choice(hyperparameters[np.random.randint(len(hyperparameters))][np.array([2,5])]))
                top_performers = np.dstack((np.array(mated_hp1),
                                            np.array(mated_hp2),
                                            np.array(mated_hp3))
                                            ).reshape((-1,6))
                print("generation:", gen,
                      "  min performance (params, metric):", hyperparameters[0], keep_metrics[0],
                      "  max performance:", hyperparameters[-1], keep_metrics[-1])
                gen_performance.append([keep_metrics[0],keep_metrics[-1]])
            self.hyperparameters = hyperparameters
            self.keep_metrics = keep_metrics
            return(hyperparameters, keep_metrics, np.array(gen_performance).reshape(-1,2))

        if self.params == 4:
            top_performers = []
            gen_performance = []
            for gen in range(self.generations):
                performance = SumTree(self.size)
                hp1 = np.random.randint(low = self.param_one[0], high = self.param_one[1], size=(self.size - len(top_performers))*2)
                hp2 = np.random.randint(low = self.param_two[0], high = self.param_two[1], size=(self.size - len(top_performers))*2)
                hp3 = np.random.randint(low = self.param_three[0], high = self.param_three[1], size=(self.size - len(top_performers))*2)
                hp4 = np.random.randint(low = self.param_four[0], high = self.param_four[1], size=(self.size - len(top_performers))*2)
                hps = np.append(np.array(top_performers).reshape((-1,8)), np.dstack((hp1,hp2,hp3,hp4))).reshape((-1,8))
                for hp in hps: # train all models and save performance
                    print(hp)
                    temp= self.model(self.x_train,self.y_train)
                    temp.build(np.mean(np.array([hp[0],hp[4]])),
                               np.mean(np.array([hp[1],hp[5]])),
                               np.mean(np.array([hp[2],hp[6]])),
                               np.mean(np.array([hp[3],hp[7]]))
                               )
                    temp.train(self.epochs)
                    metric = temp.evaluate(self.x_test, self.y_test)
                    performance.add(metric, np.array([hp]))
                keep_metrics = np.sort(performance.p_array())[-int(self.keep):] # array of the highest performing metrics
                hyperparameters = [] # array to store the best n=self.keep performing hyperparameters
                for metric in keep_metrics: # note that the order of keep metrics is lowest to highest performance
                    _, __, hp_temp = performance.get(metric)
                    hyperparameters = np.append(np.array(hyperparameters), hp_temp)
                hyperparameters = hyperparameters.reshape((-1,8))
                mated_hp1 = []
                mated_hp2 = []
                mated_hp3 = []
                mated_hp4 = []
                for mate in hyperparameters: # mating routine with mendellian inheritance from the two alleles
                    mated_hp1.append(np.random.choice(mate[np.array([0,4])]))
                    mated_hp1.append(np.random.choice(hyperparameters[np.random.randint(len(hyperparameters))][np.array([0,4])]))
                    mated_hp2.append(np.random.choice(mate[np.array([1,5])]))
                    mated_hp2.append(np.random.choice(hyperparameters[np.random.randint(len(hyperparameters))][np.array([1,5])]))
                    mated_hp3.append(np.random.choice(mate[np.array([2,6])]))
                    mated_hp3.append(np.random.choice(hyperparameters[np.random.randint(len(hyperparameters))][np.array([2,6])]))
                    mated_hp4.append(np.random.choice(mate[np.array([3,7])]))
                    mated_hp4.append(np.random.choice(hyperparameters[np.random.randint(len(hyperparameters))][np.array([3,7])]))
                top_performers = np.dstack((np.array(mated_hp1),
                                            np.array(mated_hp2),
                                            np.array(mated_hp3),
                                            np.array(mated_hp4))
                                            ).reshape((-1,8))
                print("generation:", gen,
                      "  min performance (params, metric):", hyperparameters[0], keep_metrics[0],
                      "  max performance:", hyperparameters[-1], keep_metrics[-1])
                gen_performance.append([keep_metrics[0],keep_metrics[-1]])
            self.hyperparameters = hyperparameters
            self.keep_metrics = keep_metrics
            return(hyperparameters, keep_metrics, np.array(gen_performance).reshape(-1,2))

        if self.params == 5:
            top_performers = []
            gen_performance = []
            for gen in range(self.generations):
                performance = SumTree(self.size)
                hp1 = np.random.randint(low = self.param_one[0], high = self.param_one[1], size=(self.size - len(top_performers))*2)
                hp2 = np.random.randint(low = self.param_two[0], high = self.param_two[1], size=(self.size - len(top_performers))*2)
                hp3 = np.random.randint(low = self.param_three[0], high = self.param_three[1], size=(self.size - len(top_performers))*2)
                hp4 = np.random.randint(low = self.param_four[0], high = self.param_four[1], size=(self.size - len(top_performers))*2)
                hp5 = np.random.randint(low = self.param_five[0], high = self.param_five[1], size=(self.size - len(top_performers))*2)
                hps = np.append(np.array(top_performers).reshape((-1,10)), np.dstack((hp1,hp2,hp3,hp4,hp5))).reshape((-1,10))
                for hp in hps: # train all models and save performance
                    print(hp)
                    temp= self.model(self.x_train,self.y_train)
                    temp.build(np.mean(np.array([hp[0],hp[5]])),
                               np.mean(np.array([hp[1],hp[6]])),
                               np.mean(np.array([hp[2],hp[7]])),
                               np.mean(np.array([hp[3],hp[8]])),
                               np.mean(np.array([hp[4],hp[9]]))
                               )
                    temp.train(self.epochs)
                    metric = temp.evaluate(self.x_test, self.y_test)
                    performance.add(metric, np.array([hp]))
                keep_metrics = np.sort(performance.p_array())[-int(self.keep):] # array of the highest performing metrics
                hyperparameters = [] # array to store the best n=self.keep performing hyperparameters
                for metric in keep_metrics: # note that the order of keep metrics is lowest to highest performance
                    _, __, hp_temp = performance.get(metric)
                    hyperparameters = np.append(np.array(hyperparameters), hp_temp)
                hyperparameters = hyperparameters.reshape((-1,10))
                mated_hp1 = []
                mated_hp2 = []
                mated_hp3 = []
                mated_hp4 = []
                mated_hp5 = []
                for mate in hyperparameters: # mating routine with mendellian inheritance from the two alleles
                    mated_hp1.append(np.random.choice(mate[np.array([0,5])]))
                    mated_hp1.append(np.random.choice(hyperparameters[np.random.randint(len(hyperparameters))][np.array([0,5])]))
                    mated_hp2.append(np.random.choice(mate[np.array([1,6])]))
                    mated_hp2.append(np.random.choice(hyperparameters[np.random.randint(len(hyperparameters))][np.array([1,6])]))
                    mated_hp3.append(np.random.choice(mate[np.array([2,7])]))
                    mated_hp3.append(np.random.choice(hyperparameters[np.random.randint(len(hyperparameters))][np.array([2,7])]))
                    mated_hp4.append(np.random.choice(mate[np.array([3,8])]))
                    mated_hp4.append(np.random.choice(hyperparameters[np.random.randint(len(hyperparameters))][np.array([3,8])]))
                    mated_hp5.append(np.random.choice(mate[np.array([4,9])]))
                    mated_hp5.append(np.random.choice(hyperparameters[np.random.randint(len(hyperparameters))][np.array([4,9])]))
                top_performers = np.dstack((np.array(mated_hp1),
                                            np.array(mated_hp2),
                                            np.array(mated_hp3),
                                            np.array(mated_hp4),
                                            np.array(mated_hp5))
                                            ).reshape((-1,10))
                print("generation:", gen,
                      "  min performance (params, metric):", hyperparameters[0], keep_metrics[0],
                      "  max performance:", hyperparameters[-1], keep_metrics[-1])
                gen_performance.append([keep_metrics[0],keep_metrics[-1]])
            self.hyperparameters = hyperparameters
            self.keep_metrics = keep_metrics
            return(hyperparameters, keep_metrics, np.array(gen_performance).reshape(-1,2))
