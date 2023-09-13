import mesa
import random
from opensimplex import OpenSimplex
import matplotlib.pyplot as plt
import numpy as np

#Made these global variables to stop them being reinitialised when batch run
preyQTable = {}
predQTable = {}
ApredQTable = {}


class Model(mesa.Model):

    def __init__(self, M, N, O,  width, height, learningRate, discountFactor, explorationRate, vis):
        self.learningRate = learningRate
        self.discountFactor = discountFactor
        self.explorationRate = explorationRate
        Model.resources = self.mapGen(width, height, vis)
        Model.num_prey = N
        Model.num_pred = M
        Model.num_Apred = O
        Model.highestId = 0
        Model.agentsToDie = []
        Model.agentsToAdd = []
        self.grid = mesa.space.MultiGrid(width, height, True)
        self.schedule = mesa.time.RandomActivation(self)
        self.running = True
        for i in range(self.num_pred):
            Model.highestId +=  1
            a = pred(Model.highestId, self, self.learningRate, self.discountFactor, self.explorationRate)
            self.schedule.add(a)
          
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

        for i in range(self.num_prey):
            Model.highestId += 1
            b = prey(Model.highestId, self, self.learningRate, self.discountFactor, self.explorationRate)
            self.schedule.add(b)
          
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(b, (x, y))

        for i in range(self.num_Apred):
            Model.highestId += 1
            b = Apred(Model.highestId, self, self.learningRate, self.discountFactor, self.explorationRate)
            self.schedule.add(b)
          
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(b, (x, y))

        self.datacollector = mesa.DataCollector(
            model_reporters={"prey num": Model.getPreyNum} | {"pred num": Model.getPredNum} |{"Apred num": Model.getAPredNum} | {"pred Q": Model.getPredQ} | {"prey Q": Model.getPreyQ}
        )
    
    def mapGen(self, width, height, vis):
        noise = np.zeros((width, height))
        simplex = OpenSimplex(0)

        for y in range(0, height):
            for x in range(0, width):
                noise[x][y] = int(10 * abs((simplex.noise2(x , y))))
        
        """ if vis is True:
            plt.imshow(noise, cmap='hot', interpolation='nearest')
            plt.show() """

        return noise


    def step(self):
        self.datacollector.collect(self)
        if len(Model.agentsToDie) > 0:
            self.kill()
        if len(Model.agentsToAdd) > 0:
            self.born()
        self.schedule.step()

    def getPreyNum(model):
        return model.num_prey
    
    
    def getPredNum(model):
        return model.num_pred
    
    def getAPredNum(model):
        return model.num_Apred
    
    def getPreyQ(model):
        return preyQTable
    
    
    def getPredQ(model):
        return predQTable
    
    def kill(self):
        for x in Model.agentsToDie:
            self.grid.remove_agent(x)
            self.schedule.remove(x)
            if x.type == 0:
                Model.num_prey -= 1
            elif x.type == 1:
                Model.num_pred -= 1
            elif x.type == 2:
                Model.num_Apred -= 1
        Model.agentsToDie.clear()

    def born(self):
        for y in Model.agentsToAdd:
            self.schedule.add(y)
            self.grid.place_agent(y, y.pos)
            if y.type == 0:
                    Model.num_prey += 1
            elif y.type == 1:
                Model.num_pred += 1
            elif y.type == 2:
                Model.num_Apred += 1
        Model.agentsToAdd.clear()

  

    

class Agent(mesa.Agent):
    def __init__(self, unique_id, model, learningRate, discountFactor, explorationRate):
       super().__init__(unique_id, model)
       self.hunger = 100
       self.lifespan = random.randrange(10, 50)
       self.preg = False
       self.pregCount = -1
       #size of action space - possible moves
       self.actionSize = 9
       
      
       #size of state space number of state variables and possible values for each variables
       self.stateSize = 2187
       self.currentState = []
       self.action = 0
       self.reward = 0


    def getState(self):
         state = [0, 0, 0, 0, 0, 0, 0, 0, 0]
         possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )

         for y in range(8):
            x = self.model.grid.get_cell_list_contents([possible_steps[y]])
            if len(x) > 0:
                for z in x:
                    if z.type == 1:
                        state[y] = 2
                        break
                    elif z.type == 0:
                        state[y] = 1
                        break
                    elif z.type == 2:
                        state[y] = 3
                        break
            state[8] = Model.resources[self.pos[0]][self.pos[1]]
         return state   


    def move(self, direction):
        if direction == 4:
            #right
            new_position = (self.pos[0] + 1, self.pos[1])
        elif direction == 1:
            #up
            new_position = (self.pos[0], self.pos[1] + 1)
        elif direction == 3:
            #left
            new_position = (self.pos[0] - 1, self.pos[1])
        elif direction == 6:
            #down
            new_position = (self.pos[0], self.pos[1] - 1)
        elif direction == 2:
             #up right
             new_position = (self.pos[0] + 1, self.pos[1] + 1)
        elif direction == 7:
             #down right
             new_position = (self.pos[0] + 1, self.pos[1] - 1)
        elif direction == 0:
             #up left
             new_position = (self.pos[0] - 1, self.pos[1] + 1)
        elif direction == 5:
             #down left
             new_position = (self.pos[0] - 1, self.pos[1] - 1)
        elif direction == 8:
             new_position = self.pos
       
        self.model.grid.move_agent(self, new_position)

    def perfomAction(self, action):
        self.move(action)

    def reproduce(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        if len(cellmates) > 1:
            other = self.random.choice(cellmates)
            if other.type == self.type and other.hunger >= 100: 
                self.preg = True
                self.pregCount = 10
                self.reward += 100
                self.hunger -= 50
    
    def birth(self):
        Model.highestId += 1
        if self.type == 0:
            b = prey(Model.highestId, self.model, self.learningRate, self.discountFactor, self.explorationRate)
        elif self.type == 1:
            b = pred(Model.highestId, self.model, self.learningRate, self.discountFactor, self.explorationRate)
        elif self.type == 2:
             b = Apred(Model.highestId, self.model, self.learningRate, self.discountFactor, self.explorationRate)
        b.pos = self.pos
        Model.agentsToAdd.append(b)
        

    def age(self, ageRate):
        if self.lifespan == 0:
            self.hunger = 0
        else:
            self.lifespan -= ageRate

    def eat(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        if len(cellmates) > 1:
            other = self.random.choice(cellmates)
            if self.type == 1 and other.type == 0 or self.type == 2 and other.type == 1:   
                if random.randrange(0, 10) < Model.resources[other.pos[0]][other.pos[1]]:
                    other.hunger = 0
                    self.hunger = 100
                    self.reward = 50

class Apred(Agent):
    def __init__(self, unique_id, model, learningRate, discountFactor, explorationRate):
       super().__init__(unique_id, model, learningRate, discountFactor, explorationRate)
       self.type = 2
       #rate at which new qvalues overwrite old ones
       self.learningRate = learningRate
        #imediate vs long term rewards
       self.discountFactor = discountFactor
       #exploration rate
       self.explorationRate = explorationRate
    
    def step(self):
        self.age(2)
        self.hunger -= 1
        self.currentState = self.observeState()
        if self.hunger == 0:
            Model.agentsToDie.append(self)
            self.reward = -20
        else:
           self.action = self.selectAction(self.currentState)
           self.perfomAction(self.action)
         
           self.eat()
           if self.preg == True:
                if self.pregCount == 0:
                   self.birth()
                   self.pregCount = -1
                   self.preg = False
                else: 
                    self.pregCount -= 1 
           else:
            self.reproduce() 

    def step2(self):
        newState = self.observeState()
        self.updateQtable(self.currentState, self.action, self.reward, newState)
    
    def observeState(self):
        currentState = tuple(self.getState())
        
        if currentState not in ApredQTable:
            ApredQTable[currentState] = [0] * self.actionSize
        return currentState
    
    def selectAction(self, currentState):
        if random.random() < self.explorationRate:
            action = random.randrange(self.actionSize)
        else:
            action = max(range(self.actionSize), key = lambda x: ApredQTable[currentState][x])
        return action   
    
    def updateQtable(self, state, action, reward, nextState):
        nextAction = self.selectAction(nextState)
        TD_error = reward + self.discountFactor * ApredQTable[nextState][nextAction] - ApredQTable[state][action]
        ApredQTable[state][action] += self.learningRate * TD_error
        self.reward = 0
         

class pred(Agent):
   
    def __init__(self, unique_id, model, learningRate, discountFactor, explorationRate):
       super().__init__(unique_id, model, learningRate, discountFactor, explorationRate)
       self.type = 1
       #rate at which new qvalues overwrite old ones
       self.learningRate = learningRate
        #imediate vs long term rewards
       self.discountFactor = discountFactor
       #exploration rate
       self.explorationRate = explorationRate
   

    def step(self):
        self.age(1)
        self.hunger -= 6
        self.currentState = self.observeState()
        if self.hunger == 0:
            Model.agentsToDie.append(self)
            self.reward = -20
        else:
           self.action = self.selectAction(self.currentState)
           self.perfomAction(self.action)
           if self.hunger < 200:
            self.eat()
           if self.preg == True:
                if self.pregCount == 0:
                   self.birth()
                   self.pregCount = -1
                   self.preg = False
                else: 
                    self.pregCount -= 1 
           else:
            self.reproduce()
    
      
    def step2(self):
        newState = self.observeState()
        self.updateQtable(self.currentState, self.action, self.reward, newState)
       

    def observeState(self):
        currentState = tuple(self.getState())
        
        if currentState not in predQTable:
            predQTable[currentState] = [0] * self.actionSize
        return currentState
    
    def selectAction(self, currentState):
        if random.random() < self.explorationRate:
            action = random.randrange(self.actionSize)
        else:
            action = max(range(self.actionSize), key = lambda x: predQTable[currentState][x])
        return action   
    
    def updateQtable(self, state, action, reward, nextState):
        nextAction = self.selectAction(nextState)
        TD_error = reward + self.discountFactor * predQTable[nextState][nextAction] - predQTable[state][action]
        predQTable[state][action] += self.learningRate * TD_error
        self.reward = 0



 
    
   



class prey(Agent):

    def __init__(self, unique_id, model, learningRate, discountFactor, explorationRate):
       super().__init__(unique_id, model, learningRate, discountFactor, explorationRate)
       self.type = 0
       #rate at which new qvalues overwrite old ones
       self.learningRate = learningRate
        #imediate vs long term rewards
       self.discountFactor = discountFactor
       #exploration rate
       self.explorationRate = explorationRate

    # action step
    def step(self):
        self.age(15)
        self.currentState = self.observeState()
        if self.hunger == 0:
            Model.agentsToDie.append(self)
            self.reward = -20
        else:
           self.graze()
           self.action = self.selectAction(self.currentState)
           self.perfomAction(self.action)
           if self.preg == True:
                if self.pregCount == 0:
                   self.birth()
                   self.pregCount = -1
                   self.preg = False
                else: 
                    self.pregCount -= 1 
           else:
               if self.hunger >= 200:
                   self.reproduce()
         
    # evaluation step
    def step2(self):
        newState = self.observeState()
        self.updateQtable(self.currentState, self.action, self.reward, newState)
    
    def observeState(self):
        currentState = tuple(self.getState())
        
        if currentState not in preyQTable:
            preyQTable[currentState] = [0] * self.actionSize
        return currentState
    
    def selectAction(self, currentState):
        if random.random() < self.explorationRate:
            action = random.randrange(self.actionSize)
        else:
            action = max(range(self.actionSize), key = lambda x:  preyQTable[currentState][x])
        return action   
    
    def updateQtable(self, state, action, reward, nextState):
        nextAction = self.selectAction(nextState)
        TD_error = reward + self.discountFactor *  preyQTable[nextState][nextAction] -  preyQTable[state][action]
        preyQTable[state][action] += self.learningRate * TD_error
        self.reward = 0
    
    def graze(self):
        a = Model.resources[self.pos[0]][self.pos[1]]
        self.reward += a
        self.hunger += a

        
       

 

            
       
         
                
          


