from environmentModel import *
import mesa


params = {
    "N":100, 
    "M":20, 
    "O": 2, 
    "width":20,
    "height":20,
    "learningRate": [0.1, 0.5, 0.9],
    "discountFactor": [0.1, 0.5, 0.9], 
    "explorationRate": [0.1, 0.5, 0.9],
    "vis": False
}

results = mesa.batch_run(
    Model,
    parameters=params,
    iterations=5,
    max_steps=100,
    number_processes=1,
    data_collection_period=1,
    display_progress=True,
) 



def pred_portrayal(Agent):
    portrayal = {"Shape": "circle",
                 "Filled": "true",
                 "r": 0.5}
    
    if Agent.type == 1:
        portrayal["Color"] = "red"
        portrayal["Layer"] = 0

    elif Agent.type == 0:
        portrayal["Color"] = "grey"
        portrayal["Layer"] = 1
        portrayal["r"] = 0.2

    elif Agent.type == 2:
        portrayal["Color"] = "blue"
        portrayal["Layer"] = 1
        portrayal["r"] = 1

    return portrayal

chart = mesa.visualization.ChartModule([{"Label": "prey num",
                      "Color": "Black"}, {"Label": "pred num", "Color": "Red"},
                       {"Label": "Apred num", "Color": "Blue"}],
                    data_collector_name='datacollector')

grid = mesa.visualization.CanvasGrid(pred_portrayal, 20, 20, 500, 500)

params = {
    "N":50, 
    "M":10, 
    "O":3,
    "width":20,
    "height":20,
    "learningRate": 0.5,
    "discountFactor": 0.5, 
    "explorationRate": 0.5,
    "vis":True
}

server = mesa.visualization.ModularServer(Model,
                       [grid, chart],
                       "Model",
                      params)


server.port = 8521 
server.launch()