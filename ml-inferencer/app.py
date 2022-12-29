import json
import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.hidden_layer_1 = nn.Linear(self.in_dim, 20) # input to first hidden layer
        self.hidden_layer_2 = nn.Linear(20, 10)
        
        self.output_layer = nn.Linear(10, self.out_dim)
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        x = self.hidden_layer_1(x)
        x = self.activation(x)
        
        x = self.hidden_layer_2(x)
        x = self.activation(x)
        
        y = self.output_layer(x)
        y = self.activation(y)
        
        return y

def lambda_handler(event, context):
    # { data: [ x1...x30 ] }
    body = json.loads(json.loads(event["body"])["body"])
    payload = body["data"]
    print("payload:")
    print(payload)
    print(type(payload))

    # input data
    x = torch.Tensor([payload])

    model = NeuralNetwork(30, 2)

    state = torch.load("model/model.pth")

    model.load_state_dict(state["state_dict"])

    predictions = model.forward(x).detach().cpu().numpy()[0].tolist()

    res = {
        "predictions": predictions  
    }

    print(predictions)

    return {
        "statusCode": 200,
        "body": json.dumps(res),
    }
