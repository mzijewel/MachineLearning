import torch
from torch import nn,load
from torchvision.transforms import ToTensor
import numpy as np
import keras

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.model=nn.Sequential(
            nn.Conv2d(1,32,(3,3)),
            nn.ReLU(),
            nn.Conv2d(32,64,(3,3)),
            nn.ReLU(),
            nn.Conv2d(64,64,(3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6),10) # used 3 conv2d, so reduced 3x2 = 6

        )
    def forward(self,x):
        return self.model(x)


class Detect():
    def __init__(self):
        self.model=Model()
        self.model.load_state_dict(load('model_98.80.pt',map_location=torch.device('cpu')))
        self.model2=keras.models.load_model("m98.keras")
  

    def predict(self,img):
        img=img.convert("L").resize((28,28))
        img_tensor=ToTensor()(img).unsqueeze(0).to(torch.device('cpu'))
        output=self.model(img_tensor)
        probabilities=torch.nn.functional.softmax(output[0],dim=0)

        predicted=torch.argmax(probabilities).item()
        confidence=probabilities[predicted].item()
        return predicted,confidence*100
    
    def predict_tf(self,img):
        img=img.convert("L").resize((28,28))
        img=np.asarray(img)
        img=img.reshape(1,28*28)
        pr=self.model2.predict(img)
        p=np.argmax(pr)
        
        return p,0.0