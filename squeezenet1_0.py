from torchvision import models
import torch
from PIL import Image
from skimage import io, transform
from torchvision import transforms
import json
import os 
import glob 

values=[]
#replace with your own path to the dataset
img_dir = "/Users/rorymcginnis/Documents/SFSU/Summer2023/Research/dataset" 
data_path = os.path.join(img_dir,'*g') 
files = glob.glob(data_path) 
data = [] 
predictions=[]

truth_file = open('truth.json',"r")
true= json.loads(truth_file.read())
class_file = open('classes.json',"r")
classes= json.loads(class_file.read())[0]
        
#alexnet, mobilenet_v3_small shufflenet_v2_x1_0 squeezenet1_0 mnasnet0_5 squeezenet1_1

CNN = models.squeezenet1_0(pretrained=True)

images=[]

for f1 in files: 
    data.append(f1)
conv_image=[]

transform = transforms.Compose([       
        transforms.Resize(256),         
        transforms.CenterCrop(224),            
        transforms.ToTensor()   ,
        transforms.Normalize(
        mean = [.485, .456, .406],
        std= [.229, .224, .225])
        ])
for i in range(len(data)):
        img=Image.open(data[i])   
        img.resize((256,256))     
        img = transform(img)
        images.append(torch.unsqueeze(img,0))
        CNN.eval()

for i in range (len(images)):
    conv_image.append(CNN(images[i]))

for i in range (len(conv_image)):
    conv=conv_image[i]
    
    _, indices = torch.sort(conv, descending=True)  
    percentage = torch.nn.functional.softmax(conv, dim=1)[0] * 100
    per=[]
    
    name = str([(classes[idx]) for idx in indices[0][:1]])

    string = ""
    string2=""
    string+=true[i][0]
    for z in range(len(true[i])-1):
        string+=", "
        string+=true[i][z+1]

    for z in range(len(name)-4):
        string2+=name[z+2]

    if(string==(string2)):

        predictions.append(1)
    else:
        predictions.append(0)

    for idx in indices[0]:
        per.append(percentage[idx].item())
    totPercent=[]
    
    for l in range(len(per)):
        totPercent.append(per[l])
    values.append(totPercent)
    
json_data = json.dumps(values)

with open('squeezenet1_0.json', 'w') as f:
    f.write(json_data)

ans = 0
for i in range(len(predictions)):
    if(predictions[i]==1):
        ans+=1

print(ans)
print(len(predictions))
