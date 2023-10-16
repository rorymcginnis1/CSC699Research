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
img_dir = "/Users/rorymcginnis/Documents/SFSU/Summer2023/Research/test/CSC699Research/dataset"
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
data = []
predictions=[]

truth_file = open('truth.json',"r")
true= json.loads(truth_file.read())
class_file = open('classes.json',"r")
classes= json.loads(class_file.read())[0]
        
#alexnet, mobilenet_v3_small shufflenet_v2_x1_0 squeezenet1_0 mnasnet0_5 squeezenet1_1

CNN = models.alexnet(pretrained=True)

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
names=[]
preds=[]
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

    names.append(string2)
    if(string==(string2)):

        predictions.append(1)
    else:
        predictions.append(0)

    strin="img"
    strin+=""+str(i)+""
    strin+=".jpg"

    for idx in indices[0]:
        per.append(percentage[idx].item())
    totPercent=[]
    print(max(per))
    preds.append(max(per))
    #for l in range(len(per)):
    for l in range(150):
        totPercent.append(per[l])
    
json_data = json.dumps(values)

with open('alexnet.json', 'w') as f:
    f.write(json_data)

ans = 0
for i in range(len(predictions)):
    if(predictions[i]==1):
        ans+=1

print(ans)
print(len(predictions))
print(len(preds))
import json
import random

modified_data = {
    "data": []
}


for i in range(len(predictions)):
    image_path = data[i]
    image_filename = os.path.basename(image_path)
    image_name = image_filename[-10:]
    true_labels = ", ".join(true[i])
    accuracy = False
    if (true_labels==names[i]):
        accuracy=True
    entry = {
        "image_name": image_name,
        "true_label": true_labels,
        "predicted_label_model_Alexnet": f" {names[i]}",
        "confidence_model_AlexNet_result": preds[i]/100,
        "Alex_Net_accuracy":accuracy,
        
    }



    modified_data["data"].append(entry)


output_json_file = "Alex_Net_values.json"

with open(output_json_file, "w") as json_file:
    json.dump(modified_data, json_file, indent=4)

print(f"The modified data has been written to '{output_json_file}'.")
