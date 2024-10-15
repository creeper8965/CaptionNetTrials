from transformers import SiglipProcessor, SiglipModel
from PIL import Image
import torch

modelName = 'SigLip_7500'
Processor = SiglipProcessor.from_pretrained(modelName)
model = SiglipModel.from_pretrained(modelName).eval()

image = Image.open('flickr8k/images/1000268201_693b08cb0e.jpg')
ClassTexts = ['A child','aeroplane','a child going into a house','a boat']

inputs = Processor(images=image, text=ClassTexts, padding='max_length',max_length=64,return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs).logits_per_image

print(torch.sigmoid(outputs))

                                   
