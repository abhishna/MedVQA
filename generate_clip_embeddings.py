from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPVisionModel

# load MedCLIP-ResNet50
model = MedCLIPModel(vision_cls=MedCLIPVisionModel)
model.from_pretrained()

# load MedCLIP-ViT
model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
model.from_pretrained()

from medclip import MedCLIPModel, MedCLIPVisionModelViT
from medclip import MedCLIPProcessor
from PIL import Image

model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)

img = {}
qs = {}
ans = {}
import string

def preprocess_text(text):
    """
        Converts a string to lower case, removes punctuations.
    """
    text_token_list = text.strip().split(',')
    text  = ' '.join(text_token_list)

    # Remove punctuations
    table = str.maketrans('', '', string.punctuation)
    words = text.strip().split()
    words = [w.translate(table) for w in words]

    # Set to lowercase & drop empty strings
    words = [word.lower() for word in words if word != '' and word != 's']
    
    text  = ' '.join(words)
    return text

# prepare for the demo image and texts
import json

processor = MedCLIPProcessor()
model.from_pretrained()
model.cuda()
c=0
c1=c
f = json.loads(open('datasets/train.json').read())
f=f[c:]
for i in f:
  image = Image.open('datasets/Slake1.0/imgs/'+i['img_name'])
  q = i['question']
  a= i['answer']

  inputs = processor(
    text=[q,a], 
    images=image, 
    return_tensors="pt", 
    padding=True
    )


  outputs = model(**inputs)
  img[i['img_name']] = outputs['img_embeds']
  qs[preprocess_text(q)] = outputs['text_embeds'][0]
  ans[preprocess_text(a)] = outputs['text_embeds'][1]
  c+=1
  if(c==(c1+1500)):
    break
  print(c)


import pickle

filehandler1 = open("train_image_embeddings"+str(c)+".pkl","wb")
filehandler2 = open("train_question_embeddings"+str(c)+".pkl","wb")
filehandler3 = open("train_answer_embeddings"+str(c)+".pkl","wb")
pickle.dump(img, filehandler1)
pickle.dump(qs, filehandler2)
pickle.dump(ans, filehandler3)
