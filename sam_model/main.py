import torch
import torchvision
import sys
import numpy as np
import matplotlib.pyplot as plt
import io
from io import BytesIO
from PIL import Image
import cv2,requests
from module.show_anns import show_anns


        
        
#url에서 파일 받아서 np배열로 변경        
image_url = 'https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcdBr13%2FbtsbDErQ7pd%2Fqk5fqpvFBUg3vsAdsdVOC0%2Fimg.png'
response = requests.get(image_url)       
if response.status_code ==200:
    image_data = np.frombuffer(response.content, np.uint8)
    img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img)
    


# sam 모델 로드하여 데이터 입력 
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint= "model/sam_vit_h_4b8939.pth"
model_type = 'vit_h'

device ="cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

masks = mask_generator.generate(img)

# 출력
print(len(masks))
print(masks[0].keys())
plt.figure(figsize=(20,20))
# plt.imshow(img)
show_anns(masks) #출력부
plt.axis('off')
plt.show()