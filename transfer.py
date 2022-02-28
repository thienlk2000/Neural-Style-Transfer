import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("content", help='Content Image')
parser.add_argument("style", help='Style Image')
parser.add_argument("save_name", help='save_name')



args = parser.parse_args()

content = args.content
style = args.style
save_name = args.save_name

class VGG(nn.Module):
  def __init__(self):
    super(VGG, self).__init__()
    self.layer = [0, 5, 10, 19, 28]

    self.model = models.vgg19(pretrained=True).features[:29]

  def forward(self, x):
    features_map = []
    for i,layer in enumerate(self.model):
      x = layer(x)
      if i in self.layer:
        features_map.append(x)
    return features_map

img_size = 356
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loader = T.Compose([
                    T.Resize(img_size),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_image(img_path):
  img = Image.open(img_path)
  img = img.convert('RGB')
  img = loader(img).unsqueeze(0)
  return img.to(device)



def plot_image(img_tensor,save_name):
  with torch.no_grad():
    for i, (img,name) in enumerate(zip(img_tensor, ['Content', 'Style', 'Generate'])):
        if len(img_tensor) == 1:
          name = 'Generated'
        plt.subplot(1,len(img_tensor),i+1)
        img = img.squeeze()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(device)
        img = img*std + mean
        img = img.clip(0, 1)
        img = T.ToPILImage()(img)
        plt.imshow(img)
        plt.axis('off')
        plt.title(name)
        plt.tight_layout()
  
  plt.savefig(save_name)
  plt.show()


content_img = load_image(content)
style_img = load_image(style)
generated = content_img.clone().requires_grad_(True).to(device)
model = VGG().to(device).eval().requires_grad_(False)

total_step = 6000
learning_rate = 0.001
alpha = 1
beta = 0.01
optimizer = optim.Adam([generated], lr=learning_rate)

for step in range(total_step):
  generated_features = model(generated)
  original_img_features = model(content_img)
  style_features = model(style_img)

  style_loss = 0
  content_loss = torch.mean((generated_features[-1] - original_img_features[-1])**2)
  for gen_feature, style_feature in zip(generated_features, style_features):
    batch, channel, height, width = gen_feature.shape
    gen_Gram = gen_feature.view(channel, -1).mm(gen_feature.view(channel, -1).T)
    style_Gram = style_feature.view(channel, -1).mm(style_feature.view(channel, -1).T)
    style_loss += torch.mean((gen_Gram - style_Gram)**2)
  
  loss = alpha*content_loss + beta*style_loss
  if step % 200 == 0:
    print(f"Step:{step}\tContent Loss:{content_loss.item()}\tStyle Loss:{style_loss.item()}\tTotal Loss:{loss.item()}")
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

plot_image([content_img, style_img, generated], save_name)
plot_image([generated], 'result/'+save_name)