import torch
from skimage import io
import torchvision.transforms as transforms
from model_train import Model

# paths to two images one being metallic and one not
non_metalic_path = r"*path to non metallic image*"
metallic_path = r"*path to metallic image*"

# transform the images to be readable by neural net
my_transforms = transforms.Compose([transforms.ToPILImage(), transforms.Resize((255, 255)),
                                    transforms.RandomCrop(244),
                                    transforms.ToTensor()])
images = [non_metalic_path, metallic_path]
transformed_images = []
for image in images:
    image_ten = io.imread(image)
    image_ten = my_transforms(image_ten)
    if image_ten.shape[0] < 3:
        fill = 3 - image_ten.shape[0]
        pad = torch.zeros(fill, 244, 244)
        transformed_images.append(torch.cat([image_ten, pad], dim=0))
    elif image_ten.shape[0] > 3:
        transformed_images.append(image_ten[:3, :, :])
    else:
        transformed_images.append(image_ten)

# get pre-trained model created by model_train.py
model = torch.load(r"*path to best model*")

# switch to evaluate mode
model.eval()

# get predictions from pretrained neural net
preds = []
for image in transformed_images:
    a = my_transforms(image).unsqueeze(dim=0)
    preds.append(model(a))
print(preds)
# non metallic image tensor([[0.0002]], grad_fn=<SigmoidBackward>)
# metallic_image tensor([[0.9861]], grad_fn=<SigmoidBackward>)
