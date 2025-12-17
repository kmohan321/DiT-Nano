from torchvision import transforms
from datasets import load_dataset
from torch.utils.data import DataLoader
from config import config

dataset = load_dataset("benjamin-paine/imagenet-1k-64x64")

transform = transforms.Compose([
    transforms.ToTensor(), # [0, 255] -> [0.0, 1.0]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # [0.0, 1.0] -> [-1.0, 1.0]
])

def transform_batch(batch):
    # .convert("RGB") is crucial because some ImageNet images are Grayscale (1 channel)
    batch['image'] = [transform(img.convert("RGB")) for img in batch['image']]
    return batch

dataset = dataset.with_transform(transform_batch)
train_loader = DataLoader(
    dataset['train'], 
    batch_size=config['batch_size'], 
    shuffle=True, 
    pin_memory=True
)