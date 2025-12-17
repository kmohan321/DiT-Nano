import os
import torch
import torch.nn as nn
from model import DiT
from data import train_loader
import wandb
from tqdm import tqdm
from torchvision.utils import make_grid
from config import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def sample_flow_matching(model, z, labels, steps=25, device='cuda'):
    """
    z: Random noise [B, C, H, W]
    labels: Class labels [B]
    steps: Number of integration steps (20-50 is usually enough)
    """
    model.eval()
    b = z.shape[0]
    dt = 1.0 / steps
    x = z.clone()  
    
    for i in range(steps):
        t_value = i / steps
        t = torch.full((b,), t_value, device=device, dtype=torch.float)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            v_pred = model(x, t, labels)
        x = x + v_pred * dt
        
    model.train()
    return x
    
class ImageLogger:
    def __init__(self, batch_size=16):
        self.fixed_noise = torch.randn(batch_size, 3, 64, 64)
        self.fixed_labels = torch.randint(0, 1000, (batch_size,))
    
    def log_images(self, model, device, epoch, wandb_run):
        model.eval()
        with torch.no_grad():
            z = self.fixed_noise.to(device)
            y = self.fixed_labels.to(device)
            generated = sample_flow_matching(model, z, y, steps=25, device=device)
            generated = (generated * 0.5 + 0.5).clamp(0, 1)
            grid = make_grid(generated, nrow=4)
            wandb_run.log({"Generated_Epoch": wandb.Image(grid), "epoch": epoch})
        model.train()
        
        
model = DiT(config).to(device)
model = torch.compile(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
loss_fn = nn.MSELoss()

use_amp = config['use_amp']
scaler = torch.amp.GradScaler(device, enabled=use_amp)
epochs = config["epochs"]
cfm_weight = config["cfm_weight"]
image_logger = ImageLogger()


if config['resume']:
    
    RUN_ID = "wandb run id"
    CHECKPOINT_PATH = "Path of the checkpoint"
    
    wandb_run = wandb.init(
        project="SiT-Nano-ImageNet64",
        id=RUN_ID,
        resume="must",
        config=config
    )
    
    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        print(f" Found checkpoint at {CHECKPOINT_PATH}. Loading...")
        
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        
        print(f" Resumed successfully from Epoch {start_epoch}")
    else:
        print(" No checkpoint found. Starting from scratch!")
    
else:
    wandb_run = wandb.init(project="SiT-Nano-ImageNet64", config=config)


model.train()
for epoch in range(start_epoch, epochs):
    
    tqdm_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    
    for batch in tqdm_bar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        b = images.shape[0]
    
        t = torch.rand(b, device=device)
        x0 = torch.randn_like(images)
        
        t_reshaped = t.view(-1, 1, 1, 1)
        xt = (1 - t_reshaped) * x0 + t_reshaped * images
        
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            
            v_pred = model(xt, t, labels)
            v_target = images - x0
            flow_loss = loss_fn(v_pred, v_target)
            
            contrastive_target = torch.roll(v_target, shifts=1, dims=0)
            contrastive_loss = loss_fn(v_pred, contrastive_target)
            
            loss = flow_loss + cfm_weight * contrastive_loss
    
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
        current_loss = loss.item()
        wandb_run.log({
            "total_loss": current_loss, 
            "flow_loss": flow_loss.item(), 
            "contrastive_loss": contrastive_loss.item(),
            "epoch": epoch
        })
        
        tqdm_bar.set_postfix(loss=f"{current_loss:.4f}")

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': current_loss,
    }, f"Model_weights/sit_nano_epoch_{epoch+1}.pt")

    image_logger.log_images(model, device, epoch, wandb_run)

wandb_run.finish()