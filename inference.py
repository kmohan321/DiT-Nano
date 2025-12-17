import torch
import os
from tqdm import tqdm
from model import DiT
from config import config
from torchvision.utils import save_image

def unnormalize_to_zero_one(t):
    """Converts [-1, 1] tensors back to [0, 1] for saving."""
    return t.clamp(-1, 1) * 0.5 + 0.5

@torch.no_grad()
def sample_flow_matching(model, z, labels, steps=20, device='cuda'):
    """
    Generates images from noise using Euler integration.
    
    Args:
        z: Initial noise tensor [B, C, H, W]
        labels: Class labels [B]
        steps: Number of integration steps (higher = better quality, slower)
        time_shift_s: Time shifting factor (optional, set > 1.0 to focus on structure)
    """
    model.eval()
    b = z.shape[0]
    dt = 1.0 / steps
    x = z.clone()
    
    print(f" Generating {b} images for labels: {labels.tolist()}")
    
    for i in tqdm(range(steps), desc="Sampling"):
      
        t_value = i / steps
        t = torch.full((b,), t_value, device=device, dtype=torch.float)
        
        with torch.autocast(device_type="cuda", dtype=torch.float32): 
            v_pred = model(x, t, labels)
        x = x + v_pred * dt
        
    return x

def clean_state_dict(state_dict):
    """
    Removes the '_orig_mod.' prefix added by torch.compile()
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        if new_key.startswith("_orig_mod."):
            new_key = new_key[10:] # Remove '_orig_mod.'
        
        new_state_dict[new_key] = v
    return new_state_dict

def run_inference(
    checkpoint_path, 
    output_dir="generated_images", 
    num_images=16, 
    specific_class=None, 
    steps=20
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Running on device: {device}")

    print(f" Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = DiT(config).to(device)
    state_dict = clean_state_dict(checkpoint['model_state_dict'])
    model.load_state_dict(state_dict)
    print(" Model loaded successfully.")
    
    os.makedirs(output_dir, exist_ok=True)

    if specific_class is not None:
        labels = torch.full((num_images,), specific_class, device=device, dtype=torch.long)
    else:
        labels = torch.randint(0, config['num_classes'], (num_images,), device=device)
        
    z = torch.randn(num_images, 3, 64, 64, device=device)
    
    generated_images = sample_flow_matching(model, z, labels, steps=steps, device=device)

    print(f" Saving images to '{output_dir}'...")
    generated_images = unnormalize_to_zero_one(generated_images)
    
    # for idx, img in enumerate(generated_images):
    #     class_id = labels[idx].item()
    #     save_path = os.path.join(output_dir, f"class_{class_id:04d}_sample_{idx:03d}.png")
    #     save_image(img, save_path)
    
    idx = len(os.listdir(output_dir))
    grid_path = os.path.join(output_dir, f"grid_view_{idx}.png")
    save_image(generated_images, grid_path, nrow=8)
    print(f" Done! Saved grid to {grid_path}")

if __name__ == "__main__":
    CHECKPOINT = "Model-weights/sit_nano_epoch_36.pt"
    OUTPUT_FOLDER = "inference_results"

    run_inference(CHECKPOINT, OUTPUT_FOLDER, num_images=64, steps=30)
    