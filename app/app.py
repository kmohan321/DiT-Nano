import os
import sys
import torch
import streamlit as st
from torchvision.utils import make_grid

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from inference import (
    sample_flow_matching,
    unnormalize_to_zero_one,
    clean_state_dict,
)
from model import DiT
from config import config


def load_imagenet_classes(path="app/imagenet_classes.txt"):
    with open(path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    assert len(classes) == 1000
    return classes

imagenet_classes = load_imagenet_classes()

@st.cache_resource
def load_model(checkpoint_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, map_location=device)

    model = DiT(config).to(device)
    model.load_state_dict(clean_state_dict(ckpt["model_state_dict"]))
    model.eval()

    return model, device



st.set_page_config(page_title="DiT Flow Matching", layout="wide")
st.title(" DiT Flow-Matching Image Generator")

with st.sidebar:
    st.header("Generation Settings")

    checkpoint_path = st.text_input(
        "Checkpoint Path",
        "Model-weights/sit_nano_epoch_36.pt",
    )

    num_images = st.slider("Batch Size", 1, 64, 16)
    steps = st.slider("Sampling Steps", 5, 100, 30)

    random_sampling = st.checkbox(
        "Random classes (ignore conditioning)",
        value=False,
    )

    if not random_sampling:
        class_name = st.selectbox("ImageNet Class", imagenet_classes)
        class_id = imagenet_classes.index(class_name)
    else:
        class_name = "Random"

    generate = st.button("🚀 Generate")


if generate:
    model, device = load_model(checkpoint_path)


    if random_sampling:
        labels = torch.randint(
            0,
            len(imagenet_classes),
            (num_images,),
            device=device,
        )
    else:
        labels = torch.full(
            (num_images,),
            class_id,
            device=device,
            dtype=torch.long,
        )

    z = torch.randn(num_images, 3, 64, 64, device=device)


    live_title = st.empty()
    live_grid = st.empty()
    progress_slot = st.empty()
    step_text = st.empty()

    live_title.subheader("Live Denoising (Full Batch)")
    progress = progress_slot.progress(0)

    def on_step(step, total, x):
        progress.progress(step / total)
        step_text.text(f"Step {step} / {total}")

        imgs = unnormalize_to_zero_one(x).detach().cpu()
        grid = make_grid(imgs, nrow=int(len(imgs) ** 0.5))

        live_grid.image(
            grid.permute(1, 2, 0).numpy(),
            caption=f"Denoising step {step}",
            use_container_width=True,
        )

    final = sample_flow_matching(
        model,
        z,
        labels,
        steps=steps,
        device=device,
        step_callback=on_step,
    )


    live_title.empty()
    live_grid.empty()
    progress_slot.empty()
    step_text.empty()

    final = unnormalize_to_zero_one(final).cpu()

    st.subheader("Final Generated Images")

    if random_sampling:
        cols = st.columns(4)
        for i, img in enumerate(final):
            cname = imagenet_classes[int(labels[i].item())]
            with cols[i % 4]:
                st.markdown(
                    f"<p style='text-align:center; font-weight:600'>{cname}</p>",
                    unsafe_allow_html=True,
                )
                st.image(
                    img.permute(1, 2, 0).numpy(),
                    use_container_width=True,
                )
    else:
        grid = make_grid(final, nrow=int(num_images ** 0.5))
        st.image(
            grid.permute(1, 2, 0).numpy(),
            use_container_width=True,
        )
