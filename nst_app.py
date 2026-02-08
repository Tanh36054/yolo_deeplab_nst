import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

st.set_page_config(page_title="Neural Style Transfer Demo", layout="wide")
st.title("🎨 Neural Style Transfer (NST)")

# --- Step 1: Tải ảnh ---
content_file = st.file_uploader("Tải ảnh nội dung (Content Image)", type=["jpg", "jpeg", "png"], key="content")
style_file = st.file_uploader("Tải ảnh phong cách (Style Image)", type=["jpg", "jpeg", "png"], key="style")

if content_file and style_file:
    content_image = Image.open(content_file).convert("RGB")
    style_image = Image.open(style_file).convert("RGB")

    st.image([content_image, style_image], caption=["Ảnh nội dung", "Ảnh phong cách"], use_container_width=True)

    # --- Step 2: Chuẩn bị model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    imsize = 256
    loader = transforms.Compose([transforms.Resize((imsize, imsize)), transforms.ToTensor()])

    def image_loader(image):
        image = loader(image).unsqueeze(0)
        return image.to(device, torch.float)

    content_img = image_loader(content_image)
    style_img = image_loader(style_image)

    cnn = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()

    # --- Step 3: Xây dựng model phụ ---
    def gram_matrix(input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

    class ContentLoss(nn.Module):
        def __init__(self, target):
            super().__init__()
            self.target = target.detach()

        def forward(self, input):
            self.loss = nn.functional.mse_loss(input, self.target)
            return input

    class StyleLoss(nn.Module):
        def __init__(self, target_feature):
            super().__init__()
            self.target = gram_matrix(target_feature).detach()

        def forward(self, input):
            G = gram_matrix(input)
            self.loss = nn.functional.mse_loss(G, self.target)
            return input

    # --- Step 4: Chèn các lớp content & style loss ---
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    normalization = nn.Sequential(transforms.Normalize(mean=cnn_normalization_mean, std=cnn_normalization_std))

    content_layers = ["conv_4"]
    style_layers = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"
        else:
            raise RuntimeError(f"Unrecognized layer: {layer.__class__.__name__}")

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], (ContentLoss, StyleLoss)):
            break
    model = model[: i + 1]

    # --- Step 5: Chạy tối ưu ---
    input_img = content_img.clone()
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    num_steps = 200
    style_weight = 1000000
    content_weight = 1

    st.write("⏳ Đang chạy NST, vui lòng chờ...")

    run = [0]
    while run[0] <= num_steps:

        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            loss = style_weight * style_score + content_weight * content_score
            loss.backward()

            run[0] += 1
            return style_score + content_score

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)

    # --- Step 6: Hiển thị ảnh kết quả ---
    unloader = transforms.ToPILImage()
    output_img = input_img.cpu().clone().squeeze(0)
    output_img = unloader(output_img)

    st.image(output_img, caption="Ảnh kết quả NST 🎨", use_container_width=True)
