import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import random
pip install torchsummary


class DeepLabV3Inference:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        self.model = torch.hub.load('pytorch/vision:v0.15.2', 'deeplabv3_resnet101', pretrained=True)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image):
        """
        Args:
            image: PIL.Image or str (path to image)
        Returns:
            pred_mask: numpy array (H, W) of class indices
            color_mask: PIL Image of colored segmentation mask
            blended: PIL Image of blended overlay of original image and mask
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise TypeError("Input must be a PIL.Image or image path string.")

        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)['out']
            output = torch.nn.functional.interpolate(output, size=image.size[::-1], mode='bilinear', align_corners=False)
            pred_mask = output.argmax(1).squeeze().cpu().numpy()

        unique_classes = np.unique(pred_mask)

        random.seed(42)
        colormap = {cls: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for cls in unique_classes}

        color_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
        for cls, color in colormap.items():
            color_mask[pred_mask == cls] = color
        color_mask_img = Image.fromarray(color_mask)

        image_rgba = image.convert("RGBA")
        color_mask_rgba = color_mask_img.convert("RGBA")
        blended = Image.blend(image_rgba, color_mask_rgba, alpha=0.5)

        return pred_mask, color_mask_img, blended

    def save_results(self, color_mask_img, blended_img, mask_path, blended_path):
        color_mask_img.save(mask_path)
        blended_img.save(blended_path)
    
    def model_summary(self, input_size=(3, 224, 224)):
        """
        Prints a layer-wise summary of the model with parameters.
        Args:
            input_size (tuple): Input tensor size (C, H, W)
        """
        print(f"Model Summary for input size {input_size}:")
        summary(self.model, input_size=input_size)


if __name__ == "__main__":
