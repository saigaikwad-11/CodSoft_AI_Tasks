import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, AutoTokenizer

class ImageCaptioner:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )

    def predict(self, image: Image.Image) -> str:
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Direct PIL image resizing aur tensor conversion bina numpy ke conflicts ke
        image = image.resize((224, 224))
        
        # Manual tensor conversion taaki torchvision ka numpy wala bug bypass ho jaye
        import torchvision.transforms.functional as TF
        tensor_img = TF.to_tensor(image)
        normalized_img = TF.normalize(tensor_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        pixel_values = normalized_img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                pixel_values,
                max_new_tokens=30,
                num_beams=4
            )

        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return preds[0].strip()