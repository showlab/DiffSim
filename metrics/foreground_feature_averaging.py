import torch
import torchvision
from PIL import Image
import numpy as np
from carvekit.api.high import HiInterface


class ForegroundFeatureAveraging(torch.nn.Module):
    def __init__(self, device, carvekit_object_type="object"):
        """
        :param device: string or torch.device object to run the model on.
        :param carvekit_object_type: object type for foreground segmentation. Can be "object" or "hairs-like".
        We find that "object" works well for most images in the CUTE dataset as well as vehicle ReID.
        """
        super().__init__()
        self.device = device
        self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.encoder.to(self.device)

        self.interface = HiInterface(object_type=carvekit_object_type,  # Can be "object" or "hairs-like".
                                     batch_size_seg=5,
                                     batch_size_matting=1,
                                     device=str(self.device),  # HIInterface requires a string device.
                                     seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
                                     matting_mask_size=2048,
                                     trimap_prob_threshold=231,
                                     trimap_dilation=30,
                                     trimap_erosion_iters=5,
                                     fp16=False)

    def preprocess(self, x_list):

        preprocessed_images = []

        for x in x_list:
            # width, height = x.size
            new_width = 336
            new_height = 336

            def _to_rgb(x):
                if x.mode != "RGB":
                    x = x.convert("RGB")
                return x

            preprocessed_image = torchvision.transforms.Compose([
                _to_rgb,
                torchvision.transforms.Resize((new_height, new_width), interpolation=Image.BICUBIC),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])(x)
            preprocessed_images.append(preprocessed_image)

        return torch.stack(preprocessed_images, dim=0).to(self.device)

    def get_foreground_mask(self, tensor_imgs):
        masks = []
        for tensor_img in tensor_imgs:
            tensor_img = tensor_img.detach().cpu()
            numpy_img_sum = tensor_img.sum(dim=0).numpy()
            min_value = np.min(numpy_img_sum)
            mask = ~(numpy_img_sum == min_value)
            mask = mask.astype(np.uint8)
            mask = Image.fromarray(mask * 255)
            resized_mask = mask.resize((24, 24), Image.BILINEAR)
            resized_mask_numpy = np.array(resized_mask)
            resized_mask_numpy = resized_mask_numpy / 255.0
            tensor_mask = torch.from_numpy(resized_mask_numpy.astype(np.float32))
            tensor_mask[tensor_mask > 0.5] = 1.0
            tensor_mask = tensor_mask.unsqueeze(0).long().to(self.device)
            if tensor_mask.sum() == 0:
                tensor_mask = torch.ones_like(tensor_mask)
            masks.append(tensor_mask)
        return torch.stack(masks, dim=0)

    def forward(self, variant, *x):
        """
        :param variant: either "Crop-Feat" or "Crop-Img". This determines whether foreground cropping is applied directly
        to the features ("Crop-Feat") or the images ("Crop-Img").
        :param x: Either (1) a single list/tensor of images, or (2) a single image, or (3) two lists of images of the
        same length, for which each pair of corresponding images will be compared.
        :return: If (1) or (2), the computed feature vectors for each image
        If (3), the cosine similarity between the two sets of feature vectors, which should have the same length
        as the input lists.
        """
        if len(x) == 1 and (isinstance(x[0], list) or isinstance(x[0], torch.Tensor)):
            return self.forward_single(x[0], variant)
        elif len(x) == 1:
            return self.forward_single([x[0]], variant)
        elif len(x) == 2:
            return torch.cosine_similarity(self.forward_single(x[0], variant)[0], self.forward_single(x[1], variant)[0], dim=0).cpu().item()
        else:
            raise ValueError("Invalid number of inputs, only 1 or 2 inputs are supported.")

    def forward_single(self, x_list, variant):

        with torch.no_grad():
            img_list = [np.array(self.interface([x])[0]) for x in x_list]
            for img in img_list:
                img[img[..., 3] == 0] = [0, 0, 0, 0]
            img_list = [Image.fromarray(img) for img in img_list]
            preprocessed_imgs = self.preprocess(img_list)
            masks = self.get_foreground_mask(preprocessed_imgs)
            if variant == "Crop-Feat":
                emb = self.encoder.forward_features(preprocessed_imgs)
            elif variant == "Crop-Img":
                emb = self.encoder.forward_features(self.preprocess(x_list))
            else:
                raise ValueError("Invalid variant, only Crop-Feat and Crop-Img are supported.")

            grid = emb["x_norm_patchtokens"].view(len(x_list), 24, 24, -1)

            return (grid * masks.permute(0, 2, 3, 1)).sum(dim=(1, 2)) / masks.sum(dim=(1, 2, 3)).unsqueeze(-1)
