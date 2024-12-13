# import resources
from PIL import Image

import torch
from torchvision import transforms, models
import torch.nn.functional as F

class vgg_gram:
    def __init__(self, device):
        self.device = device
        self.vgg = models.vgg19(pretrained=True).features.to(device)

        # freeze all VGG parameters since we're only optimizing the target image
        for param in self.vgg.parameters():
            param.requires_grad_(False)
    
    def load_image(self, img_path, img_size=512):
        image = Image.open(img_path).convert('RGB')
        
        in_transform = transforms.Compose([
                            transforms.Resize(img_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), 
                                                (0.229, 0.224, 0.225))])

        # discard the transparent, alpha channel (that's the :3) and add the batch dimension
        image = in_transform(image)[:3,:,:].unsqueeze(0).to(self.device)
        
        return image

    def get_features(self, image, model, layers=None):
        """ Run an image forward through a model and get the features for 
            a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
        """
        
        ## TODO: Complete mapping layer names of PyTorch's VGGNet to names from the paper
        ## Need the layers for the content and style representations of an image
        if layers is None:
            layers = {'28': 'conv5_1'}
            
        # features = 
        x = image
        # model._modules is a dictionary holding each module in the model
        for name, layer in model._modules.items():
            x = layer(x)
            if name in layers:
                return x
                
        # if layers is None:
        #     layers = {'0': 'conv1_1',
        #             '5': 'conv2_1', 
        #             '10': 'conv3_1', 
        #             '19': 'conv4_1',
        #             '21': 'conv4_2',  ## content representation
        #             '28': 'conv5_1'}

    def gram_matrix(self, tensor):
        "Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix"
        
        # get the batch_size, depth, height, and width of the Tensor
        b, d, h, w = tensor.size()
        
        # reshape so we're multiplying the features for each channel
        tensor = tensor.view(b * d, h * w)
        
        # calculate the gram matrix
        gram = torch.mm(tensor, tensor.t())
        
        return gram 

    def gram_similarity(self, image_A_path, image_B_path, img_size=512):
        image_A = self.load_image(image_A_path, img_size)
        image_B = self.load_image(image_B_path, img_size)

        features_A = self.get_features(image_A, self.vgg)
        features_B = self.get_features(image_B, self.vgg)

        style_grams_A = self.gram_matrix(features_A)
        style_grams_B = self.gram_matrix(features_B)

        return F.cosine_similarity(style_grams_A[-1].reshape(-1).unsqueeze(0), style_grams_B[-1].reshape(-1).unsqueeze(0))

if __name__ == "__main__":
    imageA = "/tiamat-NAS/songyiren/dataset/Sref508/061/02.png"
    imageB = "/tiamat-NAS/songyiren/dataset/Sref508/061/04.png"
    imageC = "/tiamat-NAS/songyiren/dataset/Sref508/151/02.png"

    vgg_gram_score = vgg_gram('cuda')

    print("A, B:", vgg_gram_score.gram_similarity(imageA, imageB))
    print("A, C:", vgg_gram_score.gram_similarity(imageA, imageC))
