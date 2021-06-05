My model for https://github.com/fastai/imagenette challenge.

I have tried 3 methods to train resnet18 for this challenge. 
1. Classic. Just use original trainset. 93% accuracy
2. Cropping dogs. 91% accuracy (boxed photo method)
3. Cropping dogs and use segmentation mask. In development. (masked photo method)
