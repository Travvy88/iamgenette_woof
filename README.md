My model for https://github.com/fastai/imagenette challenge.

I have tried 3 methods to train resnet18 for this challenge. 
1. Classic. Just use original trainset. 93% accuracy
2. Cropping dogs. 91% accuracy (cropped dogs method)
3. In development: use segmentations masks of dogs to help the net

