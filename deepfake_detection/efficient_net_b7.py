#%%
import torch

import deepfake_detection.training
from deepfake_detection.dataset_object import images_dataset, train_frame, val_frame, test_frame
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b7
from torchvision.models import EfficientNet_B7_Weights
from deepfake_detection.training import Optimization

# changed num classes to 2

#%% Model Hyperparameter
# Device configuration
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_name = 'Effnet'
# Hyper-parameters
num_epochs = 500
batch_size = 200
shuffle_batches = False
dropout_prob = 0.3
learning_rate = .1
weight_decay = 1e-8
print('defined all hyperparams')

#%%
# Dataset Generation
train_images = images_dataset(dataset_spec_frame=train_frame)
train_loader = DataLoader(train_images, batch_size=batch_size, shuffle=True)
print(train_images[0])

val_images = images_dataset(dataset_spec_frame=val_frame)
val_loader = DataLoader(val_images, batch_size=batch_size, shuffle=True)

test_images = images_dataset(dataset_spec_frame=test_frame)
test_loader = DataLoader(test_images, batch_size=batch_size, shuffle=True)
print('generated required datasets')


#%% Model Specification
model = efficientnet_b7(weights= EfficientNet_B7_Weights.DEFAULT, progress= True).to(device)
print(model)
model.classifier = torch.nn.Linear(2560, 2, bias=True)
print(model)
print('defined model specs')
#%% Print Summary of Arch
for name, param in model.named_parameters():
    if param.requires_grad:
        print (name, param.data)


#%% Model training specs
criterion = torch.nn.CrossEntropyLoss()

# Adam
# https://arxiv.org/abs/1412.6980
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# optimal ADAM parameters: Good default settings for the tested machine learning problems are stepsize "lr" = 0.001,
#  Exponential decay rates for the moment estimates "betas" β1 = 0.9, β2 = 0.999 and
#  epsilon decay rate "eps" = 10−8

# AdamW decouple weight decay regularization improving upon standard Adam
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# https://arxiv.org/abs/1711.05101

#optimizer = torch.optim.LBFGS(model.parameters(), lr=0.08)
# An LBFGS solver is a quasi-Newton method which uses the inverse of the Hessian to estimate the curvature of the
# parameter space. In sequential problems, the parameter space is characterised by an abundance of long,
# flat valleys, which means that the LBFGS algorithm often outperforms other methods such as Adam, particularly when
# there is not a huge amount of data.

# Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5,  verbose=True)
# try with Cosine Annealing LR
# torch.optim.lr_scheduler.CosineAnnealingLR

opt = Optimization(model=model, model_name=model_name, loss_fn=criterion, optimizer=optimizer, device=device)

opt.train(train_loader, val_loader, batch_size=batch_size, n_epochs=num_epochs)
