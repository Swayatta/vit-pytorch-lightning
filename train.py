import yaml
from vit import ViT
import os
import torch
import torch.utils.data as data
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf
import hydra


# with open('config/config.yaml', 'rb') as f:
#     conf = yaml.safe_load(f.read())    # load the config file

# hydra.initialize(version_base=None, config_path="./config")
# conf = hydra.compose(config_name="config")

# batch_size = conf['batch_size']
# height_of_patch = conf['height_of_patch']
# width_of_patch = conf['width_of_patch']
# channel = conf['channel']
# num_blocks = conf['num_blocks']
# dim = conf['dim']
# mlp_dim = conf['mlp_dim']
# attention_head_size = conf['attention_head_size']
# num_attention_heads = conf['num_attention_heads']
# accelerator = conf['accelerator']
# devices = conf['devices']
# epochs = conf['epochs']

# Defining lightning module

class LitVit(pl.LightningModule):
    def __init__(self, vit):
        super().__init__()
        self.vit = vit      # obtaining the created pytorch model
        self.loss = nn.CrossEntropyLoss()   # defining the loss

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x,y = batch
        b,c,img_h,img_w = x.shape   # obtaining shape from the image - batch,channel, img_height, img_width
        x = x.view(b,1,img_h, img_w, c)   # transforming the image to proper shape to facilitate training
        predictions = self.vit(x)    # obtaining the output from the model
        loss = self.loss(predictions,y)     # Calculating the loss
        
        # # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop, similar to the train loop
        x,y = batch
        b,c,img_h,img_w = x.shape
        x = x.view(b,1,img_h, img_w, c)
        predictions = self.vit(x)
        val_loss = self.loss(predictions,y)
        
        # # Logging to TensorBoard (if installed) by default
        self.log("val_loss", val_loss)
    
    def test_step(self, batch, batch_idx):
        x,y = batch
        b,c,img_h,img_w = x.shape
        x = x.view(b,1,img_h, img_w, c)
        predictions = self.vit(x)
        test_loss = self.loss(predictions,y)
        
        # # Logging to TensorBoard (if installed) by default
        self.log("test_loss", test_loss)


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)  # configuring the optimizer
        return optimizer

@hydra.main(config_path="./config", config_name="config")
def train(conf: DictConfig):
    batch_size = conf['batch_size']
    height_of_patch = conf['height_of_patch']
    width_of_patch = conf['width_of_patch']
    num_blocks = conf['num_blocks']
    dim = conf['dim']
    mlp_dim = conf['mlp_dim']
    attention_head_size = conf['attention_head_size']
    num_attention_heads = conf['num_attention_heads']
    accelerator = conf['accelerator']
    devices = conf['devices']
    epochs = conf['epochs']
    test = conf['test']

    train_set = MNIST(os.getcwd(), download=True, train = True, transform=ToTensor())
    test_set = MNIST(os.getcwd(), download=True, train = False, transform=ToTensor())

    num_of_labels = len(train_set.classes)

    # use 20% of training data for validation
    train_set_size = int(len(train_set) * 0.8)
    valid_set_size = len(train_set) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)

    ### Loading a subsample of mnist. We will be using only a subset of MNIST ###
    K = int(0.8*6000) # enter your length here
    subsample_train_indices = torch.randperm(len(train_set))[:K]        

    # train loader on a subset of trainset for training
    train_loader = utils.data.DataLoader(train_set, batch_size=batch_size, sampler=utils.data.SubsetRandomSampler(subsample_train_indices)) 

    # validation loader on a subset of validation set for validation
    K = int(0.2*6000)
    subsample_valid_indices = torch.randperm(len(valid_set))[:K]
    valid_loader = utils.data.DataLoader(valid_set, batch_size=batch_size, sampler=utils.data.SubsetRandomSampler(subsample_valid_indices))

    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    _,channel, _, _ = images.shape

    # # The main pytorch model
    vit = ViT(batch_size, height_of_patch, width_of_patch, channel, num_blocks, dim, mlp_dim, attention_head_size, num_attention_heads, num_of_labels)
    lightvit = LitVit(vit)

    # train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
    trainer = pl.Trainer(limit_train_batches = batch_size, max_epochs=epochs, devices = devices, accelerator= accelerator)
    trainer.fit(lightvit, train_loader, valid_loader)

    if test:
        ### Loading a subsample of mnist. We will be using only a subset of MNIST ###
        K = int(0.2*6000) # enter your length here
        subsample_test_indices = torch.randperm(len(test_set))[:K]        

        # train loader on a subset of trainset for training
        test_loader = utils.data.DataLoader(test_set, batch_size=batch_size, sampler=utils.data.SubsetRandomSampler(subsample_test_indices))
        trainer.test(lightvit, test_loader)

if __name__ == "__main__":
    train()
    
    