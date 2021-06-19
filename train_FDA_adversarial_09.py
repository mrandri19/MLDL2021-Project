import os
import gc

from IPython.core.display import display, HTML

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T

from tqdm import tqdm

from model.build_BiSeNet import BiSeNet
from dataset.IDDA import IDDA
from dataset.CamVid import CamVid

import numpy as np

NUM_CLASSES = 12

CROP_HEIGHT = 720
CROP_WIDTH = 960

CONTEXT_PATH = 'resnet101'

LEARNING_RATE_SEGMENTATION = 2.5e-4
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9

LEARNING_RATE_DISCRIMINATOR = 1e-4

POWER = 0.9

NUM_STEPS = 250000
ITER_SIZE = 1

BETA = 0.09

CHECKPOINT_STEP = 5
CHECKPOINT_PATH = './checkpointBeta09/'

BATCH_SIZE_CAMVID = 2
BATCH_SIZE_IDDA = 2

CAMVID_PATH = ['/content/CamVid/train/', '/content/CamVid/val/']
CAMVID_TEST_PATH = ['/content/CamVid/test/']
CAMVID_LABEL_PATH = ['/content/CamVid/train_labels/', '/content/CamVid/val_labels/']
CAMVID_TEST_LABEL_PATH = ['/content/CamVid/test_labels/']
CSV_CAMVID_PATH = '/content/CamVid/class_dict.csv'

IDDA_PATH = '/content/IDDA/rgb/'
IDDA_LABEL_PATH = '/content/IDDA/labels'
JSON_IDDA_PATH = '/content/IDDA/classes_info.json'

NUM_WORKERS = 0

LOSS = 'dice'

SOURCE_LABEL = 0
TARGET_LABEL = 1

def low_freq_mutate( amp_src, amp_trg, beta):
    n, c, h, w = amp_src.size()
    b = (np.floor(np.amin((h,w))*beta)).astype(int)         # get b (square with smallested among h,w)
    amp_src[:,:,0:b,0:b]     = amp_trg[:,:,0:b,0:b]      # top left
    amp_src[:,:,0:b,w-b:w]   = amp_trg[:,:,0:b,w-b:w]    # top right
    amp_src[:,:,h-b:h,0:b]   = amp_trg[:,:,h-b:h,0:b]    # bottom left
    amp_src[:,:,h-b:h,w-b:w] = amp_trg[:,:,h-b:h,w-b:w]  # bottom right
    return amp_src

def FDA_source_to_target(src_img, trg_img, beta=1e-2):
    # exchange magnitude
    # input: src_img, trg_img

    # get fft of both source and target
    fft_src = torch.fft.fftn(src_img.clone(), dim=(2, 3)) # check if fft2 is enough
    fft_trg = torch.fft.fftn(trg_img.clone(), dim=(2, 3))

    assert fft_src.dtype == torch.complex64, fft_src.dtype
    assert fft_trg.dtype == torch.complex64, fft_src.dtype
    assert fft_src.shape == (BATCH_SIZE_CAMVID, 3, 720, 960), fft_src.shape
    assert fft_trg.shape == (BATCH_SIZE_CAMVID, 3, 720, 960), fft_trg.shape

    # extract amplitude and phase of both ffts
    amp_src, pha_src = fft_src.abs(), fft_src.angle()
    amp_trg, pha_trg = fft_trg.abs(), fft_trg.angle()

    assert amp_src.dtype == torch.float32, f"assertion failure {amp_src.dtype}"
    assert amp_trg.dtype == torch.float32, f"assertion failure {amp_src.dtype}"
    assert amp_src.shape == (BATCH_SIZE_CAMVID, 3, 720, 960), f"assertion failure {amp_src.shape}"
    assert amp_trg.shape == (BATCH_SIZE_CAMVID, 3, 720, 960), f"assertion failure {amp_trg.shape}"

    # replace the low frequency amplitude part of source with that from target
    amp_src_ = low_freq_mutate(amp_src.clone(), amp_trg.clone(), beta=beta)

    assert amp_src_.dtype == torch.float32, f"assertion failure {amp_src_.dtype}"
    assert amp_src_.shape == (BATCH_SIZE_CAMVID, 3, 720, 960), f"assertion failure {amp_src_.shape}"

    # recompose fft of source
    fft_src_real = torch.cos(pha_src.clone()) * amp_src_.clone()
    fft_src_imag = torch.sin(pha_src.clone()) * amp_src_.clone()
    fft_src_ = torch.complex(fft_src_real, fft_src_imag)
    assert fft_src_.shape == (BATCH_SIZE_CAMVID, 3, 720, 960), f"assertion failure {fft_src_.shape}"
  
    # get the recomposed image: source content, target style
    _, _, imgH, imgW = src_img.size()
    src_in_trg = torch.fft.ifftn(fft_src_, dim=(2, 3))
    assert src_in_trg.shape == (BATCH_SIZE_CAMVID, 3, 720, 960), f"assertion failure {src_in_trg.shape}"

    return src_in_trg

def adjust_learning_rate(optimizer, initial_learning_rate, step, max_num_step, power):
  # polynomial decay of learning rate
  lr = initial_learning_rate*((1 - float(step)/max_num_step)**(power))
  optimizer.param_groups[0]['lr'] = lr

class CrossEntropy2d(nn.Module):
    def __init__(self, size_average=True, ignore_label=11):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss


class Discriminator(nn.Module):
  def __init__(self, num_classes, ndf = 64):
    super(Discriminator, self).__init__()

    self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
    self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
    self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
    self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
    self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

    self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

  def forward(self, x):
    x = self.conv1(x);
    x = self.leaky_relu(x);
    x = self.conv2(x);
    x = self.leaky_relu(x);
    x = self.conv3(x);
    x = self.leaky_relu(x);
    x = self.conv4(x);
    x = self.leaky_relu(x);
    x = self.classifier(x);

    return x


# Standardize an image with Imagenet's mean and variance
mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
normalize = T.Normalize(mean, std)
unnormalize = T.Normalize((-mean/std).tolist(), (1.0 / std).tolist())

def minibatch(source_dataloader_iter, target_dataloader_iter, generator, discriminator, cross_entropy_loss, bce_loss, source_dataloader, target_dataloader):
  # with Stage('minibatch'):
  # print(f'{torch.cuda.memory_allocated() / 10 ** 9:.2f} GB, {torch.cuda.memory_reserved() / 10 ** 9:.2f} GB')
  l_seg_to_print, l_adv_to_print, l_d_to_print = 0, 0, 0
  ##########################################################################
  # TRAIN G (segmentation network)
  ##########################################################################

  # Freeze discriminator D
  for parameter in discriminator.parameters():
      parameter.require_grad = False 
  
  # Load batch_size images from source
  try:
    batch = next(source_dataloader_iter)
  except StopIteration:
    source_dataloader_iter = iter(source_dataloader)
    batch = next(source_dataloader_iter)
  I_s, Y_s = batch
  I_s, Y_s = I_s.cuda(), Y_s.cuda()

  # Load batch_size images from target
  try:
    batch = next(target_dataloader_iter)
  except StopIteration:
    target_dataloader_iter = iter(target_dataloader)
    batch = next(target_dataloader_iter)
  I_t, _ = batch
  I_t = I_t.cuda()

  I_s_unnormalized = unnormalize(I_s)
  I_t_unnormalized = unnormalize(I_t)

  I_s2t_unnormalized = FDA_source_to_target(
    I_s_unnormalized,
    I_t_unnormalized,
    beta=BETA
  ).real.cuda()

  I_s2t = normalize(I_s2t_unnormalized)

  # Compute P_s = G(I_s2t)
  P_s, _, _ = generator(I_s2t)

  # Compute segmentation loss
  l_seg = cross_entropy_loss(P_s, torch.argmax(Y_s, dim=1))
  l_seg_to_print += l_seg.item()
  l_seg = l_seg / ITER_SIZE # normalization
  l_seg.backward()

  # Compute P_t = G(I_t)
  P_t, _, _ = generator(I_t)

  # TODO: for stage 2, cross entropy loss of P_t and Y_t_PSU

  # Compute D(sigma(G(I_t)))
  D_t = discriminator(F.softmax(P_t))

  # Compute l_adv: compare D_t with a tensor of the same dimension filled
  # with 0 (SOURCE_LABEL) -> aims to be recognize as source image 
  l_adv = bce_loss(D_t, torch.full(D_t.size(), SOURCE_LABEL, dtype=torch.float32).cuda()) 
  l_adv_to_print += l_adv.item()
  l_adv = l_adv/ITER_SIZE # normalization
  l_adv.backward()

  ##########################################################################
  # TRAIN D (discriminator)
  ##########################################################################
  for parameter in discriminator.parameters():  # unfreeze discriminator D  
      parameter.require_grad = True
  
  #################################
  # Train with SOURCE
  #################################
  P_s = P_s.detach() # no gradient will be backpropagated along this tensor 
  
  # Compute D(sigma(G(I_s)))
  D_s = discriminator(F.softmax(P_s))
  
  # Compute l_d: compare D_s with a tensor of the same dimension filled with
  # 0 (SOURCE_LABEL)
  l_d = bce_loss(D_s, torch.full(D_s.size(), SOURCE_LABEL, dtype=torch.float32).cuda()) 
  l_d_to_print += l_d.item()
  l_d = l_d / (ITER_SIZE * 2) 
  l_d.backward()

  #################################
  # Train with TARGET
  #################################
  # No gradient will be backpropagated along this tensor 
  P_t = P_t.detach()
  
  # Compute D(sigma(G(I_t)))
  D_t = discriminator(F.softmax(P_t))

  # Compute l_d: compare D_t with a tensor of the same dimension filled with
  # 1 (TARGET_LABEL)
  l_d = bce_loss(D_t, torch.full(D_t.size(), TARGET_LABEL, dtype=torch.float32).cuda()) 
  l_d_to_print += l_d.item()
  l_d = l_d / (ITER_SIZE * 2)
  l_d.backward()

  #print(l_seg_to_print, l_adv_to_print, l_d_to_print)

  return l_seg_to_print, l_adv_to_print, l_d_to_print

def main():
  # Call Python's garbage collector, and empty torch's CUDA cache. Just in case
  gc.collect()
  torch.cuda.empty_cache()
  
  # Enable cuDNN in benchmark mode. For more info see:
  # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
  torch.backends.cudnn.enabled = True
  torch.backends.cudnn.benchmark = True

  # Load Bisenet generator
  generator = BiSeNet(NUM_CLASSES, CONTEXT_PATH).cuda()
  generator.load_state_dict(torch.load('./checkpointBeta09/0.09_40_Generator.pth'))
  generator.train()
  # Build discriminator
  discriminator = Discriminator(NUM_CLASSES).cuda()
  discriminator.load_state_dict(torch.load('./checkpointBeta09/0.09_40_Discriminator.pth'))
  discriminator.train()

  # Load source dataset
  source_dataset = IDDA(
      image_path=IDDA_PATH,
      label_path=IDDA_LABEL_PATH,
      classes_info_path=JSON_IDDA_PATH,
      scale=(CROP_HEIGHT, CROP_WIDTH),
      loss=LOSS,
      mode='train'
  )
  source_dataloader = DataLoader(
        source_dataset,
        batch_size=BATCH_SIZE_IDDA,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
        pin_memory=True
    )

  # Load target dataset
  target_dataset = CamVid(
    image_path=CAMVID_PATH,
    label_path= CAMVID_LABEL_PATH,csv_path= CSV_CAMVID_PATH,
    scale=(CROP_HEIGHT,
    CROP_WIDTH),
    loss=LOSS,
    mode='adversarial_train'
  )
  target_dataloader = DataLoader(
        target_dataset,
        batch_size=BATCH_SIZE_CAMVID,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
        pin_memory=True
    )

  optimizer_BiSeNet = torch.optim.SGD(generator.parameters(), lr = LEARNING_RATE_SEGMENTATION, momentum = MOMENTUM, weight_decay = WEIGHT_DECAY)   
  optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr = LEARNING_RATE_DISCRIMINATOR, betas = (0.9,0.99))

  # Loss for discriminator training
  # Sigmoid layer + BCELoss
  bce_loss = nn.BCEWithLogitsLoss()

  # Loss for segmentation loss
  # Log-softmax layer + 2D Cross Entropy
  cross_entropy_loss = CrossEntropy2d()

  # for epoch in range(NUM_STEPS):
  for epoch in range(41, 51):
    source_dataloader_iter = iter(source_dataloader)
    target_dataloader_iter = iter(target_dataloader)

    print(f'begin epoch {epoch}')

    # Initialize gradients=0 for Generator and Discriminator
    optimizer_BiSeNet.zero_grad()
    optimizer_discriminator.zero_grad()

    # Setting losses equal to 0
    l_seg_to_print_acc, l_adv_to_print_acc, l_d_to_print_acc = 0, 0, 0

    # Compute learning rate for this epoch
    adjust_learning_rate(optimizer_BiSeNet, LEARNING_RATE_SEGMENTATION, epoch, NUM_STEPS, POWER)
    adjust_learning_rate(optimizer_discriminator, LEARNING_RATE_DISCRIMINATOR, epoch, NUM_STEPS, POWER)

    for i in tqdm(range(len(target_dataloader))):
      optimizer_BiSeNet.zero_grad()
      optimizer_discriminator.zero_grad()
      l_seg_to_print, l_adv_to_print, l_d_to_print = minibatch(source_dataloader_iter, target_dataloader_iter, generator, discriminator, cross_entropy_loss, bce_loss, source_dataloader, target_dataloader)
      l_seg_to_print_acc += l_seg_to_print
      l_adv_to_print_acc += l_adv_to_print
      l_d_to_print_acc += l_d_to_print
      # Run optimizers using the gradient obtained via backpropagations
      optimizer_BiSeNet.step()
      optimizer_discriminator.step()
    
    # Output at each epoch
    print(f'epoch = {epoch}/{NUM_STEPS}, loss_seg = {l_seg_to_print_acc:.3f}, loss_adv = {l_adv_to_print_acc:.3f}, loss_D = {l_d_to_print_acc:.3f}')

    # Save intermediate generator (checkpoint)
    if epoch % CHECKPOINT_STEP == 0 and epoch != 0:
      # If the directory does not exists create it
      if not os.path.isdir(CHECKPOINT_PATH):
        os.mkdir(CHECKPOINT_PATH)
      # Save the parameters of the generator (segmentation network) and discriminator 
      generator_checkpoint_path = os.path.join(CHECKPOINT_PATH, f"{BETA}_{epoch}_Generator.pth")
      torch.save(generator.state_dict(), generator_checkpoint_path)
      discriminator_checkpoint_path = os.path.join(CHECKPOINT_PATH, f"{BETA}_{epoch}_Discriminator.pth")
      torch.save(discriminator.state_dict(), discriminator_checkpoint_path)
      print(f"saved:\n{generator_checkpoint_path}\n{discriminator_checkpoint_path}")

# from pyinstrument import Profiler
# profiler = Profiler(interval=1e-4)
# profiler.start()
# main()
# profiler.stop()
# display(HTML(profiler.output_html()))
main()