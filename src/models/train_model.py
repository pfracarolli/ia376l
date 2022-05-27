import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision import transforms
import torch.nn.functional as F
#
from utilidades import *
from modelos import Discriminator, Generator
import lpips
import DiffAugment_pytorch
from DiffAugment_pytorch import DiffAugment
policy = 'color,translation'

def train(args):

    data_root = args.data_root
    save_path = args.save_path
    checkpoint = args.checkpoint
    batch_size = args.batch_size
    save_interval = args.save_interval
    model_interval = args.model_interval
    NUM_EPOCHS = args.num_epochs
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device:', device)

    nz = 256 # Dimensionality of the latent space
    b1, b2 = 0.0, 0.9 # Hyper-parameters for Adam optimizer
    lr1, lr2 = 0.0001, 0.0004 # Learning rates for the generator and discriminator
    img_size = 512
    
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device) # LPIPS loss

    dataset = dset.ImageFolder(root=data_root, transform=transforms.Compose([
        transforms.Resize(img_size), transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, num_workers=2) # Creating the dataloader
    print("The dataset has ", len(dataset), " images.")

    ##################################################################################################################
    gen = Generator().to(device)
    discriminator = Discriminator().to(device)

    gen.apply(weights_init)
    discriminator.apply(weights_init)
    print("Networks initialized.")

    avg_param_G = copy_G_params(gen)

    fixed_noise = torch.FloatTensor(8, nz, 1, 1).normal_(0, 1).to(device)

    iters = 0

    optimizer_g = optim.Adam(gen.parameters(), lr1, betas=(b1, b2))
    optimizer_d = optim.Adam(discriminator.parameters(), lr2, betas=(b1, b2))

    if checkpoint != 'None':
        checkpoint = torch.load(checkpoint)
        discriminator.load_state_dict(checkpoint['discriminator_state_dict']) 
        gen.load_state_dict(checkpoint['gen_state_dict'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        avg_param_G = checkpoint['g_ema']
        iters = checkpoint['iters']
        print("Networks loaded from checkpoint")
        del checkpoint
    
    ##################################################################################################################
    print("Starting Training Loop...")
    for epoch in range(NUM_EPOCHS):
        for i, (imgs, _) in enumerate(dataloader, 0):
            # Train discriminator with real batch
            discriminator.zero_grad()
            real_cpu = imgs.to(device)
            real_cpu = DiffAugment(real_cpu, policy=policy)
            real_128 = F.interpolate(real_cpu, size=(128, 128), mode='bilinear', align_corners=False)
            real_crop = random_crop(real_cpu, 128)
            b_size = real_cpu.size(0)
            pred, [rec_part, rec_all] = discriminator(real_cpu, tag='Real')
            err = F.relu(  torch.rand_like(pred) * 0.2 + 0.8 -  pred).mean() + \
            loss_fn_vgg( rec_all, real_128 ).sum() +\
            loss_fn_vgg( rec_part, real_crop).sum()
            err.backward()
            D_x = pred.mean().item()

            # Train discriminator with fake batch
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = gen(noise)
            fake = DiffAugment(fake, policy=policy)
            output = discriminator(fake.detach(), tag='Fake')
            errD_fake = F.relu( torch.rand_like(output) * 0.2 + 0.8 + output).mean()
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = err + errD_fake
            optimizer_d.step()

            # Train generator
            gen.zero_grad()
            output = discriminator(fake, tag='Fake')
            errG = -output.mean()
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizer_g.step()

            for p, avg_p in zip(gen.parameters(), avg_param_G):
              avg_p.mul_(0.995).add_(0.005 * p.data)

            # Print the log info
            if i % 10 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                    % (epoch, NUM_EPOCHS, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            
            # Save the generated samples
            if iters % (save_interval) == 0:
                print("Saving the generated samples...")
                backup_para = copy_G_params(gen)
                load_params(gen, avg_param_G)
                with torch.no_grad():
                    fake = gen(fixed_noise)
                vutils.save_image(fake.detach(), '%s/fake_samples_iter_%03d.png' % (save_path, iters), nrow=4, normalize=True)
                vutils.save_image(torch.cat([real_128, rec_all, rec_part]), '%s/real_samples_iter_%03d.png' % (save_path, iters), nrow=8, normalize=True)
                load_params(gen, backup_para)

            iters += 1
            
        # Save the model checkpoints
        if (epoch % model_interval == 0) or epoch == NUM_EPOCHS - 1:
            torch.save({'discriminator_state_dict': discriminator.state_dict(),
                        'gen_state_dict': gen.state_dict(),
                        'optimizer_g_state_dict': optimizer_g.state_dict(),
                        'optimizer_d_state_dict': optimizer_d.state_dict(),
                        'g_ema': avg_param_G,
                        'iters': iters},
                        '%s/checkpoint_iter_%03d.pth' % (save_path, iters))

            print("Checkpoint saved.")

    print("Training Finished")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_root', type=str, default='None', help='path to training data')
    parser.add_argument('--save_path', type=str, default='None', help='path to save model')
    parser.add_argument('--checkpoint', type=str, default='None', help='path to load the model')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=700, help='number of epochs')
    parser.add_argument('--model_interval', type=int, default=70, help='model save interval')
    parser.add_argument('--save_interval', type=int, default=400, help='image save interval')

    args = parser.parse_args()
    print(args)

    train(args)