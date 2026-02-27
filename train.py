import argparse
import os
import numpy as np
from tqdm import tqdm
import warnings
from dataloaders import make_data_loader_heightmap
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.PHENet import *
from utils.loss import SegmentationLosses
from utils.lr_scheduler import LR_Scheduler
from utils.saver2 import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
import shutil
warnings.filterwarnings("ignore")
import sys

sys.path.append("/data/coding/HazyCDNet")
class PhysicalLosses(nn.Module):

    def __init__(self):
        super().__init__()
        
    def tv_loss(self, t):

        diff_x = torch.abs(t[:, :, 1:, :] - t[:, :, :-1, :])
        diff_y = torch.abs(t[:, :, :, 1:] - t[:, :, :, :-1])
        return torch.mean(diff_x) + torch.mean(diff_y)
    
    def dark_channel_loss(self, J):

        min_channel = torch.min(J, dim=1)[0] 
        return torch.mean(min_channel**2)
    
    def forward(self, t, J):
        return self.tv_loss(t), self.dark_channel_loss(J)
class AdversarialLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.discriminator = nn.Sequential(
            nn.Conv2d(7, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1)
        )
    
    def forward(self, combined_input):
        return self.discriminator(combined_input)

class ShallowCNN(nn.Module):

    def __init__(self, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  
        x = F.relu(self.conv3(x))
        return x
def generate_pseudo_label(x1, x2):

    shallow_cnn = ShallowCNN().to(x1.device)
    with torch.no_grad():
        feat1 = shallow_cnn(x1)  
        feat2 = shallow_cnn(x2)
    diff = torch.norm(feat1 - feat2, p=2, dim=1)  
    

    pseudo_labels = []
    for b in range(diff.size(0)):
        img = diff[b].cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min()) * 255
        img = img.astype(np.uint8)
        _, mask = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        pseudo_labels.append(torch.from_numpy(mask).unsqueeze(0))
    
    M_low = torch.stack(pseudo_labels).to(x1.device).float()  
    M_low = F.interpolate(M_low, scale_factor=4, mode='nearest')  
    return M_low

class Trainer(object):
    def __init__(self, args):
        self.args = args
        

        
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.nclass = make_data_loader_heightmap(args, **kwargs)

        model = PHENet(num_classes=self.nclass,
                           backbone=args.backbone,
                           output_stride=args.out_stride,
                           sync_bn=args.sync_bn,
                           freeze_bn=args.freeze_bn)

        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr}]

        
        if args.optim == "SGD":
            optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                        weight_decay=args.weight_decay, nesterov=args.nesterov)
        if args.optim == "Adam":
            optimizer = torch.optim.Adam(train_params)
        
        if args.use_balanced_weights:
            pass
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer
        self.phy_loss = PhysicalLosses()
        self.adv_loss = AdversarialLoss().cuda()  
        self.adv_optim = torch.optim.Adam(self.adv_loss.parameters(), lr=args.lr)

        
        self.evaluator = Evaluator(self.nclass)
        
        self.optim = args.optim
        if args.optim == "Adam":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.9,
                                                                        patience=5, verbose=True)
        if args.optim == "SGD":
            self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                          args.epochs, len(self.train_loader), lr_step=args.lr_step)

        
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            self.best_pred = 0
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        self.adv_loss.discriminator.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, (image1, image2, target, height1, height2, id) in enumerate(tbar):
            if self.args.cuda:
                image1, image2, target, height1, height2 = image1.cuda(), image2.cuda(), target.cuda(), height1.cuda(), height2.cuda()
            if self.optim == "SGD":
                self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            
            
            output, I_HR_hazy1, J1, t1, A1, I_HR_hazy2, J2, t2, A2 = self.model(image1, image2, height1, height2)
            target = target.squeeze()
            loss = self.criterion(output, target)

            loss_tv1, loss_dark1 = self.phy_loss(t1, J1)
            loss_tv2, loss_dark2 = self.phy_loss(t2, J2)
            loss_phy = (loss_tv1 + loss_tv2) * 0.2 + (loss_dark1 + loss_dark2) * 0.2
            pseudo_label = generate_pseudo_label(image1, image2) 
            combined_input = torch.cat([I_HR_hazy1, I_HR_hazy2, pseudo_label], dim=1)  
            
            fake_logits = self.adv_loss(combined_input)
            
            loss_adv = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
            total_loss = loss + loss_phy + loss_adv * 0.2

            total_loss.backward()
            self.optimizer.step()

            
            self.adv_optim.zero_grad()
            real_combined = torch.cat([image1, image2, pseudo_label], dim=1)  
            real_logits = self.adv_loss(real_combined)
            loss_d_real = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
            loss_d_fake = F.binary_cross_entropy_with_logits(fake_logits.detach(), torch.zeros_like(fake_logits))
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            self.adv_optim.step()
            
            train_loss += total_loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
        if self.optim == "Adam":
            self.scheduler.step(train_loss / num_img_tr)
        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image1.data.shape[0]))
        print('Loss: %.3f' % train_loss)
        if self.args.no_val:
            
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'best_pred': self.best_pred,
            }, is_best=False)
            
            
    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, (image1, image2, target, height1, height2, id) in enumerate(tbar):
            if self.args.cuda:
                image1, image2, target = image1.cuda(), image2.cuda(), target.cuda()
            with torch.no_grad():
                output, I_HR_hazy1, J1, t1, A1, I_HR_hazy2, J2, t2, A2 = self.model(image1, image2, height1, height2)
            target = target.squeeze()
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            
            self.evaluator.add_batch(target, pred)

        
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image1.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU
        is_best = new_pred > self.best_pred
        if is_best:
            self.best_pred = new_pred

        
        self.saver.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.module.state_dict(),
            'best_pred': self.best_pred,
        }, is_best)  
















def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--isdeconv', type=str, default='deconv',
                        choices=['deconv', 'nodeconv'],
                        help='whether use deconv ')
    parser.add_argument("--data_root", type=str, default=r'/data/coding/data/hazy-whu_cd_256',
                        help="path to Dataset,LEVIR-CD,WHU,DSIFN,255pair_png")
    parser.add_argument('--backbone', type=str, default='mobilenet',
                        choices=['resnet', 'xception', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='cityscapes',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=256,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=256,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='dice_ce',
                        choices=['ce', 'focal', 'dice_ce'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=8,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=8,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-2)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos', 'linear'],
                        help='lr scheduler mode: (default: poly) for SGD')
    parser.add_argument('--lr-step', type=int, default=100,
                        help='lr scheduler mode: (default: 100)')
    parser.add_argument('--optim', type=str, default='Adam',
                        choices=['SGD', 'Adam'],
                        help='lr scheduler mode: (default: SGD)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9),line :0.99')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
    False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--end-max_epoches', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default='111',
                        help='set the checkpoint name')

    parser.add_argument('--ft', action='store_true', default=True,
                        help='finetuning on a different dataset')

    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    parser.add_argument('--max-end', type=int, default=20,
                        help='epoch numbers for pre best')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False


    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 100,
            'pascal': 50,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    if args.checkname is None:
        args.checkname = 'deeplab-' + str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        # # if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
        if not trainer.args.no_val and epoch % args.eval_interval == 0:
            trainer.validation(epoch)
    trainer.writer.close()


if __name__ == "__main__":
    main()
