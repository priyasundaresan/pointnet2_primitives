import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np

from pathlib import Path
from tqdm import tqdm
from data_utils.PrimitiveDataLoader import PrimitiveDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='model_cls_off_rot', help='model name')
    parser.add_argument('--batch_size', type=int, default=4, help='batch Size during training')
    parser.add_argument('--epoch', default=100, type=int, help='epoch to run')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='log path')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--npoint', type=int, default=20000, help='point Number')
    parser.add_argument('--inp_dim', type=int, default=3, help='dimensionality of point cloud')
    parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')
    return parser.parse_args()

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    exp_dir = Path('./log/')
    log_dir = exp_dir
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    #root = 'data/dset_open_real'
    #root = 'data/dset_close_real'
    #root = 'data/tool_cabinet'
    #root = 'data/dset_open_v2'
    root = 'data/dset_place'

    TRAIN_DATASET = PrimitiveDataset(root=root, npoints=args.npoint, split='train')
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    TEST_DATASET = PrimitiveDataset(root=root, npoints=args.npoint, split='test')
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)
    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    num_outputs = 7 # (3 offset + 4 quaternion)
    inp_dim = 7 # (xyz, rgb, keypoint mask)
    num_classes = 2 # (nothing, start waypoint)

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))

    classifier = MODEL.get_model(num_outputs, inp_dim, num_classes).cuda()
    cls_criterion = MODEL.get_cls_loss().cuda()
    offset_criterion = MODEL.get_offset_loss().cuda()
    rot_criterion = MODEL.get_rot_loss().cuda()
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0

    test_metrics = {'loss': float('inf')}

    #best_test_acc = 0
    best_offset = float('inf')

    for epoch in range(start_epoch, args.epoch):
        mean_correct = []
        mean_start_offset_loss = []
        mean_start_rot_loss = []

        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()

        '''learning one epoch'''
        for i, (points, cls_target, start_waypt, start_rot) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()

            # AUG
            points[:, :, 0:3], scales = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3], shifts  = provider.shift_point_cloud(points[:, :, 0:3])

            start_waypt *= scales
            start_waypt += shifts

            start_waypt_batched = torch.zeros((args.npoint, args.batch_size, 3)).cuda()
            start_waypt_batched[:,:,:] = start_waypt
            start_waypt_batched = start_waypt_batched.transpose(0,1)

            start_rot_batched = torch.zeros((args.npoint, args.batch_size, 4)).cuda()
            start_rot_batched[:,:,:] = start_rot
            start_rot_batched = start_rot_batched.transpose(0,1)

            points = torch.Tensor(points)
            points, cls_target, start_waypt, start_rot = points.float().cuda(), \
                                                                             cls_target.long().cuda(), \
                                                                             start_waypt.float().cuda(), \
                                                                             start_rot.float().cuda()
            points = points.transpose(2, 1)

            cls_pred, offsets_pred, rots_pred = classifier(points)
            cls_pred = cls_pred.contiguous().view(-1, num_classes)

            cls_target = cls_target.view(-1, 1)[:, 0]

            pred_choice = cls_pred.data.max(1)[1]

            start_idxs = (pred_choice == 1).view(args.batch_size, args.npoint)

            points_xyz = points.transpose(2,1)[:,:,:3]
            
            offsets_start = points_xyz[start_idxs] - start_waypt_batched[start_idxs]
            offsets_start_pred = offsets_pred[:,:,:3][start_idxs]

            rots_start_pred = rots_pred[:,:,:4][start_idxs]
            rots_start = start_rot_batched[start_idxs]

            norm_start = 1/torch.sqrt(torch.sum(rots_start_pred*rots_start_pred, axis=1))
            rots_start_pred_norm = (rots_start_pred.permute(1,0) * norm_start).permute(1,0)

            correct = pred_choice.eq(cls_target.data).cpu().sum()
            mean_correct.append(correct.item() / (args.batch_size * args.npoint))

            cls_loss = cls_criterion(cls_pred, cls_target)

            start_offset_loss = offset_criterion(offsets_start_pred, offsets_start)
            start_rot_loss = rot_criterion(rots_start_pred_norm, rots_start)

            mean_start_offset_loss.append(start_offset_loss.item())
            mean_start_rot_loss.append(start_rot_loss.item())

            alpha = 0.2
            gamma = 0.4
            beta = 0.4
            loss = alpha*cls_loss + gamma*start_offset_loss + beta*start_rot_loss 

            loss.backward()
            optimizer.step()

        log_string('Train accuracy is: %.5f' % np.mean(mean_correct))
        log_string('Train start offset loss is: %.5f' % np.mean(mean_start_offset_loss))
        log_string('Train start rot loss is: %.5f' % np.mean(mean_start_rot_loss))

        with torch.no_grad():
            classifier = classifier.eval()
            mean_correct = []
            mean_start_offset_loss = []

            for batch_id, (points, cls_target, start_waypt, start_rot) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                cur_batch_size, NUM_POINT, _ = points.size()

                points, cls_target, start_waypt, start_rot = points.float().cuda(), \
                                                                                 cls_target.long().cuda(), \
                                                                                 start_waypt.float().cuda(), \
                                                                                 start_rot.float().cuda()
                start_waypt_batched = torch.zeros((args.npoint, cur_batch_size, 3)).cuda()
                start_waypt_batched[:,:,:] = start_waypt
                start_waypt_batched = start_waypt_batched.transpose(0,1)

                start_rot_batched = torch.zeros((args.npoint, cur_batch_size, 4)).cuda()
                start_rot_batched[:,:,:] = start_rot
                start_rot_batched = start_rot_batched.transpose(0,1)

                points = points.transpose(2, 1)
                cls_pred, offsets_pred, rots_pred = classifier(points)
                #print(cls_pred.shape, offsets_pred.shape, rots_pred.shape)

                cls_pred = cls_pred.contiguous().view(-1, num_classes)
                cls_target = cls_target.view(-1, 1)[:, 0]
                pred_choice = cls_pred.data.max(1)[1]

                start_idxs = (pred_choice == 1).view(cur_batch_size, args.npoint)

                points_xyz = points.transpose(2,1)[:,:,:3]

                offsets_start = points_xyz[start_idxs] - start_waypt_batched[start_idxs]
                offsets_start_pred = offsets_pred[:,:,:3][start_idxs]

                rots_start_pred = rots_pred[:,:,:4][start_idxs]
                rots_start = start_rot_batched[start_idxs]
                norm_start = 1/torch.sqrt(torch.sum(rots_start_pred*rots_start_pred, axis=1))
                rots_start_pred_norm = (rots_start_pred.permute(1,0) * norm_start).permute(1,0)

                #print(rots_start_pred_norm.shape)

                correct = pred_choice.eq(cls_target.data).cpu().sum()
                acc = correct.item() / (cur_batch_size * NUM_POINT)
                mean_correct.append(acc)

                start_offset_loss = offset_criterion(offsets_start_pred, offsets_start)
                start_rot_loss = rot_criterion(rots_start_pred_norm, rots_start)

                if not np.isnan(start_offset_loss.item()):
                    mean_start_offset_loss.append(start_offset_loss.item())
                if not np.isnan(start_rot_loss.item()):
                    mean_start_rot_loss.append(start_rot_loss.item())

        test_instance_acc = np.mean(mean_correct)
        log_string('Test Epoch %d, Test Acc: %.5f, Best Loss: %.5f' % (epoch, test_instance_acc, best_offset))
        log_string('Test start offset loss is: %.5f' % np.mean(mean_start_offset_loss))
        log_string('Test start rot loss is: %.5f' % np.mean(mean_start_rot_loss))

        #if test_instance_acc > best_test_acc:
        if np.mean(mean_start_rot_loss) < best_offset:
            logger.info('Save model...')
            savepath = str(log_dir) + '/best_model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'acc': test_instance_acc,
                'model_state_dict': classifier.state_dict(),
            }
            #best_offset = np.mean(mean_start_offset_loss)
            best_offset = np.mean(mean_start_rot_loss)
            torch.save(state, savepath)
            log_string('Saving model....')

        global_epoch += 1

if __name__ == '__main__':
    args = parse_args()
    main(args)
