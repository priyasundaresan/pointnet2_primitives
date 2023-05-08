"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.PourDataLoader_cls_off import PourDataset
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np
import open3d as o3d

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR

sys.path.append(os.path.join(ROOT_DIR, 'models'))

def visualize(pcl_input, offsets):
    pcl_input = pcl_input.squeeze().cpu().numpy().T
    points = pcl_input[:,:3]
    colors = pcl_input[:,3:]
    offsets = offsets.squeeze()

    # Set up offsets 
    distances_vis = np.zeros((len(points), 3))
    distances = np.linalg.norm(offsets, axis=1)
    distances_normalized = (distances - np.amin(distances))/(np.amax(distances) - np.amin(distances))
    distances_vis[:, 2] = distances_normalized
    colors += (distances_vis*255).astype(np.uint8)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors/255.)

    o3d.visualization.draw_geometries([pcd]) # with display

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = args.log_dir

    '''LOG'''
    #args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_string('PARAMETER ...')
    log_string(args)

    root = 'data/pour_dset4/'

    TEST_DATASET = PourDataset(root=root, npoints=args.npoint, split='test')
    #TEST_DATASET = PourDataset(root=root, npoints=args.npoint, split='train')
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    num_outputs = 3
    inp_dim = 6
    num_classes = 3

    MODEL = importlib.import_module(args.model)
    classifier = MODEL.get_model(num_outputs, inp_dim, num_classes).cuda()
    checkpoint = torch.load(str(experiment_dir) + 'best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    if not os.path.exists('preds'):
        os.mkdir('preds')

    ctr = 0
    with torch.no_grad():
        classifier = classifier.eval()
        for batch_id, (points, target, start_waypt, end_waypt) in tqdm(enumerate(testDataLoader), total=len(testDataLoader),
                                                      smoothing=0.9):
            batchsize, num_point, _ = points.size()
            cur_batch_size, NUM_POINT, _ = points.size()
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            pred, offsets = classifier(points)
            pred = pred.contiguous().view(-1, num_classes)
            pred_cls  = pred.data.max(1)[1].cpu().numpy()
            
            points_np = points.squeeze().detach().cpu().numpy().T
            colors_np = points_np[:,3:]
            points_np = points_np[:,:3]

            offsets = offsets.squeeze().detach().cpu().numpy().T

            data = {'xyz':points_np, 'xyz_color':colors_np, 'cls':pred_cls, 'offsets':offsets}
            np.save('preds/%d.npy'%ctr, data)
            ctr += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in testing')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--npoint', type=int, default=5000, help='point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--model', type=str, default='model_cls_off', help='model name')
    args = parser.parse_args()
    
    main(args)
