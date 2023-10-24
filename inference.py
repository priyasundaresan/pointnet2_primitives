"""
Author: Benny
Date: Nov 2019
"""
import argparse
import matplotlib.pyplot as plt
import os
from data_utils.PrimitiveDataLoader import PrimitiveDataset
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

def pc_normalize(pc, centroid=None, m=None):
    pc = pc.squeeze().cpu().numpy()
    if centroid is None:
        centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    if m is None:
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc, centroid, m

def visualize_input(points_xyz, colors_xyz, keypoint_mask_xyz):
    points = points_xyz.squeeze().cpu().numpy()
    colors = colors_xyz.squeeze().cpu().numpy()
    keypoint_mask_xyz = keypoint_mask_xyz.squeeze().cpu().numpy()
    colors[keypoint_mask_xyz > 0] = (0,1,0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd]) # with display

def visualize_pred(points_xyz, colors_xyz, offsets, cls, waypt):
    points = points_xyz.squeeze().cpu().numpy()
    colors = colors_xyz.squeeze().cpu().numpy()

    relevant_idxs = cls > 0
    # Set up offsets 
    distances_vis = np.zeros((len(points), 3))
    distances = np.linalg.norm(offsets, axis=1)
    distances_normalized = (distances - np.amin(distances))/(np.amax(distances) - np.amin(distances))
    distances_vis = plt.cm.jet(distances_normalized)[:,:-1]
    #vis_colors = colors.copy()
    #vis_colors[relevant_idxs] = distances_vis[relevant_idxs]
    vis_colors = distances_vis

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(vis_colors)

    waypoint_vis = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    waypoint_vis.paint_uniform_color([0, 0.0, 0.5])
    pose_transform = np.asarray(
                [[1, 0, 0, waypt[0]],
                 [0, 1, 0, waypt[1]],
                 [0, 0, 1, waypt[2]],
                 [0, 0, 0, 1]])
    waypoint_vis.transform(pose_transform)
    o3d.visualization.draw_geometries([pcd, waypoint_vis]) # with display

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

    root = 'data/dset_open'
    #root = 'data/dset_grasp'

    TEST_DATASET = PrimitiveDataset(root=root, npoints=args.npoint, split='test')
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    num_outputs = 7 # (3 offset + 4 quaternion)
    inp_dim = 7 # (xyz, rgb, keypoint mask)
    num_classes = 2 # (nothing, start waypoint)

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
            points_xyz = points.T[:3,:].T
            colors_xyz = points.T[3:-1,:].T
            keypoint_mask_xyz = points.T[-1,:].T

            points = points.transpose(2, 1)

            pred, offsets, rots = classifier(points) # Classification of relevant points (near the desired waypoint), predicted per-point offsets, predicted per-point quaternions
            pred = torch.softmax(pred.contiguous().view(-1, num_classes), axis=1)
            cls  = pred.data.max(1)[1].cpu().numpy() 

            # Extract predicted gripper pose for relevant points
            offsets = offsets.squeeze().detach().cpu().numpy()
            rots = rots.squeeze().detach().cpu().numpy()
            idxs = np.where(cls == 1)[0] 
            offsets = offsets[:,:3]

            print(offsets.shape)
            rots = rots[:,:4][idxs]
            norm = 1/np.sqrt(np.sum(rots*rots, axis=1))
            rots_norm = np.transpose(np.transpose(rots, (1,0))*norm, (1,0))
            pred_quat = rots_norm[0] # Get the predicted gripper orientation

            points_xyz_centered, centroid, m = pc_normalize(points_xyz)
            waypts = points_xyz_centered[idxs] - offsets[idxs]
            pred_waypt = np.mean(waypts, axis=0)
            pred_waypt *= m
            pred_waypt += centroid

            visualize_input(points_xyz, colors_xyz, keypoint_mask_xyz)
            visualize_pred(points_xyz, colors_xyz, offsets, cls, pred_waypt)

            print('Predicted waypoint and gripper orientation', pred_waypt, pred_quat)
            ctr += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser('PointNet')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in testing')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--npoint', type=int, default=20000, help='point Number')
    parser.add_argument('--log_dir', type=str, default='log/', help='experiment root')
    parser.add_argument('--model', type=str, default='model_cls_off_rot', help='model name')
    args = parser.parse_args()
    
    main(args)
