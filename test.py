import torch
import os
import argparse
from datasets.crowd import Crowd
from models.builder import MultiEncoderDecoder
from utils.evaluation import eval_game, eval_relative
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--data-dir', default='',
                            help='training data directory')
    parser.add_argument('--save-dir', default='',
                            help='model directory')
    parser.add_argument('--dataset', default='', choices=['RGBTCC', 'DroneRGBT'],
                        help='dataset name')
    parser.add_argument('--backbone', default='swin_b', choices=['swin_b'],
                        help='backbone name')
    parser.add_argument('--pretrained-model', default='',
                        help='the path of pretrained model')
    parser.add_argument('--bn-eps', type=float, default=1e-3,
                        help='batch normalization epsilon')
    parser.add_argument('--bn-momentum', type=float, default=0.1,
                        help='batch normalization momentum')
    parser.add_argument('--device', default='0', help='gpu device')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu

    for phase in ['val']:
        datasets = Crowd(os.path.join(args.data_dir, phase), args.dataset, 256, 8, phase)
        dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                                 num_workers=8, pin_memory=False)
    
        device = torch.device('cuda')
        model = MultiEncoderDecoder(args)
        model.to(device)
        model.eval()
        model.load_state_dict(torch.load(args.save_dir, device))

        # Iterate over data.
        game = [0, 0, 0, 0]
        mse = [0, 0, 0, 0]
        total_relative_error = 0
        
        for inputs, target, name, value in dataloader:
            if type(inputs) == list:
                inputs[0] = inputs[0].to(device)
                inputs[1] = inputs[1].to(device)
            else:
                inputs = inputs.to(device)
            value = value.to(device)

            # inputs are images with different sizes
            if type(inputs) == list:
                assert inputs[0].size(0) == 1
            else:
                assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
            with torch.set_grad_enabled(False):
                _, _, outputs, _, _, _ = model(inputs, value)

                for L in range(4):
                    abs_error, square_error = eval_game(outputs, target, L)
                    game[L] += abs_error
                    mse[L] += square_error
                relative_error = eval_relative(outputs, target)
                total_relative_error += relative_error
            
            print(name[0].split('_')[0], target.sum().numpy(), outputs.sum().cpu().numpy().round(2))
            
            rgb_path = os.path.join(args.data_dir, phase, name[0].split('_')[0] + '_RGB.jpg')
            t_path = os.path.join(args.data_dir, phase, name[0].split('_')[0] + '_T.jpg')
            density_path = os.path.join(args.data_dir, phase, name[0].split('_')[0] + '_density.npy')
            rgb = cv2.imread(rgb_path)[..., ::-1].copy()
            bgr = cv2.imread(rgb_path)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            value = hsv[:,:,2]
            t = cv2.imread(t_path)[..., ::-1].copy()
            density = np.load(density_path)
            outputs = outputs.cpu().numpy().squeeze()
            
            if False:
                plt.subplot(221)
                plt.imshow(rgb)
                plt.axis('off')
                plt.subplot(222)
                plt.imshow(t)
                plt.axis('off')
                plt.subplot(223)
                plt.imshow(density)
                plt.axis('off')
                string = str(target.sum().numpy().round(2))
                plt.text(12, 60, string, fontsize=10, color = 'y')
                plt.subplot(224)
                plt.imshow(outputs)
                plt.axis('off')
                string2 = str(outputs.sum().round(2))
                plt.text(3, 7, string2, fontsize=10, color = 'y')
                plt.show()
                plt.clf()


        N = len(dataloader)
        game = [m / N for m in game]
        mse = [torch.sqrt(m / N) for m in mse]
        total_relative_error = total_relative_error / N

        log_str = 'Epoch 0 Test{}, GAME0 {game0:.2f} GAME1 {game1:.2f} GAME2 {game2:.2f} GAME3 {game3:.2f} MSE {mse:.2f}, ' \
                     .format(N, game0=game[0], game1=game[1], game2=game[2], game3=game[3], mse=mse[0])
        print(log_str)
        
