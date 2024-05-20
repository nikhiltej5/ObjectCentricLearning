import argparse
import infer_helper as helper
import torch
import torchvision
import os
import math

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir')
parser.add_argument('--model_path')
parser.add_argument('--part')
parser.add_argument('--output_dir')
args = parser.parse_args()

img_dir = args.input_dir
model_path = args.model_path
part = int(args.part)
out_dir = args.output_dir

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

transform = helper.DataSet(helper.param.resolution, img_dir).transform

if part == 1:
    model = helper.ObjectDiscovery(param=helper.param, device=helper.device).to(helper.device)
    log = model.load_state_dict(torch.load(model_path), strict=False)
    print(log)
    model.eval()

    with torch.no_grad():
        for img_name in os.listdir(img_dir):
            img_name_without_extension = img_name[:-4]
            img = torchvision.datasets.folder.default_loader(os.path.join(img_dir, img_name))
            img = transform(img).to(torch.float)
            img = img.unsqueeze(0).to(helper.device)
            recon_img, img, mask, slots = model(img)

            torchvision.utils.save_image(recon_img[0], os.path.join(out_dir, f'{img_name_without_extension}.png'))

            for i in range(mask.shape[1]):
                torchvision.utils.save_image(mask[0, i, :, :], os.path.join(out_dir, f'{img_name_without_extension}_{i}.png'))
            
            print("Completed", img_name)

if part == 2:
    model = helper.SlotDiffusion(param=helper.param, device=helper.device, load_vae = False).to(helper.device)
    log = model.load_state_dict(torch.load(model_path), strict=False)
    print(log)
    model.eval()


    with torch.no_grad():
        for img_name in os.listdir(img_dir):
            img_name_without_extension = img_name[:-4]
            img = torchvision.datasets.folder.default_loader(os.path.join(img_dir, img_name))
            output_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(img.size, antialias=False)
            ])
            img = transform(img).to(torch.float)
            img = img.unsqueeze(0).to(helper.device)
            recon_img, attn_map = model(img, infer=True)

            torchvision.utils.save_image(recon_img[0], os.path.join(out_dir, f'{img_name_without_extension}.png'))

            attn_map = attn_map[:, 2, :, :]
            B, N, K = attn_map.shape
            dim = math.isqrt(N)
            new_map = []
            attn_map = attn_map.reshape(B, dim, dim, K).permute(0, 3, 1, 2)

            for i in range(attn_map.shape[1]):
                new_map.append(output_transform(attn_map[:, i, :, :])[0])
            
            mask = torch.stack(new_map)

            for i in range(mask.shape[0]):
                torchvision.utils.save_image(mask[i, :, :], os.path.join(out_dir, f'{img_name_without_extension}_{i}.png'))
            
            print("Completed", img_name)
