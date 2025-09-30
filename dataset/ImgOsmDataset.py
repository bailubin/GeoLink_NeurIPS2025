import os
from PIL import Image
import numpy as np
from torch_geometric.data import Dataset, Data
import torch
from torchvision import transforms
import random
from torchvision.transforms import InterpolationMode
import time
from PIL import ImageFilter, ImageOps, Image, ImageDraw
from torch_geometric.data import HeteroData

class ImageOSMHeterDataset(Dataset):
    def __init__(self, data_path, osm_mask_ratio=0.2, mode='train'):
        super(ImageOSMHeterDataset, self).__init__()

        self.img_path = os.path.join(data_path, 'all_img30')
        self.graph_path = os.path.join(data_path, 'all_graph30')
        self.mode = mode
        self.osm_mask_ratio = osm_mask_ratio

        self.RandomResizeCrop = transforms.RandomResizedCrop(224, scale=(0.67, 1.),
                                                        ratio=(3. / 4., 4. / 3.),
                                                        interpolation=InterpolationMode.BICUBIC)
        self.ColorJitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        self.RandomGrayscale = transforms.RandomGrayscale(p=0.2)
        self.RandomHorizontalFlip = transforms.RandomHorizontalFlip(p=1)
        self.resize = transforms.Resize((224, 224))
        self.toTensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        with open(os.path.join(data_path, 'selected.txt'), 'r') as f:
            self.train_files = f.read().splitlines()

    def get(self, index):
        # s_time=time.time()
        file_name = self.train_files[index]
        img = Image.open(os.path.join(self.img_path, file_name[:-2]+'jpg'))
        graph=torch.load(os.path.join(self.graph_path, file_name), weights_only=False)

        img_t, graph = self.random_aug(img, graph)

        return graph, img_t

    def random_mask_node(self, g):
        out_g = g.clone()
        for t in g.node_types:
            n = g[t].x.shape[0]
            # add flag
            out_g[t].x = torch.cat([out_g[t].x, torch.zeros(n, 1)], dim=1)
            if n > 0:
                k = int(round(n * self.osm_mask_ratio))
                k = max(1, min(k, n))

                perm = torch.randperm(n)
                selected = perm[:k]

                out_g[t].x[selected, -1] = 1

        return out_g

    def random_aug(self, img, graph):

        # if random.random()>0.5:
        crop_ratio = np.random.uniform(0.4, 0.9)
        graph, img = self.random_crop_img_and_sample_graph(graph, img, crop_ratio)
        img = self.resize(img)
        # img = self.RandomResizeCrop(img)
        # img = self.RandomGrayscale(img)
        if random.random() > 0.5:
            img = self.RandomHorizontalFlip(img)
            for t in graph.node_types:
                if t == 'polygon':
                    graph[t].x[:, [512, 514, 516, 518]] = 1 - graph[t].x[:, [512, 514, 516, 518]]
                elif t == 'line':
                    graph[t].x[:, [512, 514, 516]] = 1 - graph[t].x[:, [512, 514, 516]]
                elif t == 'point':
                    graph[t].x[:, 512] = 1 - graph[t].x[:, 512]
        if random.random()>0.5:
            img=self.ColorJitter(img)
        img = self.toTensor(img)
        img = self.normalize(img)

        return img, graph

    def compute_bounding_box(self, points, bias=0.005):
        mins, _ = torch.min(points, dim=0)
        maxs, _ = torch.max(points, dim=0)
        adjusted_coords = torch.stack([
            torch.clamp(mins - bias, min=0.0),
            torch.clamp(maxs + bias, max=1.0)
        ])
        # 解包坐标
        x_min, y_min = adjusted_coords[0]
        x_max, y_max = adjusted_coords[1]
        return (x_min.item(), y_min.item(), x_max.item(), y_max.item())

    def random_crop_img_and_sample_graph(self, data, img, crop_ratio=0.75, scale = [0.75, 1.5]):
        sampled_data = HeteroData()
        node_idx_mapping = {}

        aspect_ratio = torch.rand(1).item() * scale[0] + scale[1]
        w = np.sqrt(crop_ratio / aspect_ratio)
        h = aspect_ratio * w

        w = min(w, 1.0)
        h = min(h, 1.0)

        x_min = torch.rand(1).item() * (1 - w)
        y_min = torch.rand(1).item() * (1 - h)
        x_max = x_min + w
        y_max = y_min + h

        coords = []

        for node_type in data.node_types:
            num_nodes = data[node_type].x.size(0)
            if num_nodes > 0:
                # sampled_indices = torch.randperm(num_nodes)[:int(sample_ratio * num_nodes + 1)]
                x_mask = (data[node_type].x[:, 512] >= x_min) & (data[node_type].x[:, 512] <= x_max)
                y_mask = (data[node_type].x[:, 513] >= y_min) & (data[node_type].x[:, 513] <= y_max)
                mask = (x_mask & y_mask) == 1
                sampled_indices = torch.nonzero(mask).squeeze(-1)

                sampled_data[node_type].x = data[node_type].x[sampled_indices]

                mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(sampled_indices)}
                node_idx_mapping[node_type] = mapping
            else:
                sampled_data[node_type].x = data[node_type].x

            if node_type == 'polygon':
                coords.append(sampled_data[node_type].x[:, 512:520].reshape(-1, 2))
            elif node_type == 'line':
                coords.append(sampled_data[node_type].x[:, 512:518].reshape(-1, 2))
            else:
                coords.append(sampled_data[node_type].x[:, 512:514].reshape(-1, 2))
        coords = torch.concat(coords)
        if coords.shape[0] == 0:
            return data, img

        coords = torch.concat([coords, torch.tensor([[x_min, y_min], [x_max, y_max]])])
        bbox = self.compute_bounding_box(coords)

        for edge_type in data.edge_types:
            src_type, _, dst_type = edge_type
            edge_index = data[edge_type].edge_index
            if edge_index.size(1) > 0:
                edge_attr = data[edge_type].edge_attr if 'edge_attr' in data[edge_type] else None

                src_mapping = node_idx_mapping[src_type]
                dst_mapping = node_idx_mapping[dst_type]

                valid_edges = []
                valid_edge_attrs = []

                for i in range(edge_index.size(1)):
                    src, dst = edge_index[:, i]
                    if src.item() in src_mapping and dst.item() in dst_mapping:
                        new_src = src_mapping[src.item()]
                        new_dst = dst_mapping[dst.item()]
                        valid_edges.append([new_src, new_dst])

                        if edge_attr is not None:
                            valid_edge_attrs.append(edge_attr[i])

                if valid_edges:
                    sampled_data[edge_type].edge_index = torch.tensor(valid_edges, dtype=torch.long).t()
                    if edge_attr is not None:
                        sampled_data[edge_type].edge_attr = torch.stack(valid_edge_attrs)
                else:
                    sampled_data[edge_type].edge_index = torch.empty(edge_index.shape[0], 0, dtype=torch.long)
                    sampled_data[edge_type].edge_attr = torch.empty(0, edge_attr.shape[1], dtype=torch.float)
            else:
                sampled_data[edge_type].edge_index = edge_index
                sampled_data[edge_type].edge_attr = data[edge_type].edge_attr

        final_w, final_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        for node_type in data.node_types:
            if node_type == 'polygon':
                sampled_data[node_type].x[:, [512, 514, 516, 518]] = (sampled_data[node_type].x[:,
                                                                      [512, 514, 516, 518]] - bbox[0]) / final_w
                sampled_data[node_type].x[:, [513, 515, 517, 519]] = (sampled_data[node_type].x[:,
                                                                      [513, 515, 517, 519]] - bbox[1]) / final_h
            elif node_type == 'line':
                sampled_data[node_type].x[:, [512, 514, 516]] = (sampled_data[node_type].x[:, [512, 514, 516]] - bbox[
                    0]) / final_w
                sampled_data[node_type].x[:, [513, 515, 517]] = (sampled_data[node_type].x[:, [513, 515, 517]] - bbox[
                    1]) / final_h
            else:
                sampled_data[node_type].x[:, 512] = (sampled_data[node_type].x[:, 512] - bbox[0]) / final_w
                sampled_data[node_type].x[:, 513] = (sampled_data[node_type].x[:, 513] - bbox[1]) / final_h

        img_w, img_h = img.size
        crop_box = (
            int(bbox[0] * img_w + 0.5),
            int(bbox[1] * img_h + 0.5),
            int(bbox[2] * img_w + 0.5),
            int(bbox[3] * img_h + 0.5))
        cropped_img = img.crop(crop_box)
        return sampled_data, cropped_img

    def len(self):
        return len(self.train_files)
