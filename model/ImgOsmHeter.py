import torch
import torch.nn as nn
import torch_geometric
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_from_grid, get_1d_sincos_pos_embed_from_grid
from .transformer import TwoWayAttentionBlock_Attention
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear
import numpy as np



class OSMHeteroGAT(torch.nn.Module):
    def __init__(self, out_chans=128):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        conv1 = HeteroConv({
            ('point', 'to', 'point'): GATConv(512, out_chans, add_self_loops=False),
            ('point', 'to', 'line'): GATConv(512, out_chans, add_self_loops=False),
            ('point', 'to', 'polygon'): GATConv(512, out_chans, add_self_loops=False),
            ('line', 'to', 'line'): GATConv(512, out_chans, add_self_loops=False),
            ('line', 'to', 'point'): GATConv(512, out_chans, add_self_loops=False),
            ('line', 'to', 'polygon'): GATConv(512, out_chans, add_self_loops=False),
            ('polygon', 'to', 'polygon'): GATConv(512, out_chans, add_self_loops=False),
            ('polygon', 'to', 'point'): GATConv(512, out_chans, add_self_loops=False),
            ('polygon', 'to', 'line'): GATConv(512, out_chans, add_self_loops=False),
        }, aggr='mean')
        self.convs.append(conv1)

    def forward(self, data):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict, edge_attr_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return x_dict



class GeoLink(nn.Module):

    def __init__(self, img_encoder, osm_encoder, img_in_chans=3,
                 osm_in_chans=512, osm_out_embed_dim=128, graph_aggr_embed_dim=128,
                 fusion_embed_dim=256, img_decoder_embed_dim=512,
                 consistency_embed_dim=64, img_decoder_depth=2, img_decoder_num_heads=16,
                 osm_cross_modal_encoder_depth=1, osm_cross_modal_encoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=True):
        super().__init__()

        self.osm_types = ['point', 'line', 'polygon']
        # --------------------------------------------------------------------------
        # Image encoder specifics
        self.img_encoder = img_encoder
        self.patch_embed = self.img_encoder.patch_embed
        num_patches = self.patch_embed.num_patches
        patch_size = self.patch_embed.patch_size[0]
        img_embed_dim = self.img_encoder.embed_dim
        self.grid_size=int(num_patches ** .5)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, img_embed_dim))
        self.img_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, img_embed_dim), requires_grad=False)
        self.img_norm = norm_layer(img_embed_dim)

        self.osm_encoder = osm_encoder
        self.osm_out_embed_dim = osm_out_embed_dim
        self.osm_norm = {}
        for t in self.osm_types:
            self.osm_norm[t] = torch_geometric.nn.LayerNorm(osm_out_embed_dim)
        self.osm_norm = nn.ModuleDict(self.osm_norm)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Image reconstruction specifics
        self.img_decoder_embed_dim = img_decoder_embed_dim
        self.rec_img_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, img_decoder_embed_dim), requires_grad=False)

        self.img_decoder_mlp = nn.Linear(img_embed_dim, img_decoder_embed_dim, bias=True)
        self.img_mask_token = nn.Parameter(torch.zeros(1, 1, img_decoder_embed_dim))
        self.img_cross_modal_decoder = nn.ModuleList([
            Block(img_decoder_embed_dim, img_decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(img_decoder_depth)])
        self.img_decoder_norm = norm_layer(img_decoder_embed_dim)
        self.img_decoder_pred = nn.Linear(img_decoder_embed_dim, patch_size ** 2 * img_in_chans,
                                          bias=True)  # decoder to patch

        # ---------------------------------------------------------------------------
        # OSM aggregation
        self.graph_aggr_embed_dim = graph_aggr_embed_dim
        self.osm_read_out = {}
        self.osm_aggr_mlp = {}
        for t in self.osm_types:
            self.osm_read_out[t] = torch_geometric.nn.Set2Set(osm_out_embed_dim, processing_steps=5)
            self.osm_aggr_mlp[t] = nn.Linear(int(osm_out_embed_dim*2), graph_aggr_embed_dim, bias=True)
        self.osm_read_out = nn.ModuleDict(self.osm_read_out)
        self.osm_aggr_mlp = nn.ModuleDict(self.osm_aggr_mlp)
        self.type_attention = nn.Linear(graph_aggr_embed_dim, 1)
        self.osm_graph_consistency_prj = nn.Linear(graph_aggr_embed_dim, consistency_embed_dim, bias=True)
        self.img_consistency_prj = nn.Linear(img_embed_dim, consistency_embed_dim, bias=True)

        # --------------------------------------------------------------------------
        # Osm fusion encoder and decoder specifics
        self.fusion_embed_dim = fusion_embed_dim
        self.fusion_img_pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, fusion_embed_dim),requires_grad=False)
        self.img_fusion_mlp = nn.Linear(img_embed_dim, fusion_embed_dim, bias=True)
        self.osm_node_fusion_mlp = {}
        self.osm_mask_token = {}

        for t in self.osm_types:
            self.osm_node_fusion_mlp[t] = nn.Linear(osm_out_embed_dim, fusion_embed_dim, bias=True)
            self.osm_mask_token[t] = torch.nn.Parameter(torch.zeros(1, osm_in_chans))

        self.osm_node_fusion_mlp = nn.ModuleDict(self.osm_node_fusion_mlp)
        self.osm_mask_token = nn.ParameterDict(self.osm_mask_token)

        self.osm_cross_modal_encoder = nn.ModuleList([
            TwoWayAttentionBlock_Attention(
                embedding_dim=fusion_embed_dim,
                num_heads=osm_cross_modal_encoder_num_heads,
                dropout=0.1)
            for i in range(osm_cross_modal_encoder_depth)
        ])
        self.osm_node_pred = nn.Linear(fusion_embed_dim, osm_in_chans, bias=True)
        # --------------------------------------------------------------------------

        # -------------------------------------------------------------------------
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        img_pos_embed = get_2d_sincos_pos_embed(self.img_pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                                cls_token=True)
        self.img_pos_embed.data.copy_(torch.from_numpy(img_pos_embed).float().unsqueeze(0))

        rec_img_pos_embed = get_2d_sincos_pos_embed(self.rec_img_pos_embed.shape[-1],
                                                      int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.rec_img_pos_embed.data.copy_(torch.from_numpy(rec_img_pos_embed).float().unsqueeze(0))

        fusion_img_pos_embed = get_2d_sincos_pos_embed(self.fusion_img_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.fusion_img_pos_embed.data.copy_(torch.from_numpy(fusion_img_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.img_encoder.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.img_mask_token, std=.02)
        for t in self.osm_types:
            torch.nn.init.normal_(self.osm_mask_token[t], std=.02)

        # initialize nn.Linear and nn.LayerNorm
        # self.apply(self._init_weights)
        for name,module in self.named_children():
            if 'img_encoder' not in name:
                module.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_mask_img(self, x, mask_ratio, mask_scale=2):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise_L = L // (mask_scale ** 2)
        noise = np.random.rand(N, noise_L).reshape(N, int(noise_L ** 0.5), int(noise_L ** 0.5))
        noise = noise.repeat(mask_scale, axis=1).repeat(mask_scale, axis=2)
        noise = noise.reshape(N, -1)
        noise = torch.from_numpy(noise).to(x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # get img_pos_embeding for fusion
        fusion_img_pos_embed = self.fusion_img_pos_embed[:, 1:].expand(N, -1, -1)
        fusion_img_pos_embed_masked = torch.gather(fusion_img_pos_embed, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, fusion_img_pos_embed.shape[-1]))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, fusion_img_pos_embed_masked, mask, ids_restore

    def random_mask_node(self, g, device, mask_ratio):
        out_g = g.clone()
        mask_nodes = {}
        for t in g.node_types:
            batch = g[t].batch
            # subgraph node num
            num_nodes_per_graph = torch.bincount(batch)
            num_graphs = num_nodes_per_graph.size(0)

            starts = torch.cat([
                torch.zeros(1, dtype=torch.long, device=device),
                torch.cumsum(num_nodes_per_graph, dim=0)[:-1]
            ])
            mask_indices = []
            for i in range(num_graphs):
                n = num_nodes_per_graph[i].item()
                if n>0:
                    start = starts[i].item()

                    k = int(round(n * mask_ratio))
                    k = max(1, min(k, n))

                    perm = torch.randperm(n, device=device)
                    selected = perm[:k]

                    global_selected = selected + start
                    mask_indices.append(global_selected)


            mask_indices = torch.cat(mask_indices) if mask_indices else torch.tensor([], device=batch.device)
            out_g[t].x[mask_indices] = 0.0
            out_g[t].x[mask_indices] += self.osm_mask_token[t]
            mask_nodes[t] = mask_indices

        return out_g, mask_nodes

    def pad_node(self, node_features, node_pos_embed, node_label, batch, max_sample_nodes = 10):
        num_graphs = batch.max().item() + 1
        node_counts = batch.bincount()  # Number of nodes per graph
        max_nodes = min(node_counts.max(dim=0)[0], max_sample_nodes)
        padding_mask = torch.arange(max_nodes, device=node_counts.device).unsqueeze(0) < node_counts.unsqueeze(1)  # (num_graphs, max_nodes) from True to False

        padded_node_features = torch.zeros((num_graphs, max_nodes, node_features.size(1)), device=node_features.device)
        padded_node_pos_embed = torch.zeros((num_graphs, max_nodes, node_pos_embed.size(1)), device=node_features.device)
        padded_node_label = torch.zeros((num_graphs, max_nodes, node_label.size(1)),
                                            device=node_features.device)
        # Fill in padded_node_features using the batch tensor
        for i in range(num_graphs):
            # Get the indices of the nodes that belong to the i-th graph
            graph_node_indices = (batch == i).nonzero(as_tuple=True)[0]
            # # if more than max_nodes, random select max_nodes
            if graph_node_indices.shape[0] > max_nodes:
                graph_node_indices = graph_node_indices[torch.randperm(graph_node_indices.shape[0])[:max_nodes]]

            # Copy the node features into the padded tensor
            padded_node_features[i, :len(graph_node_indices)] = node_features[graph_node_indices]
            padded_node_pos_embed[i, :len(graph_node_indices)] = node_pos_embed[graph_node_indices]
            padded_node_label[i, :len(graph_node_indices)] = node_label[graph_node_indices]

        return padded_node_features, padded_node_pos_embed, padding_mask, padded_node_label

    def get_node_pos(self, node_pos, t, embed_dim, device):
        if t == 'polygon':
            node_pos = node_pos * (self.patch_embed.num_patches ** .5)
            node_pos = node_pos.reshape(-1, 4, 2)
            node_pos = node_pos.transpose(2, 0, 1).reshape(2, -1)
            node_pos_embedding = get_2d_sincos_pos_embed_from_grid(embed_dim, grid=node_pos)
            node_pos_embedding = node_pos_embedding.reshape(-1, 4, embed_dim)
            node_pos_embedding = node_pos_embedding.mean(axis=1)
            node_pos_embedding = torch.from_numpy(node_pos_embedding).to(device).float()
            return node_pos_embedding
        elif t == 'line':
            node_pos = node_pos * (self.patch_embed.num_patches ** .5)
            node_pos = node_pos.reshape(-1, 3, 2)
            node_pos = node_pos.transpose(2, 0, 1).reshape(2, -1)
            node_pos_embedding = get_2d_sincos_pos_embed_from_grid(embed_dim, grid=node_pos)
            node_pos_embedding = node_pos_embedding.reshape(-1, 3, embed_dim)
            node_pos_embedding = node_pos_embedding.mean(axis=1)
            node_pos_embedding = torch.from_numpy(node_pos_embedding).to(device).float()
            return node_pos_embedding
        elif t == 'point':
            node_pos = node_pos * (self.patch_embed.num_patches ** .5)  # relative pos to absolute pos+
            node_pos = node_pos.transpose(1, 0)
            node_pos_embedding = get_2d_sincos_pos_embed_from_grid(embed_dim, grid=node_pos)
            node_pos_embedding = torch.from_numpy(node_pos_embedding).to(device).float()
            return node_pos_embedding


    def forward_img_encoder(self, x, mask_ratio, mask_scale=2):
        # embed patches
        x = self.img_encoder.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.img_pos_embed[:, 1:, :]

        if mask_ratio > 0:
            # masking: length -> length * mask_ratio
            x, fusion_img_pos_embed_masked, mask, ids_restore = self.random_mask_img(x, mask_ratio, mask_scale)
        else:
            fusion_img_pos_embed_masked, mask, ids_restore = self.fusion_img_pos_embed[:, 1:].expand(x.shape[0], -1, -1), None, None

        # append cls token
        cls_token = self.cls_token + self.img_pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.img_encoder.blocks:
            x = blk(x)
        x = self.img_norm(x)

        return x, fusion_img_pos_embed_masked, mask, ids_restore

    def forward_osm_encoder(self, osms, node_pos, mask_ratio):
        device = self.img_pos_embed.device
        # use_osms=osms.clone()

        if mask_ratio > 0:
            use_osms, mask_nodes = self.random_mask_node(osms, device, mask_ratio)
        else:
            use_osms = osms.clone()
            mask_nodes = None


        out = self.osm_encoder(use_osms)
        out = {key: self.osm_norm[key](x) for key, x in out.items()}

        return out, mask_nodes

    def forward_img_decoder(self, img_embedding, ids_restore):
        # embed tokens
        img_embedding = self.img_decoder_mlp(img_embedding)

        # append mask tokens to sequence
        mask_tokens = self.img_mask_token.repeat(img_embedding.shape[0], ids_restore.shape[1] + 1 - img_embedding.shape[1], 1)
        x_ = torch.cat([img_embedding[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, img_embedding.shape[2]))  # unshuffle
        img_embedding = torch.cat([img_embedding[:, :1, :], x_], dim=1)  # append cls token

        # apply Decoder Transformer blocks
        img_embedding = img_embedding + self.rec_img_pos_embed
        for blk in self.img_cross_modal_decoder:
            img_embedding = blk(img_embedding)

        img_embedding = self.img_decoder_norm(img_embedding)
        # predictor projection
        out = self.img_decoder_pred(img_embedding)
        # remove cls token
        out = out[:, 1:, :]

        return out


    def forward_osm_cross_encoder_decoder(self, osms, osm_node_embeddings, img_embedding, mask_nodes, node_pos, fusion_img_pos_embed_masked):
        device = self.img_pos_embed.device

        node_pos_embedding, osm_mask_embedding, node_label, mask_batch_index = [], [], [], []
        for t in mask_nodes.keys():
            node_pos_embedding_t = self.get_node_pos(node_pos[t][mask_nodes[t].cpu().numpy()], t, self.fusion_embed_dim, device)

            # get osm mask embeddings
            osm_mask_embedding_t = osm_node_embeddings[t][mask_nodes[t]]
            osm_mask_embedding_t = self.osm_node_fusion_mlp[t](osm_mask_embedding_t)
            node_label_t = osms[t].x[mask_nodes[t]]

            mask_batch_index_t = osms[t].batch[mask_nodes[t]]

            node_pos_embedding.append(node_pos_embedding_t)
            osm_mask_embedding.append(osm_mask_embedding_t)
            node_label.append(node_label_t)
            mask_batch_index.append(mask_batch_index_t)

        node_pos_embedding = torch.concat(node_pos_embedding, dim=0)
        osm_mask_embedding = torch.concat(osm_mask_embedding, dim=0)
        node_label = torch.concat(node_label, dim=0)
        mask_batch_index = torch.concat(mask_batch_index, dim=0)

        padded_osm_mask_embedding, padded_node_pos_embedding, padding_mask, padded_node_label = self.pad_node(osm_mask_embedding, node_pos_embedding,
                                                                                           node_label, mask_batch_index) # B*mask_node_num*dim

        # repeat img mask embeddings
        img_embedding = self.img_fusion_mlp(img_embedding)
        # print(img_embedding.shape, padded_osm_mask_embedding.shape, padded_node_pos_embedding.shape)

        # apply Cross-modal Transformer blocks
        for blk in self.osm_cross_modal_encoder:
            img_embedding, osm_mask_embedding = blk(img_embedding, padded_osm_mask_embedding, fusion_img_pos_embed_masked,
                                                    padded_node_pos_embedding, ~padding_mask)

        return osm_mask_embedding[padding_mask], padded_node_label[padding_mask]

    def osm_aggregation(self, osms, osm_node_embedding, batch_size, device, dtype):
        osm_batch_dict = osms.batch_dict
        osm_graph_embedding = {}
        osm_embeddings = torch.zeros((batch_size, 3, int(self.graph_aggr_embed_dim)), dtype=torch.bfloat16).to(device)
        for t in osm_node_embedding.keys():
            if osm_node_embedding[t].shape[0] > 0:
                osm_graph_embedding_t = self.osm_read_out[t](osm_node_embedding[t],
                                                             osms[t].batch.long()).float()
                batch_dict_t = osm_batch_dict[t].unique().long()
                osm_graph_embedding_t = self.osm_aggr_mlp[t](osm_graph_embedding_t[batch_dict_t])

                if t == 'polygon':
                    osm_embeddings[batch_dict_t, 0] = osm_graph_embedding_t
                elif t == 'line':
                    osm_embeddings[batch_dict_t, 1] = osm_graph_embedding_t
                elif t == 'point':
                    osm_embeddings[batch_dict_t, 2] = osm_graph_embedding_t

        osm_embedding_w = torch.softmax(self.type_attention(osm_embeddings), dim=-1)
        osm_embedding_aggr = torch.sum(osm_embedding_w * osm_embeddings, dim=1, keepdim=False)
        return osm_embedding_aggr

    def forward_img_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        # loss = torch.abs(pred - target)
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_contrastive_loss(self, img_features, graph_features, t=0.2):

        # L2 normalize
        img_features = img_features / img_features.norm(dim=1, keepdim=True)
        graph_features = graph_features / graph_features.norm(dim=1, keepdim=True)


        # dot product to get logits
        logits_per_img = img_features @ graph_features.t() /t
        logits_per_graph = graph_features @ img_features.t() / t

        # organize labels
        num_logits = logits_per_img.shape[0]
        labels = torch.arange(num_logits, device=img_features.device, dtype=torch.long)
        labels = labels

        # calculate loss
        loss = (F.cross_entropy(logits_per_img, labels) + F.cross_entropy(logits_per_graph, labels)) / 2
        return loss

    def forward_geo_loss(self, label, pred):
        loss=(label-pred)**2
        return loss.mean(dim=-1).mean()

    def forward(self, imgs_t, osms, img_mask_ratio=0.75, osm_mask_ratio=0.2):
        device = self.img_pos_embed.device
        node_pos = {}
        for t in osms.node_types:
            if t == 'polygon':
                node_pos[t] = osms[t].x[:, 512:520].clone().detach().cpu().numpy()
                osms[t].x = osms[t].x[:, :512]
            elif t == 'line':
                node_pos[t] = osms[t].x[:, 512:518].clone().detach().cpu().numpy()
                osms[t].x = osms[t].x[:, :512]
            elif t == 'point':
                node_pos[t] = osms[t].x[:, 512:514].clone().detach().cpu().numpy()
                osms[t].x = osms[t].x[:, :512]
                # print(osms[t].x.shape)

        # reconstruct img
        # get masked img embedding
        img_embedding, fusion_img_pos_embed_masked, mask_imgs, ids_restore = self.forward_img_encoder(imgs_t, mask_ratio=img_mask_ratio, mask_scale=1)
        pred_imgs = self.forward_img_decoder(img_embedding, ids_restore)
        img_loss = self.forward_img_loss(imgs_t, pred_imgs, mask_imgs)

        # get original osm embedding
        osm_node_embeddings, mask_nodes = self.forward_osm_encoder(osms, node_pos, mask_ratio=osm_mask_ratio)
        osm_aggr_embedding = self.osm_aggregation(osms, osm_node_embeddings, batch_size=img_embedding.shape[0],
                                                    device=device, dtype=img_embedding.dtype)



        # img-graph level consitency loss
        img_embedding_consistency = self.img_consistency_prj(img_embedding[:, 1:].mean(dim=1))

        osm_aggr_embedding_aug_consistency = self.osm_graph_consistency_prj(osm_aggr_embedding)

        graph_level_contrastive_loss = self.forward_contrastive_loss(img_embedding_consistency, osm_aggr_embedding_aug_consistency)

        # get masked osm embedding
        pred_osm, label_osm = self.forward_osm_cross_encoder_decoder(osms, osm_node_embeddings, img_embedding[:, 1:],
                                                                     mask_nodes,
                                                                     node_pos, fusion_img_pos_embed_masked)

        pred_osm = self.osm_node_pred(pred_osm)
        node_level_rec_loss = self.forward_geo_loss(label_osm, pred_osm)

        return img_loss, graph_level_contrastive_loss, node_level_rec_loss, pred_imgs, mask_imgs, pred_osm, mask_nodes


    def get_osm_to_img_attn(self, imgs, osms, mask_node_inds, mask_t):
        device = self.img_pos_embed.device
        node_pos = {}
        for t in osms.node_types:
            if t == 'polygon':
                node_pos[t] = osms[t].x[:, 512:520].clone().detach().cpu().numpy()
                osms[t].x = osms[t].x[:, :512]
            elif t == 'line':
                node_pos[t] = osms[t].x[:, 512:518].clone().detach().cpu().numpy()
                osms[t].x = osms[t].x[:, :512]
            elif t == 'point':
                node_pos[t] = osms[t].x[:, 512:514].clone().detach().cpu().numpy()
                osms[t].x = osms[t].x[:, :512]
                # print(osms[t].x.shape)

        use_g = osms.clone()
        # use_g[mask_t].x[mask_node_inds] = 0.0
        # use_g[mask_t].x[mask_node_inds] += self.osm_mask_token[mask_t]

        img_embedding, fusion_img_pos_embed, _, _ = self.forward_img_encoder(imgs, mask_ratio=0)
        osm_node_embeddings, _ = self.forward_osm_encoder(use_g, node_pos, mask_ratio=0)

        device = self.img_pos_embed.device

        node_pos = node_pos[mask_t]
        node_pos_embedding = self.get_node_pos(node_pos, mask_t, self.fusion_embed_dim, device)[mask_node_inds].reshape(-1, 1, self.fusion_embed_dim).float()
        # get osm mask embeddings
        osm_mask_embedding = osm_node_embeddings[mask_t][mask_node_inds]
        osm_mask_embedding = osm_mask_embedding.reshape(1, -1, osm_mask_embedding.shape[-1])
        osm_mask_embedding = self.osm_node_fusion_mlp[mask_t](osm_mask_embedding)

        # repeat img mask embeddings
        img_embedding = self.img_fusion_mlp(img_embedding)
        # img_embedding = img_embedding.repeat(osm_mask_embedding.shape[0], 1, 1)
        # print(img_embedding.shape, osm_mask_embedding.shape, node_pos_embedding.shape)

        # apply Cross-modal Transformer blocks
        attns=[]
        for blk in self.osm_cross_modal_encoder:
            img_embedding, osm_mask_embedding, attn = blk(img_embedding[:, 1:], osm_mask_embedding, fusion_img_pos_embed,
                                                    node_pos_embedding, require_attn=True)
            attns.append(attn)
        return attns

    def get_img_embedding(self, imgs, osms):
        device = self.img_pos_embed.device
        node_pos = {}
        for t in osms.node_types:
            if t == 'polygon':
                node_pos[t] = osms[t].x[:, 512:520].clone().detach().cpu().numpy()
                osms[t].x = osms[t].x[:, :512]
            elif t == 'line':
                node_pos[t] = osms[t].x[:, 512:518].clone().detach().cpu().numpy()
                osms[t].x = osms[t].x[:, :512]
            elif t == 'point':
                node_pos[t] = osms[t].x[:, 512:514].clone().detach().cpu().numpy()
                osms[t].x = osms[t].x[:, :512]
                # print(osms[t].x.shape)

        use_g = osms.clone()

        img_embedding, fusion_img_pos_embed, _, _ = self.forward_img_encoder(imgs, mask_ratio=0)
        osm_node_embeddings, _ = self.forward_osm_encoder(use_g, node_pos, mask_ratio=0)

        device = self.img_pos_embed.device

        node_pos_embedding, osm_fusion_embedding, batch_index = [], [], []
        for t in osms.node_types:
            node_pos_embedding_t = self.get_node_pos(node_pos[t], t, self.fusion_embed_dim, device)

            # get osm mask embeddings
            osm_embedding_t = osm_node_embeddings[t]
            osm_fusion_t = self.osm_node_fusion_mlp[t](osm_embedding_t)

            # batch_index_t = osms[t].batch

            node_pos_embedding.append(node_pos_embedding_t)
            osm_fusion_embedding.append(osm_fusion_t)
            # batch_index.append(batch_index_t)

        node_pos_embedding = torch.concat(node_pos_embedding, dim=0).reshape(1, -1, self.fusion_embed_dim)
        osm_fusion_embedding = torch.concat(osm_fusion_embedding, dim=0).reshape(1, -1, self.fusion_embed_dim)
        # batch_index = torch.concat(batch_index, dim=0)

        # padded_osm_mask_embedding, padded_node_pos_embedding, padding_mask = \
        #     self.pad_node(osm_fusion_embedding, node_pos_embedding, batch_index)  # B*mask_node_num*dim

        # repeat img mask embeddings
        img_embedding_fusion = self.img_fusion_mlp(self.img_norm(img_embedding[:, 1:]))
        # apply Cross-modal Transformer blocks
        for blk in self.osm_cross_modal_encoder:
            img_embedding_fusion, osm_mask_embedding = blk(img_embedding_fusion, osm_fusion_embedding,
                                                    fusion_img_pos_embed,
                                                    node_pos_embedding)
        return img_embedding[:, 1:], img_embedding_fusion











