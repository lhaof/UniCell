# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.init import normal_

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from mmdet.utils import OptConfigType
from mmdet.models.layers import SinePositionalEncoding, DeformableDetrTransformerEncoder
from ...deformable_detr_nuclei import DeformableDETR_Nuclei, MultiScaleDeformableAttention
from ..dino_layers_multihead_nuclei import CdnQueryGenerator_MultiHead, DinoTransformerDecoder_MultiHead
from ..ClsPrompt.tokenizer import SimpleTokenizer, Tokenize
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmengine.model import BaseModule, ModuleList
from mmcv.cnn import build_norm_layer
from einops import rearrange


@MODELS.register_module()
class DINO_Nuclei_MultiHead_Feature_4D_CMOL_l_Layers(DeformableDETR_Nuclei):
    r"""Implementation of `DINO: DETR with Improved DeNoising Anchor Boxes
    for End-to-End Object Detection <https://arxiv.org/abs/2203.03605>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/DINO>`_.

    Args:
        dn_cfg (:obj:`ConfigDict` or dict, optional): Config of denoising
            query generator. Defaults to `None`.
    """

    def __init__(self, *args, dn_cfg: OptConfigType = None, text_cfg: OptConfigType = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert self.as_two_stage, 'as_two_stage must be True for DINO'
        assert self.with_box_refine, 'with_box_refine must be True for DINO'

        if dn_cfg is not None:
            assert 'num_classes' not in dn_cfg and \
                   'num_queries' not in dn_cfg and \
                   'hidden_dim' not in dn_cfg, \
                'The three keyword args `num_classes`, `embed_dims`, and ' \
                '`num_matching_queries` are set in `detector.__init__()`, ' \
                'users should not set them in `dn_cfg` config.'
            dn_cfg['num_classes'] = self.bbox_head.num_classes
            dn_cfg['embed_dims'] = self.embed_dims
            dn_cfg['num_matching_queries'] = self.num_queries
        self.dn_query_generator = CdnQueryGenerator_MultiHead(**dn_cfg)
        self.init_text_prompt(**text_cfg)

    def init_text_prompt(self, max_seg_len, context_length, width, layers, vocab_size,
                         proj_num_layers, n_ctx, n_query_prompt, **kwargs):
        self._tokenizer = Tokenize(SimpleTokenizer(), max_seq_len=max_seg_len)
        self.dataset_mlp = MLP(context_length, width, width, 2)

        self.categories_mlp = MLP(context_length, width, width, 2)

        dataset_names = ["CoNSeP", "MoNuSAC", "OCELOT", "Lizard"]
        dataset_n_ctx = n_ctx
        ctx_vectors = torch.empty((dataset_n_ctx))
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * dataset_n_ctx)
        self.dataset_ctx = nn.Parameter(ctx_vectors)
        prompts = [prompt_prefix + " " + name + "." for name in dataset_names]
        dataset_prompt = self._tokenizer(prompts).float()
        self.register_buffer("dataset_token_prefix", dataset_prompt[:, :1])
        self.register_buffer("dataset_token_suffix", dataset_prompt[:, 1 + dataset_n_ctx:])

        # categories vector
        # import pdb; pdb.set_trace()
        category_names = ["Lymphoctes", "Epithelial", "Stromal", "Plasma", "Neutrophils", "Eosinophils", "Macrophages",
                          "Tumor"]
        self.mask_map = {
            0: [3, 4, 5, 6, 7],
            1: [2, 3, 5, 7],
            2: [],
            3: [6, 7]
        }
        self.num_categories = len(category_names)
        category_n_ctx = n_ctx
        ctx_vectors = torch.empty((category_n_ctx))
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * category_n_ctx)
        self.category_ctx = nn.Parameter(ctx_vectors)
        prompts = [prompt_prefix + " " + name + "." for name in category_names]
        categories_prompt = self._tokenizer(prompts).float()
        self.register_buffer("categories_token_prefix", categories_prompt[:, :1])
        self.register_buffer("categories_token_suffix", categories_prompt[:, 1 + category_n_ctx:])
        # self.categories_prompt = categories_prompt
        self.categories_embedding = nn.Embedding(vocab_size, width)
        # dataset and categories crossattention
        self.Data_Cate_CA = MultiLabel_Prompt_Module(embed_dims=width, num_heads=8, num_layers=layers)

        self.FeaturePromptCA = MultiLabel_Prompt_Module(embed_dims=width, num_heads=8, num_layers=layers)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = DeformableDetrTransformerEncoder(**self.encoder)
        self.decoder = DinoTransformerDecoder_MultiHead(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        # NOTE In DINO, the query_embedding only contains content
        # queries, while in Deformable DETR, the query_embedding
        # contains both content and spatial queries, and in DETR,
        # it only contains spatial queries.

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super(DeformableDETR_Nuclei, self).init_weights()
        for coder in self.encoder, self.decoder:
            for p in coder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        nn.init.xavier_uniform_(self.memory_trans_fc.weight)
        nn.init.xavier_uniform_(self.query_embedding.weight)
        normal_(self.level_embed)

    def forward_transformer(
            self,
            img_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None,
    ) -> Dict:
        """Forward process of Transformer.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.
        The difference is that the ground truth in `batch_data_samples` is
        required for the `pre_decoder` to prepare the query of DINO.
        Additionally, DINO inherits the `pre_transformer` method and the
        `forward_encoder` method of DeformableDETR. More details about the
        two methods can be found in `mmdet/detector/deformable_detr.py`.

        Args:
            img_feats (tuple[Tensor]): Tuple of feature maps from neck. Each
                feature map has shape (bs, dim, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            dict: The dictionary of bbox_head function inputs, which always
            includes the `hidden_states` of the decoder output and may contain
            `references` including the initial and intermediate references.
        """
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)
        # import pdb; pdb.set_trace()
        encoder_outputs_dict['memory'] = self.fusion_layer(encoder_outputs_dict['memory'], batch_data_samples)
        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)
        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict, dataset_id=batch_data_samples[0].dataset_id)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def fusion_layer(self, memory, data_samples):
        dataset_token = []
        dataset_id = data_samples[0].dataset_id
        for data in data_samples:
            dataset_token.append((
                                     torch.cat([self.dataset_token_prefix[data.dataset_id], self.dataset_ctx,
                                                self.dataset_token_suffix[data.dataset_id]], dim=0).unsqueeze(0)
                                 ).unsqueeze(0))

        dataset_token = torch.cat(dataset_token, dim=0)
        dataset_token_mask = torch.zeros(
            (memory.shape[1], self.num_categories, dataset_token.shape[2])).to(
            dataset_token.device)  # 11736, 8, 77

        dataset_token_mask[:, self.mask_map[dataset_id], :] = True

        dataset_token_mask = torch.cat(
            (dataset_token_mask.flatten(1), torch.zeros(memory.shape[1], 1).to(dataset_token.device)), dim=1)

        dataset_token = self.dataset_mlp(dataset_token)  # 2, 1, 256
        categories_prompt = torch.cat([self.categories_token_prefix, self.category_ctx.repeat(self.num_categories, 1),
                                       self.categories_token_suffix], dim=1).unsqueeze(0).repeat(len(dataset_token), 1,
                                                                                                 1)
        category_token = self.categories_embedding(categories_prompt.int())  # 2, 8, 77, 256

        data_cate_mask = torch.zeros((1, self.num_categories, category_token.shape[2])).to(dataset_token.device)
        data_cate_mask[:, self.mask_map[dataset_id], :] = True

        category_token = rearrange(category_token, 'b c t e -> b (c t) e')  # 2, 8*77, 256
        prompt_token = self.Data_Cate_CA(query=dataset_token, key=category_token, value=category_token,
                                         cross_attn_mask=data_cate_mask.flatten(1))  # 2, 1, 256
        dataset_token = torch.cat((category_token, prompt_token), dim=1)  # 2, 8*77+1, 256

        memory = self.FeaturePromptCA(query=memory, key=dataset_token, value=dataset_token,
                                      cross_attn_mask=dataset_token_mask)
        return memory

    def pre_decoder(
            self,
            memory: Tensor,
            memory_mask: Tensor,
            spatial_shapes: Tensor,
            batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `query_pos`, and `reference_points`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points). Will only be used when
                `as_two_stage` is `True`.
            spatial_shapes (Tensor): Spatial shapes of features in all levels.
                With shape (num_levels, 2), last dimension represents (h, w).
                Will only be used when `as_two_stage` is `True`.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict]: The decoder_inputs_dict and head_inputs_dict.

            - decoder_inputs_dict (dict): The keyword dictionary args of
              `self.forward_decoder()`, which includes 'query', 'memory',
              `reference_points`, and `dn_mask`. The reference points of
              decoder input here are 4D boxes, although it has `points`
              in its name.
            - head_inputs_dict (dict): The keyword dictionary args of the
              bbox_head functions, which includes `topk_score`, `topk_coords`,
              and `dn_meta` when `self.training` is `True`, else is empty.
        """
        bs, _, c = memory.shape
        dataset_id = batch_data_samples[0].dataset_id
        cls_out_features = self.bbox_head.cls_branches[dataset_id][
            self.decoder.num_layers].out_features

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)
        enc_outputs_class = self.bbox_head.cls_branches[dataset_id][
            self.decoder.num_layers](
            output_memory)
        enc_outputs_coord_unact = self.bbox_head.reg_branches[dataset_id][
                                      self.decoder.num_layers](output_memory) + output_proposals

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]
        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 2))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training:
            dn_label_query, dn_centroid_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_centroid_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask)
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        return decoder_inputs_dict, head_inputs_dict

    def forward_decoder(self,
                        query: Tensor,
                        memory: Tensor,
                        memory_mask: Tensor,
                        reference_points: Tensor,
                        spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor,
                        dn_mask: Optional[Tensor] = None,
                        dataset_id: int = None) -> Dict:
        """Forward with Transformer decoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            query (Tensor): The queries of decoder inputs, has shape
                (bs, num_queries_total, dim), where `num_queries_total` is the
                sum of `num_denoising_queries` and `num_matching_queries` when
                `self.training` is `True`, else `num_matching_queries`.
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries_total, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            dn_mask (Tensor, optional): The attention mask to prevent
                information leakage from different denoising groups and
                matching parts, will be used as `self_attn_mask` of the
                `self.decoder`, has shape (num_queries_total,
                num_queries_total).
                It is `None` when `self.training` is `False`.

        Returns:
            dict: The dictionary of decoder outputs, which includes the
            `hidden_states` of the decoder output and `references` including
            the initial and intermediate reference_points.
        """
        inter_states, references = self.decoder(
            query=query,
            value=memory,
            key_padding_mask=memory_mask,
            self_attn_mask=dn_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches[dataset_id])

        if len(query) == self.num_queries:
            # NOTE: This is to make sure label_embeding can be involved to
            # produce loss even if there is no denoising query (no ground truth
            # target in this GPU), otherwise, this will raise runtime error in
            # distributed training.
            inter_states[0] += \
                self.dn_query_generator.label_embedding.weight[0, 0] * 0.0

        decoder_outputs_dict = dict(
            hidden_states=inter_states, references=list(references))
        return decoder_outputs_dict


class MultiLabel_Prompt_Module(BaseModule):
    def __init__(self, embed_dims, num_heads,
                 init_cfg: OptConfigType = None,
                 num_layers: int = 1):
        super().__init__(init_cfg=init_cfg)
        self.num_layers = num_layers
        self.layers = ModuleList([
            CrossAttention(embed_dims=embed_dims, num_heads=num_heads)
            for _ in range(self.num_layers)
        ])

    def forward(self,
                query: Tensor,
                key: Tensor = None,
                value: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                cross_attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                **kwargs) -> Tensor:
        """Forward function for MultiLabel_Prompt_Module.
        """
        output = query
        for layer in self.layers:
            output = layer(query=output,
                           key=key,
                           value=value,
                           query_pos=query_pos,
                           key_pos=key_pos,
                           cross_attn_mask=cross_attn_mask,
                           key_padding_mask=key_padding_mask,
                           **kwargs)
        return output


class CrossAttention(BaseModule):
    def __init__(self, embed_dims, num_heads,
                 ffn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     feedforward_channels=512,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 norm_cfg: OptConfigType = dict(type='LN'),
                 init_cfg: OptConfigType = None
                 ):
        super().__init__(init_cfg=init_cfg)
        self.cross_attn = MultiheadAttention(embed_dims=embed_dims, num_heads=num_heads, dropout=0.0, batch_first=True)
        self.embed_dims = embed_dims
        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)

    def forward(self,
                query: Tensor,
                key: Tensor = None,
                value: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                cross_attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                **kwargs) -> Tensor:
        identity = query
        query = self.norms[0](query)
        query = self.cross_attn(
            query=query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
            **kwargs)
        query = self.norms[1](query)
        query = self.ffn(query)
        query = self.norms[2](query)
        query = identity + query
        return query


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
