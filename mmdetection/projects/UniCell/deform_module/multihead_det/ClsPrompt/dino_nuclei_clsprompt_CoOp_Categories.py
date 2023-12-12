# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple, Union
from mmdet.structures import OptSampleList, SampleList
import torch
from torch import Tensor, nn
from torch.nn.init import normal_

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from mmdet.utils import OptConfigType
from mmdet.models.layers import SinePositionalEncoding, DeformableDetrTransformerEncoder
# from .deformable_detr import DeformableDETR, MultiScaleDeformableAttention
from ...deformable_detr_nuclei import DeformableDETR_Nuclei, MultiScaleDeformableAttention
from ..dino_layers_multihead_nuclei import CdnQueryGenerator_MultiHead, DinoTransformerDecoder_MultiHead
from .text_transformer import TextTransformer, MLP
from .tokenizer import SimpleTokenizer, Tokenize
from einops import rearrange
from .QueryFusionLayer import QueryFusionModule
from mmdet.models.layers import coordinate_to_encoding

# import pdb; pdb.set_trace()
@MODELS.register_module()
class DINO_Nuclei_MultiHead_ClsPrompt_CoOp_Categories(DeformableDETR_Nuclei):
    r"""Implementation of `DINO: DETR with Improved DeNoising Anchor Boxes
    for End-to-End Object Detection <https://arxiv.org/abs/2203.03605>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/DINO>`_.

    Args:
        dn_cfg (:obj:`ConfigDict` or dict, optional): Config of denoising
            query generator. Defaults to `None`.
    """

    def __init__(self, *args, dn_cfg: OptConfigType = None, text_cfg:OptConfigType=None, **kwargs) -> None:
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
        self.init_text_transformer(**text_cfg)


    def init_text_transformer(self, max_seg_len, context_length, width, layers, vocab_size,
                              proj_num_layers, n_ctx, n_query_prompt, **kwargs):
        self.dataset_tokenizer = Tokenize(SimpleTokenizer(), max_seq_len=max_seg_len)
        self.dataset_prompt = nn.Parameter(torch.empty(n_query_prompt))
        # self.dataset_embedding = nn.Embedding(vocab_size, width)
        # self.dataset_positional_encoding = nn.Parameter(torch.empty(max_seg_len+n_query_prompt, width))
        # nn.init.normal_(self.dataset_positional_encoding, std=0.02)
        # nn.init.normal_(self.dataset_embedding.weight, std=0.02)
        nn.init.normal_(self.dataset_prompt, std=0.02)
        self.dataset_mlp = MLP(context_length+n_query_prompt, width, width, 2)

        self.cls_tokenizer = Tokenize(SimpleTokenizer(), max_seq_len=max_seg_len)
        self.text_encoder = TextTransformer(context_length=context_length, width=width, layers=layers, vocab_size=vocab_size,)
        self.text_projector = MLP(width, width, width, proj_num_layers)
        self.prompt_ctx = nn.Embedding(n_ctx, width)
        self.n_ctx = n_ctx

        # fusion
        self.query_fusion_layer = QueryFusionModule()


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

    def get_info_text(self, data, info_str):
        info_text = ['a pathological image from {} dataset'.format(data.dataset_name)] * (self.num_queries - self.n_ctx)
        info = data.gt_instances[info_str][:(self.num_queries - self.n_ctx)]
        info_text[:len(info)] = info
        return info_text

    def encode_text(self, text):
        assert text.ndim in [2, 3], 'text should be 2 or 3'
        squeeze_dim = False
        num_text = 1
        if text.ndim == 3:
            num_text = text.shape[1]
            text = rearrange(text, 'b n l -> (b n) l', n=num_text)
            squeeze_dim = True

        x = self.text_encoder(text)
        text_x = self.text_projector(x)

        if squeeze_dim:
            text_x = rearrange(text_x, '(b n) c -> b n c', n=num_text)
            text_ctx = self.prompt_ctx.weight.unsqueeze(0).repeat(text_x.shape[0], 1, 1)
            text_x = torch.cat([text_x, text_ctx], dim=1)

        return text_x

    def tokenize_text(self, data_samples):
        cls_text = []
        dataset = []

        for data in data_samples:
            cls_text.append(self.get_info_text(data, 'cls_prompt_text'))
            dataset.append(data.dataset_name)
        dataset_token = torch.cat([
            self.dataset_tokenizer(data).to(data_samples[0].gt_instances.centroids.device).unsqueeze(0) for data in dataset], dim=0)
        # import pdb; pdb.set_trace()

        dataset_token = torch.cat([self.dataset_prompt.unsqueeze(0).repeat(dataset_token.shape[0], 1), dataset_token], dim=1)
        dataset_token = self.dataset_mlp(dataset_token)

        cls_token = torch.cat(
                [self.cls_tokenizer(cls).to(data_samples[0].gt_instances.centroids.device).unsqueeze(0) for cls in
                 cls_text], dim=0)
        cls_text_x = self.encode_text(cls_token)

        # cls_text_x: (2, 900, 256) dataset_token: (2, 256)
        return cls_text_x, dataset_token


    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:

        img_feats = self.extract_feat(batch_inputs)
        cls_text_x, dataset_token = self.tokenize_text(batch_data_samples)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples,
                                                    dataset_token)
        output = {**head_inputs_dict, "cls_text_x": cls_text_x}
        losses = self.bbox_head.loss(
            **output, batch_data_samples=batch_data_samples)

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:

        img_feats = self.extract_feat(batch_inputs)
        _, dataset_token = self.tokenize_text(batch_data_samples)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples,
                                                    dataset_token)
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def forward_transformer(
            self,
            img_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None,
            dataset_token: Tensor = None,
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

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        tmp_dec_in['query'], matching_query_token = self.query_dataset_fusion(tmp_dec_in['query'], tmp_dec_in['reference_points'],
                                                                 decoder_inputs_dict['valid_ratios'], head_inputs_dict.get('dn_meta', None), dataset_token)

        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict, dataset_id=batch_data_samples[0].dataset_id)
        head_inputs_dict.update(decoder_outputs_dict)
        if self.training:
            head_inputs_dict.update({'query_token': matching_query_token})
        return head_inputs_dict

    def query_dataset_fusion(self, query, reference_points, valid_ratios, dn_meta, dataset_token):
        if dn_meta is not None:
            num_denoising_queries = dn_meta['num_denoising_queries']
            dn_query = query[:, :num_denoising_queries, :]
            matching_query = query[:, num_denoising_queries:, :]
        else:
            dn_query = None
            matching_query = query
        reference_points_input = \
            reference_points[:, :, None] * valid_ratios[:, None]
        query_sine_embed = coordinate_to_encoding(
            reference_points_input[:, :, 0, :])
        query_pos = self.decoder.ref_point_head(query_sine_embed)
        matching_query = self.query_fusion_layer(query=matching_query, key=dataset_token.unsqueeze(1), value=dataset_token.unsqueeze(1), key_pos=0, query_pos=query_pos[:, num_denoising_queries:, :] if dn_meta else query_pos)
        if dn_meta:
            return torch.cat([dn_query, matching_query], dim=1), matching_query
        else:
            return matching_query, matching_query

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
        cls_out_features = self.bbox_head.num_classes[dataset_id]

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
                        dataset_id: int=None) -> Dict:
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
