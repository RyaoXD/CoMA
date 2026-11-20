"""KAD CLIP+TQN modules, vendored from the official repo without petrel/ViT deps."""

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from einops import rearrange
from torch import nn
from transformers import AutoModel, BertConfig

from coma.models.downstream.external.kad_transformer import TransformerDecoder, TransformerDecoderLayer


def _resnet50():
    try:
        return models.resnet50(weights=None)
    except TypeError:
        return models.resnet50(pretrained=False)


class CLP_clinical(nn.Module):
    def __init__(
        self,
        bert_model_name: str,
        embed_dim: int = 768,
        freeze_layers: Union[Tuple[int, int], int] = None,
    ):
        super().__init__()
        self.bert_model = self._get_bert_basemodel(
            bert_model_name=bert_model_name, freeze_layers=freeze_layers
        )
        self.mlp_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.embed_dim = embed_dim
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.init_parameters()

    def init_parameters(self):
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))
        for m in self.mlp_embed:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=self.embed_dim ** -0.5)

    def _get_bert_basemodel(self, bert_model_name, freeze_layers=None):
        config = BertConfig.from_pretrained(bert_model_name, output_hidden_states=True)
        model = AutoModel.from_pretrained(bert_model_name, config=config)
        if freeze_layers is not None:
            for layer_idx in freeze_layers:
                for param in list(model.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False
        return model

    def encode_text(self, text):
        output = self.bert_model(
            input_ids=text["input_ids"], attention_mask=text["attention_mask"]
        )
        pooler_output = output[1]
        return self.mlp_embed(pooler_output)

    def forward(self, text1, text2):
        text1_features = F.normalize(self.encode_text(text1), dim=-1)
        text2_features = F.normalize(self.encode_text(text2), dim=-1)
        return text1_features, text2_features, self.logit_scale.exp()


class ModelRes512(nn.Module):
    def __init__(self, res_base_model="resnet50"):
        super().__init__()
        self.resnet = _resnet50()
        num_ftrs = int(self.resnet.fc.in_features)
        self.res_features = nn.Sequential(*list(self.resnet.children())[:-2])
        self.res_l1 = nn.Linear(num_ftrs, num_ftrs)
        self.res_l2 = nn.Linear(num_ftrs, 768)

    def forward(self, img):
        batch_size = img.shape[0]
        res_fea = self.res_features(img)
        res_fea = rearrange(res_fea, "b d n1 n2 -> b (n1 n2) d")
        h = rearrange(res_fea, "b n d -> (b n) d")
        x = F.relu(self.res_l1(h))
        x = self.res_l2(x)
        out_emb = rearrange(x, "(b n) d -> b n d", b=batch_size)
        out_pool = torch.mean(out_emb, dim=1)
        return out_emb, out_pool


class ModelRes(nn.Module):
    def __init__(self, res_base_model="resnet50"):
        super().__init__()
        self.resnet = _resnet50()
        num_ftrs = int(self.resnet.fc.in_features / 2)
        self.res_features = nn.Sequential(*list(self.resnet.children())[:-3])
        self.res_l1 = nn.Linear(num_ftrs, num_ftrs)
        self.res_l2 = nn.Linear(num_ftrs, 768)

    def forward(self, img):
        batch_size = img.shape[0]
        res_fea = self.res_features(img)
        res_fea = rearrange(res_fea, "b d n1 n2 -> b (n1 n2) d")
        h = rearrange(res_fea, "b n d -> (b n) d")
        x = F.relu(self.res_l1(h))
        x = self.res_l2(x)
        out_emb = rearrange(x, "(b n) d -> b n d", b=batch_size)
        out_pool = torch.mean(out_emb, dim=1)
        return out_emb, out_pool


class TQN_Model(nn.Module):
    def __init__(self, embed_dim: int = 768, class_num: int = 2):
        super().__init__()
        self.d_model = embed_dim
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        decoder_layer = TransformerDecoderLayer(
            self.d_model, 4, 1024, 0.1, "relu", normalize_before=True
        )
        self.decoder_norm = nn.LayerNorm(self.d_model)
        self.decoder = TransformerDecoder(
            decoder_layer, 4, self.decoder_norm, return_intermediate=False
        )
        self.dropout_feas = nn.Dropout(0.1)
        self.mlp_head = nn.Sequential(nn.Linear(embed_dim, class_num))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, image_features, text_features, return_atten: bool = False):
        batch_size = image_features.shape[0]
        image_features = image_features.transpose(0, 1)
        text_features = text_features.unsqueeze(1).repeat(1, batch_size, 1)
        image_features = self.decoder_norm(image_features)
        text_features = self.decoder_norm(text_features)
        features, atten_map = self.decoder(
            text_features,
            image_features,
            memory_key_padding_mask=None,
            pos=None,
            query_pos=None,
        )
        features = self.dropout_feas(features).transpose(0, 1)
        out = self.mlp_head(features)
        if return_atten:
            return out, atten_map
        return out
