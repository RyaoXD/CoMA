from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from sklearn.cluster import KMeans
from einops import rearrange
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.plugins import DDP2Plugin, DDPPlugin
from coma.models.backbones.encoder import BertEncoder, ImageEncoder
from torch import distributed as dist

class CoMA(LightningModule):
    '''Pytorch lightning implementation of CoMA'''

    def __init__(self,
                 img_encoder: str = "vit_base",
                 freeze_bert: bool = False,
                 emb_dim: int = 128,
                 learning_rate: float = 2e-5,
                 momentum: float = 0.9,
                 weight_decay: float = 0.05,
                 batch_size: int = 64,
                 num_workers: int = 8,
                 # TODO: tune this hyperparameter
                 gia_temperature: float = 0.1,
                 tca_temperature: float = 0.1,
                 ipa_temperature: float = 0.2,
                 cpa_temperature: float = 0.2,
                 group_temperature: float = 0.2,
                 num_instance_prototypes: int = 500,
                 num_concept_prototypes: int = 500,
                 bidirectional: bool = True,
                 use_local_atten: bool = False,
                 prompt_vector: bool = False,
                 use_dpl: bool = False,
                 dpl_hidden_dim: int = None,
                 language_group: bool = False,
                 num_heads: int = 1,
                 lamb: float = 0.75,
                 lambda_gia: float = 1,
                 lambda_tca: float = 1,
                 lambda_ipa: float = 1,
                 lambda_cpa: float = 1,
                 lambda_group: float = 1,
                 freeze_prototypes_epochs: int = 5,
                 sinkhorn_iterations: int = 3,
                 epsilon: float = 0.05,
                 *args,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()

        # init encoders
        self.img_encoder_q = ImageEncoder(
            model_name=img_encoder, output_dim=self.hparams.emb_dim)
        self.text_encoder_q = BertEncoder(
            output_dim=self.hparams.emb_dim, freeze_bert=freeze_bert)
        
        # define concept clause decompose (CCD)
        # TODO default max length of language
        self.hparams.max_concepts = 30 \
            if self.hparams.max_concepts is None else self.hparams.max_concepts
        if self.hparams.concept_decompose:
            self.clauses_attn_pool = AttentionPool2d(
                self.hparams.max_concepts, self.text_encoder_q.embedding_dim, 64)
        else:
            self.clauses_attn_pool = None

        # global query prompts Q_g (static --prompt_vector or DPL base)
        self.global_query_prompt = None
        self.prompt_meta_net = None
        try:
            use_dpl = getattr(self.hparams, "use_dpl", False)
            use_static_prompt = (
                self.hparams.prompt_vector is not None and self.hparams.prompt_vector
            )
            if use_dpl or use_static_prompt:
                self.hparams.vector_length = 30 \
                    if self.hparams.vector_length is None else self.hparams.vector_length
                feat_dim = self.img_encoder_q.feature_dim
                num_queries = self.hparams.vector_length
                self.global_query_prompt = nn.Parameter(
                    torch.randn(num_queries, feat_dim) * 0.02, requires_grad=True)
                if use_dpl:
                    hidden_dim = dpl_hidden_dim or feat_dim
                    self.prompt_meta_net = nn.Sequential(
                        nn.Linear(feat_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, num_queries * feat_dim),
                    )
        except Exception:
            self.global_query_prompt = None
            self.prompt_meta_net = None
        self._migrate_legacy_prompt_vector()

        if self.hparams.use_local_atten:
            # patch local attention layer
            self.patch_local_atten_layer = nn.MultiheadAttention(
                self.hparams.emb_dim, self.hparams.num_heads, batch_first=True)
            # token local attention layer
            self.token_local_atten_layer = nn.MultiheadAttention(
                self.hparams.emb_dim, self.hparams.num_heads, batch_first=True)
        else:
            self.patch_local_atten_layer = None
            self.token_local_atten_layer = None

        self.hparams.freeze_prototypes_epochs = 5 if self.hparams.freeze_prototypes_epochs is None \
            else self.hparams.freeze_prototypes_epochs
        
        # define concept prototype
        self.concept_prototype_init_pool = None
        try:
            if self.hparams.concept_prototype is None:
                self.concept_prototype_layer = None
            else:
                self.concept_prototype_layer = nn.Linear(emb_dim, num_concept_prototypes, bias=False)
        except:
            self.concept_prototype_layer = None
        
        # define instance prototype
        self.instance_prototype_init_pool = None
        try:
            if self.hparams.instance_prototype == "mapping":
                assert not self.concept_prototype_layer is None, "No concept_prototype_layer"
                self.instance_prototype_layer = nn.Sequential(
                    nn.Linear(num_instance_prototypes, emb_dim, bias=False),
                    nn.Linear(emb_dim, num_instance_prototypes, bias=False))
            else:
                self.instance_prototype_layer = nn.Linear(emb_dim, num_instance_prototypes, bias=False)
        except:
            self.instance_prototype_layer = nn.Linear(emb_dim, num_instance_prototypes, bias=False)

        if self._use_ddp_or_dpp2(self.trainer):
            self.get_assignments = self.distributed_sinkhorn
        else:
            self.get_assignments = self.sinkhorn


    # merge decomposed concept clause for alignment
    def decomposed_clause_merge(self, clauses_feat_q, clauses, clauses_ids):
        n_clauses, n_dim = clauses_feat_q.shape
        now_index = -1
        pos = 0
        batch_tokens_feat_q = torch.tensor([])
        tokens_feat_q = None
        reports_clauses = []
        report_clauses = []
        for clau_feat_q, clau, clau_id in zip(clauses_feat_q, clauses, clauses_ids):
            # clau_id = clau_ids[0]
            if now_index != clau_id:
                now_index = clau_id
                if not tokens_feat_q is None:
                    batch_tokens_feat_q=torch.cat((batch_tokens_feat_q, tokens_feat_q.unsqueeze(0)),dim=0)
                    report_clauses += ["[PAD]" for _ in range(self.hparams.max_concepts - len(report_clauses))]
                    reports_clauses.append(report_clauses)
                tokens_feat_q = torch.zeros((self.hparams.max_concepts,n_dim)).to(clauses_feat_q.dtype)
                report_clauses = []
                pos = 0
            if pos >= self.hparams.max_concepts:
                continue
            tokens_feat_q[pos] = clau_feat_q
            clau = ''.join([t+' ' for t in clau if (t not in ['[CLS]', '[PAD]', '[SEP]'])])
            report_clauses.append(clau)
            pos += 1
        batch_tokens_feat_q=torch.cat((batch_tokens_feat_q,tokens_feat_q.unsqueeze(0)),dim=0).to(clauses_feat_q.device)
        report_clauses += ["[PAD]" for _ in range(self.hparams.max_concepts - len(report_clauses))]
        reports_clauses.append(report_clauses)
        
        # rebuild report/token feature from word-wise to clause-wise
        report_feat_q, pool_attn = self.clauses_attn_pool(batch_tokens_feat_q)
        return report_feat_q, batch_tokens_feat_q, pool_attn, reports_clauses

    @property
    def prompt_vector(self):
        """Backward-compatible alias for global_query_prompt."""
        return self.global_query_prompt

    @prompt_vector.setter
    def prompt_vector(self, value):
        if value is None:
            self.global_query_prompt = None
        elif isinstance(value, nn.Parameter):
            self.register_parameter("global_query_prompt", value)
        else:
            self.register_parameter("global_query_prompt", nn.Parameter(value))

    def _migrate_legacy_prompt_vector(self):
        """Move legacy prompt_vector weights into global_query_prompt if needed."""
        legacy = self._parameters.pop("prompt_vector", None)
        if legacy is None:
            legacy = self.__dict__.pop("prompt_vector", None)
        if legacy is None:
            return
        if self.global_query_prompt is None:
            if isinstance(legacy, nn.Parameter):
                self.register_parameter("global_query_prompt", legacy)
            else:
                self.register_parameter(
                    "global_query_prompt", nn.Parameter(legacy.clone()))

    def _has_query_prompts(self):
        self._migrate_legacy_prompt_vector()
        return self.global_query_prompt is not None

    def _num_query_prompts(self):
        self._migrate_legacy_prompt_vector()
        if self.global_query_prompt is not None:
            return self.global_query_prompt.size(0)
        return 0

    def _get_patch_meta_input(self, imgs):
        """Global patch context v_i: mean over patches (B, D) for DPL meta-network."""
        patch_tokens = self.img_encoder_q.model.patch_embed(imgs)
        return patch_tokens.mean(dim=1)

    def _build_dynamic_query_prompts(self, imgs):
        """Q_i = Q_g + f_q(v_i) as in Dynamic Prompt Learning."""
        v_i = self._get_patch_meta_input(imgs)
        batch_size = v_i.size(0)
        num_queries = self.hparams.vector_length
        feat_dim = self.img_encoder_q.feature_dim
        delta_q = self.prompt_meta_net(v_i).view(batch_size, num_queries, feat_dim)
        global_q = self.global_query_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        return global_q + delta_q

    def _encode_images(self, imgs):
        self._migrate_legacy_prompt_vector()
        if self.global_query_prompt is None:
            return self.img_encoder_q(imgs, None)
        if self.prompt_meta_net is not None:
            return self.img_encoder_q(imgs, self._build_dynamic_query_prompts(imgs))
        return self.img_encoder_q(imgs, self.global_query_prompt)

    def load_state_dict(self, state_dict, strict=True):
        renamed = {}
        for k, v in state_dict.items():
            if "prompt_vector" in k and "global_query_prompt" not in k:
                new_k = k.replace("prompt_vector", "global_query_prompt")
                if new_k not in renamed:
                    renamed[new_k] = v
                continue
            renamed[k] = v
        result = super().load_state_dict(renamed, strict=False)
        self._migrate_legacy_prompt_vector()
        if strict:
            unexpected = [
                k for k in result.unexpected_keys if "prompt_vector" not in k]
            if result.missing_keys or unexpected:
                raise RuntimeError(
                    f"Missing keys: {result.missing_keys}\n"
                    f"Unexpected keys: {unexpected}")
        return result

    def group_embedings(self, l_embed, v_embed, similarity_threshold=0.1):
        with torch.no_grad():
            sim_metrix = torch.bmm(l_embed, v_embed.permute(0, 2, 1))
            min_vals = torch.min(sim_metrix, dim=-1, keepdim=True)[0]
            max_vals = torch.max(sim_metrix, dim=-1, keepdim=True)[0]
            range_vals = torch.clamp(max_vals - min_vals, min=1e-8)
            sim_metrix = (sim_metrix - min_vals) / range_vals
            sim_metrix = torch.where(sim_metrix < similarity_threshold, 0.0, sim_metrix)
        
        sum_sim = torch.clamp(torch.sum(sim_metrix, dim=-1, keepdim=True), min=1e-8)
        v_align_weights = sim_metrix / sum_sim

        grouped_v_embed = torch.einsum('btp,bpd->btd', v_align_weights, v_embed)
        grouped_v_embed = F.normalize(grouped_v_embed, p=2, dim=-1)
        return grouped_v_embed, v_embed

    def forward(self, batch, batch_idx, split="train"):
        '''Forward step of our method'''

        # Forward of query text encoder
        report_feat_q, token_feat_q, token_attn_q, sents = self.text_encoder_q(
            batch["caption_ids"], batch["attention_mask"], batch["token_type_ids"])
        if self.hparams.concept_decompose:
            report_feat_q, token_feat_q, token_attn_q, sents = self.decomposed_clause_merge(report_feat_q, sents, batch["reportindex"])
        token_emb_q = self.text_encoder_q.local_embed(token_feat_q)
        report_emb_q = self.text_encoder_q.global_embed(report_feat_q)
        
        # Forward of query image encoder (static prompt or DPL dynamic prompts)
        img_feat_q, patch_feat_q, query_feat_q = self._encode_images(batch["imgs"])
        patch_emb_q = self.img_encoder_q.local_embed(patch_feat_q)
        query_emb_q = self.img_encoder_q.local_embed(query_feat_q)
        img_emb_q = self.img_encoder_q.global_embed(img_feat_q)
        
        if self.hparams.language_group:
            patch_emb_q, o_patch_emb_q = self.group_embedings(token_emb_q, patch_emb_q)
            o_patch_emb_q = F.normalize(o_patch_emb_q, dim=-1)
            if self._has_query_prompts():
                query_emb_q, o_query_emb_q = self.group_embedings(token_emb_q, query_emb_q)
                o_query_emb_q = F.normalize(o_query_emb_q, dim=-1)
        
        token_emb_q = F.normalize(token_emb_q, dim=-1)
        report_emb_q = F.normalize(report_emb_q, dim=-1)
        patch_emb_q = F.normalize(patch_emb_q, dim=-1)
        query_emb_q = F.normalize(query_emb_q, dim=-1)
        img_emb_q = F.normalize(img_emb_q, dim=-1)

        if self.hparams.language_group:
            bz = patch_emb_q.size(0)
            labels = torch.arange(bz).type_as(patch_emb_q).long()

            group_sim = torch.bmm(token_emb_q, patch_emb_q.permute(0, 2, 1)) / \
                self.hparams.group_temperature
                
            group_num = group_sim.size(1)
            group_sim_1 = rearrange(group_sim, "b n1 n2 -> (b n1) n2")
            targets = torch.arange(group_num).type_as(
                token_emb_q).long().repeat(bz)
            loss_group_1 = torch.sum(F.cross_entropy(
                group_sim_1, targets, reduction="none") * token_atten_weights.view(-1)) / bz

            group_sim_2 = rearrange(token_sim, "b n1 n2 -> (b n2) n1")
            loss_group_2 = torch.sum(F.cross_entropy(
                group_sim_2, targets, reduction="none") * token_atten_weights.view(-1)) / bz

            loss_group = loss_group_1 + loss_group_2
        
            if self._has_query_prompts():
                bz = query_emb_q.size(0)
                labels = torch.arange(bz).type_as(query_emb_q).long()

                group_sim = torch.bmm(token_emb_q, query_emb_q.permute(0, 2, 1)) / \
                    self.hparams.group_temperature
                    
                group_num = group_sim.size(1)
                group_sim_1 = rearrange(group_sim, "b n1 n2 -> (b n1) n2")
                targets = torch.arange(group_num).type_as(
                    token_emb_q).long().repeat(bz)
                loss_group_1 = torch.sum(F.cross_entropy(
                    group_sim_1, targets, reduction="none") * token_atten_weights.view(-1)) / bz

                group_sim_2 = rearrange(token_sim, "b n1 n2 -> (b n2) n1")
                loss_group_2 = torch.sum(F.cross_entropy(
                    group_sim_2, targets, reduction="none") * token_atten_weights.view(-1)) / bz

                loss_group += loss_group_1 + loss_group_2
        else:
            loss_group = 0
            
        # build prototype clustering pool
        if self.current_epoch == self.hparams.freeze_prototypes_epochs - 1:
            if self.hparams.instance_prototype == "cluster":
                self.instance_prototype_init_pool = report_emb_q if self.instance_prototype_init_pool is None \
                    else torch.vstack((self.instance_prototype_init_pool,report_emb_q))
            if self.hparams.concept_prototype == "cluster":
                self.concept_prototype_init_pool = rearrange(token_emb_q, "b nt d -> (b nt) d") if self.concept_prototype_init_pool is None \
                    else torch.vstack((self.concept_prototype_init_pool,rearrange(token_emb_q, "b nt d -> (b nt) d")))
        
        if self.current_epoch == self.hparams.freeze_prototypes_epochs:
            if not self.instance_prototype_init_pool is None:
                if len(self.instance_prototype_init_pool) < self.hparams.num_center:
                    self.instance_prototype_init_pool = torch.vstack(self.instance_prototype_init_pool)
                else:
                    device = self.instance_prototype_init_pool.device
                    self.instance_prototype_init_pool = self.instance_prototype_init_pool.to('cpu')
                    kmean_model = KMeans(n_clusters=self.hparams.num_center, n_init='auto')
                    kmean_model.fit(self.instance_prototype_init_pool)
                    dic = {i:{"ind":[], "vector":None, "center":None} for i in range(self.hparams.num_center)}
                    for ind,n_label in enumerate(kmean_model.labels_):
                        dic[n_label]["ind"].append(ind)
                        dic[n_label]["vector"] = self.instance_prototype_init_pool[ind].reshape(1,-1) if dic[n_label]["vector"] is None else\
                            torch.vstack((dic[n_label]["vector"],self.instance_prototype_init_pool[ind].reshape(1,-1)))
                    for value in dic.values():
                        value["center"] = torch.mean(value["vector"],axis=0)
                    self.instance_prototype_init_pool = torch.vstack([value["center"] for value in dic.values()]).to(device)
                    
            if not self.concept_prototype_init_pool is None:
                if len(self.concept_prototype_init_pool) < self.hparams.num_center:
                    self.concept_prototype_init_pool = torch.vstack(self.concept_prototype_init_pool)
                else:
                    device = self.concept_prototype_init_pool.device
                    self.concept_prototype_init_pool = self.concept_prototype_init_pool.to('cpu')
                    kmean_model = KMeans(n_clusters=self.hparams.num_center,n_init='auto')
                    kmean_model.fit(self.concept_prototype_init_pool)
                    
                    dic = {i:{"ind":[],"vector":None,"center":None} for i in range(self.hparams.num_center)}
                    for ind,n_label in enumerate(kmean_model.labels_):
                        dic[n_label]["ind"].append(ind)
                        dic[n_label]["vector"] = self.concept_prototype_init_pool[ind].reshape(1,-1) if dic[n_label]["vector"] is None else\
                            torch.vstack((dic[n_label]["vector"],self.concept_prototype_init_pool[ind].reshape(1,-1)))
                    for value in dic.values():
                        value["center"] = torch.mean(value["vector"],axis=0)
                    self.concept_prototype_init_pool = torch.vstack([value["center"] for value in dic.values()]).to(device)

        # Global Instance-level Alignment (GIA)
        bz = img_emb_q.size(0)
        labels = torch.arange(bz).type_as(report_emb_q).long()

        scores = img_emb_q.mm(report_emb_q.t())
        scores /= self.hparams.gia_temperature
        scores1 = scores.transpose(0, 1)
        loss0 = F.cross_entropy(scores, labels)
        loss1 = F.cross_entropy(scores1, labels)
        loss_gia = loss0 + loss1

        # compute retrieval accuracy
        i2t_acc1, i2t_acc5 = self.precision_at_k(
            scores, labels, top_k=(1, 5))
        t2i_acc1, t2i_acc5 = self.precision_at_k(
            scores1, labels, top_k=(1, 5))
        acc1 = (i2t_acc1 + t2i_acc1) / 2.
        acc5 = (i2t_acc5 + t2i_acc5) / 2.
        
        # Concept-level Token-wise Alignment (TCA)
        # cross attention patch to tokens
        if self.hparams.concept_decompose:
            mask = torch.from_numpy(np.array(sents) == "[PAD]").type_as(
                batch["imgs"]).bool()
        else:
            mask = torch.from_numpy(np.array(sents)[:, 1:] == "[PAD]").type_as(
                batch["imgs"]).bool()

        # sim_metrix = torch.bmm(token_emb_q, patch_emb_q.permute(0, 2, 1))
        # sim_metrix = (sim_metrix- torch.min(sim_metrix, keepdim=True, axis=-1)[0]) / \
        #     (torch.max(sim_metrix, keepdim=True, axis=-1)[0] - torch.min(sim_metrix, keepdim=True, axis=-1)[0])
            
        if self.hparams.use_local_atten:
            token_atten_output, _ = self.token_local_atten_layer(
                token_emb_q, patch_emb_q, patch_emb_q)
        else:
            atten_sim = torch.bmm(token_emb_q, patch_emb_q.permute(0, 2, 1))
            token_num = token_emb_q.size(1)
            atten_scores = F.softmax(
                atten_sim / self.hparams.tca_temperature, dim=-1)
            token_atten_output = torch.bmm(atten_scores, patch_emb_q)

        token_atten_output = F.normalize(token_atten_output, dim=-1)

        token_sim = torch.bmm(
            token_emb_q, token_atten_output.permute(0, 2, 1)) / self.hparams.tca_temperature

        with torch.no_grad():
            atten_weights = token_attn_q.detach()
            token_atten_weights = []
            for i in range(bz):
                atten_weight = atten_weights[i]
                nonzero = atten_weight.nonzero().squeeze()
                low = torch.quantile(atten_weight[nonzero], 0.1)
                high = torch.quantile(atten_weight[nonzero], 0.9)
                atten_weight[nonzero] = atten_weight[nonzero].clip(low, high)
                token_atten_weights.append(atten_weight.clone())
            token_atten_weights = torch.stack(token_atten_weights)

        token_atten_weights /= token_atten_weights.sum(dim=1, keepdims=True)

        token_sim = torch.bmm(token_emb_q, token_atten_output.permute(
            0, 2, 1)) / self.hparams.tca_temperature
        token_num = token_sim.size(1)
        token_sim_1 = rearrange(token_sim, "b n1 n2 -> (b n1) n2")
        targets = torch.arange(token_num).type_as(
            token_emb_q).long().repeat(bz)
        loss_token_1 = torch.sum(F.cross_entropy(
            token_sim_1, targets, reduction="none") * token_atten_weights.view(-1)) / bz

        token_sim_2 = rearrange(token_sim, "b n1 n2 -> (b n2) n1")
        loss_token_2 = torch.sum(F.cross_entropy(
            token_sim_2, targets, reduction="none") * token_atten_weights.view(-1)) / bz

        loss_token = (loss_token_1 + loss_token_2) / 2.
        
        # cross attention token to patchs
        if self.hparams.bidirectional:
            if self.hparams.use_local_atten:
                patch_atten_output, _ = self.patch_local_atten_layer(
                    patch_emb_q, token_emb_q, token_emb_q, key_padding_mask=mask)
            else:
                atten_sim = torch.bmm(patch_emb_q, token_emb_q.permute(0, 2, 1))
                patch_num = patch_emb_q.size(1)
                atten_sim[mask.unsqueeze(1).repeat(1, patch_num, 1)] = float("-inf")
                atten_scores = F.softmax(
                    atten_sim / self.hparams.tca_temperature, dim=-1)
                patch_atten_output = torch.bmm(atten_scores, token_emb_q)

            if "vit" not in self.hparams.img_encoder:
                patch_atten_output = F.normalize(patch_atten_output, dim=-1)
                patch_num = patch_atten_output.size(1)
                patch_atten_weights = torch.ones(
                    bz, patch_num).type_as(batch["imgs"]) / patch_num
                query_atten_weights = patch_atten_weights.clone()
            else:
                with torch.no_grad():
                    img_attn_map = self.img_encoder_q.model.blocks[-1].attn.attention_map.detach()
                    if not self._has_query_prompts():
                        atten_weights = img_attn_map[:, :, 0, 1:].mean(dim=1)
                        patch_atten_weights = []
                        for i in range(bz):
                            atten_weight = atten_weights[i]
                            atten_weight = atten_weight.clip(torch.quantile(
                                atten_weight, 0.1), torch.quantile(atten_weight, 0.9))
                            patch_atten_weights.append(atten_weight.clone())
                        patch_atten_weights = torch.stack(patch_atten_weights)
                        query_atten_weights = patch_atten_weights.clone()
                    else:
                        atten_weights = img_attn_map[:, :, 0, 1:-self.hparams.vector_length].mean(dim=1)
                        patch_atten_weights = []
                        for i in range(bz):
                            atten_weight = atten_weights[i]
                            atten_weight = atten_weight.clip(torch.quantile(
                                atten_weight, 0.1), torch.quantile(atten_weight, 0.9))
                            patch_atten_weights.append(atten_weight.clone())
                        patch_atten_weights = torch.stack(patch_atten_weights)
                        
                        atten_weights = img_attn_map[:, :, 0, -self.hparams.vector_length:].mean(dim=1)
                        query_atten_weights = []
                        for i in range(bz):
                            atten_weight = atten_weights[i]
                            atten_weight = atten_weight.clip(torch.quantile(
                                atten_weight, 0.1), torch.quantile(atten_weight, 0.9))
                            query_atten_weights.append(atten_weight.clone())
                        query_atten_weights = torch.stack(query_atten_weights)

                patch_atten_weights /= patch_atten_weights.sum(
                    dim=1, keepdims=True)
                query_atten_weights /= query_atten_weights.sum(
                    dim=1, keepdims=True)

            patch_sim = torch.bmm(patch_emb_q, patch_atten_output.permute(
                0, 2, 1)) / self.hparams.tca_temperature
            patch_num = patch_sim.size(1)
            patch_sim_1 = rearrange(patch_sim, "b n1 n2 -> (b n1) n2")
            targets = torch.arange(patch_num).type_as(
                patch_emb_q).long().repeat(bz)
            loss_patch_1 = torch.sum(F.cross_entropy(
                patch_sim_1, targets, reduction="none") * patch_atten_weights.view(-1)) / bz

            patch_sim_2 = rearrange(patch_sim, "b n1 n2 -> (b n2) n1")
            loss_patch_2 = torch.sum(F.cross_entropy(
                patch_sim_2, targets, reduction="none") * patch_atten_weights.view(-1)) / bz

            loss_patch = (loss_patch_1 + loss_patch_2) / 2.

            loss_tca = loss_patch + loss_token
        else:
            loss_tca = loss_token
        
        # Instance-level Prototype Alignment (IPA)
        # normalize prototype layer
        with torch.no_grad():
            if self.hparams.instance_prototype != "mapping":
                w = self.instance_prototype_layer.weight.data.clone()
                if self.current_epoch == self.hparams.freeze_prototypes_epochs and \
                    not self.instance_prototype_init_pool is None:
                    w[:len(self.instance_prototype_init_pool)] = self.instance_prototype_init_pool
                w = F.normalize(w, dim=1, p=2)
                self.instance_prototype_layer.weight.copy_(w)
            if not self.concept_prototype_layer is None:
                w = self.concept_prototype_layer.weight.data.clone()
                if self.current_epoch == self.hparams.freeze_prototypes_epochs and \
                    not self.concept_prototype_init_pool is None:
                    w[:len(self.concept_prototype_init_pool)]=self.concept_prototype_init_pool
                w = F.normalize(w, dim=1, p=2)
                self.concept_prototype_layer.weight.copy_(w)

        # Compute assign code of images
        if self.hparams.instance_prototype == "mapping":
            img_proto_out = self.concept_prototype_layer(img_emb_q)
            img_proto_out = self.instance_prototype_layer(img_proto_out)
            report_proto_out = self.concept_prototype_layer(report_emb_q)
            report_proto_out = self.instance_prototype_layer(report_proto_out)
        else:
            img_proto_out = self.instance_prototype_layer(img_emb_q)
            report_proto_out = self.instance_prototype_layer(report_emb_q)

        with torch.no_grad():
            img_code = torch.exp(
                img_proto_out / self.hparams.epsilon).t()
            img_code = self.get_assignments(
                img_code, self.hparams.sinkhorn_iterations)         # bz, 500
            report_code = torch.exp(
                report_proto_out / self.hparams.epsilon).t()
            report_code = self.get_assignments(
                report_code, self.hparams.sinkhorn_iterations)       # bz, 500

        img_proto_prob = F.softmax(
            img_proto_out / self.hparams.ipa_temperature, dim=1)
        report_proto_prob = F.softmax(
            report_proto_out / self.hparams.ipa_temperature, dim=1)

        loss_i2t_proto = -torch.mean(
                torch.sum(img_code * torch.log(report_proto_prob), dim=1))
        loss_t2i_proto = -torch.mean(
                torch.sum(report_code * torch.log(img_proto_prob), dim=1))

        loss_ipa = (loss_i2t_proto + loss_t2i_proto) / 2.

        # Concept-level Prototype Alignment (CPA)
        if not self.hparams.concept_prototype is None:
            # Compute assign code of images
            query_proto_out = self.concept_prototype_layer(query_emb_q)
            token_proto_out = self.concept_prototype_layer(token_emb_q)
            
            n_batch, n_query, _ = query_proto_out.shape
            _, n_token, _ = token_proto_out.shape
            query_proto_out = rearrange(query_proto_out, "b np d -> (b np) d")
            token_proto_out = rearrange(token_proto_out, "b ns d -> (b ns) d")

            with torch.no_grad():
                query_code = torch.exp(
                    query_proto_out / self.hparams.epsilon).t()
                query_code = self.get_assignments(
                    query_code, self.hparams.sinkhorn_iterations)         # bz, 500
                token_code = torch.exp(
                    token_proto_out / self.hparams.epsilon).t()
                token_code = self.get_assignments(
                    token_code, self.hparams.sinkhorn_iterations)       # bz, 500

            query_proto_prob = F.softmax(
                query_proto_out / self.hparams.cpa_temperature, dim=1)
            token_proto_prob = F.softmax(
                token_proto_out / self.hparams.cpa_temperature, dim=1)

            token_code = token_code.reshape((n_batch,n_token,-1))
            token_proto_prob = token_proto_prob.reshape((n_batch,n_token,-1))
            query_code = query_code.reshape((n_batch,n_query,-1))
            query_proto_prob = query_proto_prob.reshape((n_batch,n_query,-1))
            
            token_atten_weights = token_atten_weights.unsqueeze(-1).repeat(1,1,self.hparams.num_concept_prototypes)
            query_atten_weights = query_atten_weights.unsqueeze(-1).repeat(1,1,self.hparams.num_concept_prototypes)
            token_code = torch.mean(token_code*token_atten_weights,dim=1).squeeze()
            token_proto_prob = torch.mean(token_proto_prob*token_atten_weights,dim=1).squeeze()
            query_code = torch.mean(query_code*query_atten_weights,dim=1).squeeze()
            query_proto_prob = torch.mean(query_proto_prob*query_atten_weights,dim=1).squeeze()

            loss_q2t_proto = -torch.mean(
                    torch.sum(query_code * torch.log(token_proto_prob), dim=1))
            loss_t2q_proto = -torch.mean(
                    torch.sum(token_code * torch.log(query_proto_prob), dim=1))

            loss_cpa = (loss_q2t_proto + loss_t2q_proto) / 2.
            
            if self.hparams.prototype_orth:
                if not self.concept_prototype_layer is None:
                    prototype_vector = torch.squeeze(self.concept_prototype_layer.weight)
                    prototype_vector_T = torch.transpose(prototype_vector,0,1)
                    orth_operator = torch.matmul(prototype_vector,prototype_vector_T)
                    I_operator = torch.eye(prototype_vector.size(0),prototype_vector.size(0)).to(orth_operator.device)
                    difference_value = orth_operator - I_operator
                    loss_sorth = (torch.norm(difference_value,p=1) ** 0.5)/difference_value.size(0)
                    loss_orth = loss_sorth
                loss_cpa += self.hparams.lambda_orth * loss_orth
        else:
            loss_cpa = 0

        return loss_gia, loss_tca, loss_ipa, loss_cpa, loss_group, acc1, acc5

    def sinkhorn(self, Q, nmb_iters):
        ''' 
            :param Q: (num_prototypes, batch size)

        '''
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            Q /= sum_Q

            K, B = Q.shape

            if self.hparams.gpus > 0:
                u = torch.zeros(K).cuda()
                r = torch.ones(K).cuda() / K
                c = torch.ones(B).cuda() / B
            else:
                u = torch.zeros(K)
                r = torch.ones(K) / K
                c = torch.ones(B) / B

            for _ in range(nmb_iters):
                u = torch.sum(Q, dim=1)
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)

            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    def distributed_sinkhorn(self, Q, nmb_iters):
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            dist.all_reduce(sum_Q)
            Q /= sum_Q

            if self.hparams.gpus > 0:
                u = torch.zeros(Q.shape[0]).cuda(non_blocking=True)
                r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
                c = torch.ones(Q.shape[1]).cuda(
                    non_blocking=True) / (self.gpus * Q.shape[1])
            else:
                u = torch.zeros(Q.shape[0])
                r = torch.ones(Q.shape[0]) / Q.shape[0]
                c = torch.ones(Q.shape[1]) / (self.gpus * Q.shape[1])

            curr_sum = torch.sum(Q, dim=1)
            dist.all_reduce(curr_sum)

            for it in range(nmb_iters):
                u = curr_sum
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
                curr_sum = torch.sum(Q, dim=1)
                dist.all_reduce(curr_sum)
            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    def _compute_total_loss(self, loss_gia, loss_tca, loss_ipa, loss_cpa, loss_group):
        loss = self.hparams.lambda_gia * loss_gia
        if not self.hparams.without_loss_tca:
            loss = loss + self.hparams.lambda_tca * loss_tca
        if not self.hparams.without_loss_ipa:
            loss = loss + self.hparams.lambda_ipa * loss_ipa
        if not self.hparams.without_loss_cpa:
            loss = loss + self.hparams.lambda_cpa * loss_cpa
        if not self.hparams.without_loss_group:
            loss = loss + self.hparams.lambda_group * loss_group
        return loss

    def training_step(self, batch, batch_idx):
        loss_gia, loss_tca, loss_ipa, loss_cpa, loss_group, acc1, acc5 = self(
            batch, batch_idx, "train")
        loss = self._compute_total_loss(loss_gia, loss_tca, loss_ipa, loss_cpa, loss_group)

        log = {
            "train/loss": loss,
            "train/lgia": self.hparams.lambda_gia * loss_gia,
            "train/ltca": self.hparams.lambda_tca * loss_tca,
            "train/lipa": self.hparams.lambda_ipa * loss_ipa,
            "train/lcpa": self.hparams.lambda_cpa * loss_cpa,
            "train/lgroup": self.hparams.lambda_group * loss_group,
            "train/acc1": acc1,
            "train/acc5": acc5
        }
        
        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True, prog_bar=True)
        return loss

    # freeze prototype layer
    def on_after_backward(self):
        if self.current_epoch < self.hparams.freeze_prototypes_epochs:
            for param in self.instance_prototype_layer.parameters():
                param.grad = None
            if not self.hparams.concept_prototype is None:
                for param in self.concept_prototype_layer.parameters():
                    param.grad = None

    def validation_step(self, batch, batch_idx):
        loss_gia, loss_tca, loss_ipa, loss_cpa, loss_group, acc1, acc5 = self(
            batch, batch_idx, "valid")

        loss = self._compute_total_loss(loss_gia, loss_tca, loss_ipa, loss_cpa, loss_group)

        log = {
            "val/loss": loss,
            "val/lgia": self.hparams.lambda_gia * loss_gia,
            "val/ltca": self.hparams.lambda_tca * loss_tca,
            "val/lipa": self.hparams.lambda_ipa * loss_ipa,
            "val/lcpa": self.hparams.lambda_cpa * loss_cpa,
            "val/lgroup": self.hparams.lambda_group * loss_group,
            "val/acc1": acc1,
            "val/acc5": acc5
        }
        self.log_dict(log, batch_size=self.hparams.batch_size,
                      sync_dist=True, prog_bar=True)
        return loss

    @staticmethod
    def precision_at_k(output: torch.Tensor, target: torch.Tensor, top_k=(1,)):
        ''' Compute the accuracy over the k top predictions for the specified values of k'''
        with torch.no_grad():
            maxk = max(top_k)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in top_k:
                correct_k = correct[:k].contiguous(
                ).view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            self.hparams.learning_rate,
            betas=(self.hparams.momentum, 0.999),
            weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.training_steps,
            cycle_mult=1.0,
            max_lr=self.hparams.learning_rate,
            min_lr=1e-8,
            warmup_steps=int(self.training_steps * 0.4)
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # params of model & data
        parser.add_argument("--img_encoder", type=str, default="vit_base")
        parser.add_argument("--freeze_bert", action="store_true")
        parser.add_argument("--emb_dim", type=int,
                            default=128, help="128, 256")
        parser.add_argument("--num_workers", type=int, default=16)
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--mixed_precision", type=str,
                            default="bf16", help="64, 32, 16, bf16")
        parser.add_argument("--weight_decay", type=float, default=0.05)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--data_pct", type=float, default=1.)
        # concept enhance
        parser.add_argument("--rewrite", action="store_true", default=False)
        parser.add_argument("--concept_decompose", action="store_true", default=False)
        parser.add_argument("--max_concepts", type=int or None, default=50)
        parser.add_argument("--prompt_vector", action="store_true", default=False)
        parser.add_argument("--use_dpl", action="store_true", default=False,
                            help="Enable Dynamic Prompt Learning (DPL) from CoMA v1.5")
        parser.add_argument("--dpl_hidden_dim", type=int, default=None,
                            help="Hidden dim of DPL meta-network f_q; default feat_dim")
        parser.add_argument("--vector_length", type=int or None, default=None,
                            help="Number of query prompts N_Q for prompt_vector or DPL")
        parser.add_argument("--language_group", action="store_true", default=False)
        parser.add_argument("--num_heads", type=int, default=1)
        parser.add_argument("--bidirectional", action="store_false", default=True)
        parser.add_argument("--freeze_prototypes_epochs", type=int or None, default=5)
        parser.add_argument("--instance_prototype", type=str,
                            default="random", help="random, cluster, mapping")
        parser.add_argument("--concept_prototype", type=str or None,
                            default=None, help="random, cluster")
        parser.add_argument("--num_instance_prototypes", type=int, default=500)
        parser.add_argument("--num_concept_prototypes", type=int, default=500)
        parser.add_argument("--num_center", type=int, default=200)
        parser.add_argument("--prototype_orth", action="store_true", default=False)
        parser.add_argument("--gia_temperature", type=float, default=0.07)
        parser.add_argument("--tca_temperature", type=float, default=0.1)
        parser.add_argument("--ipa_temperature", type=float, default=0.2)
        parser.add_argument("--cpa_temperature", type=float, default=0.2)
        parser.add_argument("--group_temperature", type=float, default=0.3)
        parser.add_argument("--lambda_gia", type=float, default=1.)
        parser.add_argument("--lambda_tca", type=float, default=1.)
        parser.add_argument("--lambda_ipa", type=float, default=1.)
        parser.add_argument("--lambda_cpa", type=float, default=1.)
        parser.add_argument("--lambda_group", type=float, default=1.)
        parser.add_argument("--lambda_orth", type=float, default=1.)
        parser.add_argument("--without_loss_tca", action="store_true", default=False)
        parser.add_argument("--without_loss_ipa", action="store_true", default=False)
        parser.add_argument("--without_loss_cpa", action="store_true", default=False)
        parser.add_argument("--without_loss_group", action="store_true", default=False)
        
        # parser.add_argument("--img_encoder", type=str, default="vit_base")
        # parser.add_argument("--freeze_bert", action="store_true")
        # parser.add_argument("--emb_dim", type=int,
        #                     default=128, help="128, 256")
        # parser.add_argument("--num_workers", type=int, default=16)
        # parser.add_argument("--learning_rate", type=float, default=2e-5)
        # parser.add_argument("--momentum", type=float, default=0.9)
        # parser.add_argument("--mixed_precision", type=str,
        #                     default="32", help="64, 32, 16, bf16")
        # parser.add_argument("--weight_decay", type=float, default=0.05)
        # parser.add_argument("--batch_size", type=int, default=64)
        # parser.add_argument("--data_pct", type=float, default=1.)
        # # concept enhance
        # parser.add_argument("--rewrite", action="store_true", default=True)
        # parser.add_argument("--concept_decompose", action="store_true", default=True)
        # parser.add_argument("--max_concepts", type=int or None, default=40)
        # parser.add_argument("--prompt_vector", action="store_true", default=True)
        # parser.add_argument("--vector_length", type=int or None, default=40)
        # parser.add_argument("--language_group", action="store_true", default=False)
        # parser.add_argument("--num_heads", type=int, default=1)
        # parser.add_argument("--bidirectional", action="store_false", default=True)
        # parser.add_argument("--freeze_prototypes_epochs", type=int or None, default=5)
        # parser.add_argument("--instance_prototype", type=str,
        #                     default="random", help="random, cluster, mapping")
        # parser.add_argument("--concept_prototype", type=str or None,
        #                     default="random", help="random, cluster")
        # parser.add_argument("--num_instance_prototypes", type=int, default=500)
        # parser.add_argument("--num_concept_prototypes", type=int, default=500)
        # parser.add_argument("--num_center", type=int, default=200)
        # parser.add_argument("--prototype_orth", action="store_true", default=True)
        # parser.add_argument("--gia_temperature", type=float, default=0.07)
        # parser.add_argument("--tca_temperature", type=float, default=0.1)
        # parser.add_argument("--ipa_temperature", type=float, default=0.2)
        # parser.add_argument("--cpa_temperature", type=float, default=0.2)
        # parser.add_argument("--group_temperature", type=float, default=0.3)
        # parser.add_argument("--lambda_gia", type=float, default=1.)
        # parser.add_argument("--lambda_tca", type=float, default=1.)
        # parser.add_argument("--lambda_ipa", type=float, default=1.)
        # parser.add_argument("--lambda_cpa", type=float, default=1.)
        # parser.add_argument("--lambda_group", type=float, default=1.)
        # parser.add_argument("--lambda_orth", type=float, default=1.)

        parser.add_argument("--experiment_name", type=str, default="test")
        parser.add_argument("--ckpt_path", type=str or None, default=None)
        parser.add_argument("--seed", type=int, default=42)
        return parser

    @staticmethod
    def _use_ddp_or_dpp2(trainer: Trainer) -> bool:
        if trainer:
            return isinstance(trainer.training_type_plugin, (DDPPlugin, DDP2Plugin))
        else:
            return torch.distributed.is_initialized()

    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = dm.train_dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, trainer.num_gpus, trainer.num_processes)
        if trainer.tpu_cores:
            num_devices = max(num_devices, trainer.tpu_cores)
        effective_batch_size = trainer.accumulate_grad_batches * num_devices

        return (dataset_size // effective_batch_size) * trainer.max_epochs

@torch.no_grad()
def concat_all_gather(tensor):
    '''
    Performs all_gather operation on the provided tensors
    '''
    tensors_gather = [torch.ones_like(tensor) for _ in range(
        torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


class AttentionPool2d(nn.Module):
    def __init__(self, context_length: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(context_length + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.permute(1, 0, 2)  # NLC -> LNC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (L+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (L+1)NC
        x, weights = nn.functional.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=True
        )
        return x.squeeze(0), weights[:,0,1:]
