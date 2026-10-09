import math

import torch
from torch import nn

from descfuse import FACETS
from descfuse.facets import FacetGenerator


class DocumentEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, mask):
        embeddings = self.dropout(self.embedding(input_ids))
        packed = nn.utils.rnn.pack_padded_sequence(embeddings, mask.sum(-1).cpu(),
                                                   batch_first=True, enforce_sorted=False)
        states, _ = self.rnn(packed)
        states, _ = nn.utils.rnn.pad_packed_sequence(states, batch_first=True, total_length=input_ids.shape[1])
        return self.dropout(states)


class EvidenceFusion(nn.Module):
    def __init__(self, dim, mlp_hidden, dropout=0.3, use_null=True, use_interaction=True):
        super().__init__()
        self.dim, self.use_null, self.use_interaction = dim, use_null, use_interaction
        self.query, self.key, self.value = [nn.Linear(dim, dim, bias=False) for _ in range(3)]
        self.null_key = nn.Parameter(torch.empty(dim))
        nn.init.normal_(self.null_key, std=0.02)
        self.matching = nn.Linear(4 * dim, dim, bias=False)
        self.gate_weight = nn.Parameter(torch.zeros(dim))
        self.identity = nn.Parameter(torch.empty(len(FACETS), dim))
        nn.init.normal_(self.identity, std=0.02)
        self.match_norm = nn.LayerNorm(dim)
        self.cross_query, self.cross_key, self.cross_value = [nn.Linear(dim, dim, bias=False) for _ in range(3)]
        self.cross_norm, self.ffn_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim))
        self.pool = nn.Linear(dim, 1, bias=False)
        self.predictor = nn.Sequential(nn.Linear(dim, mlp_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(mlp_hidden, 1))

    def forward(self, document, document_mask, queries, facet_mask, diagnostics=False):
        if not facet_mask.any(-1).all():
            raise ValueError("Every code requires at least one applicable facet.")
        q, key = self.query(queries), self.key(document)
        scores = torch.einsum("ckd,bnd->bckn", q, key) / math.sqrt(self.dim)
        scores = scores.masked_fill(~document_mask[:, None, None, :], float("-inf"))
        if self.use_null:
            null_scores = (q @ self.null_key / math.sqrt(self.dim))[None, :, :, None]
            scores = torch.cat([scores, null_scores.expand(len(document), -1, -1, -1)], dim=-1)
        attention = scores.softmax(-1)
        null = attention[..., -1] if self.use_null else scores.new_zeros(scores.shape[:-1])
        doc_attention = attention[..., :-1] if self.use_null else attention
        evidence = torch.einsum("bckn,bnd->bckd", doc_attention, self.value(document))
        gate = 1 - null
        q_full = queries[None].expand(len(document), -1, -1, -1)
        features = torch.cat([q_full, evidence, q_full * evidence, q_full - evidence], dim=-1)
        matched = self.match_norm(self.matching(features) + gate[..., None] * self.gate_weight + self.identity)
        cross = None
        if self.use_interaction:
            cross_scores = self.cross_query(matched) @ self.cross_key(matched).transpose(-1, -2) / math.sqrt(self.dim)
            cross_scores = cross_scores.masked_fill(~facet_mask[None, :, None, :], float("-inf"))
            cross = cross_scores.softmax(-1)
            matched = self.cross_norm(matched + cross @ self.cross_value(matched))
            matched = self.ffn_norm(matched + self.ffn(matched))
        pool_scores = self.pool(matched).squeeze(-1).masked_fill(~facet_mask[None], float("-inf"))
        weights = pool_scores.softmax(-1)
        pooled = (weights[..., None] * matched).sum(-2)
        logits = self.predictor(pooled).squeeze(-1)
        detail = {"null_probability": null, "facet_weights": weights, "cross_attention": cross,
                  "token_attention": doc_attention, "evidence": evidence} if diagnostics else None
        return logits, detail


class DescFuse(nn.Module):
    def __init__(self, config, codes, vocab_size, generator=None):
        super().__init__()
        if 2 * config["hidden_size"] != config["facet_dim"]:
            raise ValueError("BiLSTM output and facet dimensions must match.")
        self.generator = generator if generator is not None else FacetGenerator(config, codes)
        self.encoder = DocumentEncoder(vocab_size, config["embedding_dim"], config["hidden_size"], config["dropout"])
        self.fusion = EvidenceFusion(config["facet_dim"], config["prediction_hidden"], config["dropout"],
                                     config.get("use_null", True), config.get("use_interaction", True))
        self.register_buffer("facet_mask", torch.tensor([[f["supported"] for f in row["facets"]] for row in codes]))

    def forward(self, batch, code_indices, description_indices=None, stop_gradient=False):
        document = self.encoder(batch["input_ids"], batch["mask"])
        queries, states = self.generator(code_indices)
        if stop_gradient:
            queries = queries.detach()
        logits, _ = self.fusion(document, batch["mask"], queries, self.facet_mask[code_indices])
        desc_loss = logits.sum() * 0.0
        if description_indices is not None:
            if torch.equal(code_indices, description_indices):
                desc_states = states
            else:
                _, desc_states = self.generator(description_indices)
            desc_loss = self.generator.reconstruction_loss(desc_states, description_indices)
        return logits, desc_loss

    @torch.no_grad()
    def predict_all(self, batch, block_size=256):
        if self.training:
            raise RuntimeError("Call eval() before full-label inference.")
        document = self.encoder(batch["input_ids"], batch["mask"])
        scores = []
        for start in range(0, len(self.facet_mask), block_size):
            indices = torch.arange(start, min(start + block_size, len(self.facet_mask)), device=document.device)
            queries, _ = self.generator(indices)
            logits, _ = self.fusion(document, batch["mask"], queries, self.facet_mask[indices])
            scores.append(logits.sigmoid())
        return torch.cat(scores, dim=1)
