import copy
import itertools
import json
import unittest
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from descfuse.data import load_codes
from descfuse.facets import FacetGenerator
from descfuse.metrics import equal_frequency_ece, f1_scores, select_threshold
from descfuse.model import DescFuse, EvidenceFusion
from descfuse.objectives import sample_candidates, sampled_bce
from descfuse.references import BM25, validate_reference


def tiny_components():
    codes = load_codes(Path(__file__).resolve().parents[1] / "sample/codes.json")[:2]
    words = ["[PAD]", "[UNK]", "[EOS]", "ICD", "Describe", "the", "clinical", "definition"]
    vocab = {word: i for i, word in enumerate(words)}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]")
    llm = LlamaForCausalLM(LlamaConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
                                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                      max_position_embeddings=256, attention_dropout=0.0,
                                      pad_token_id=0, eos_token_id=2))
    config = {"facet_dim": 16, "hidden_size": 8, "embedding_dim": 8, "prediction_hidden": 24,
              "dropout": 0.0, "lora_rank": 2, "lora_alpha": 32, "lora_targets": ["q_proj", "v_proj"],
              "max_prompt_tokens": 32, "max_reference_tokens": 16, "facet_micro_batch": 6,
              "gradient_checkpointing": False}
    generator = FacetGenerator(config, codes, llm, tokenizer)
    return config, codes, generator


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_classification_reaches_shared_lora_and_slots(self):
        config, codes, generator = tiny_components()
        model = DescFuse(config, codes, 12, generator)
        batch = {"input_ids": torch.tensor([[1, 2, 3], [4, 5, 0]]),
                 "mask": torch.tensor([[True, True, True], [True, True, False]])}
        logits, _ = model(batch, torch.arange(2))
        torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([[1., 0.], [0., 1.]])).backward()
        lora_grad = sum(float(p.grad.abs().sum()) for n, p in generator.llm.named_parameters()
                        if "lora_" in n and p.grad is not None)
        self.assertGreater(lora_grad, 0)
        self.assertGreater(float(generator.slots.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for n, p in generator.llm.named_parameters() if "lora_" not in n))

    def test_reconstruction_reaches_slots_and_lora(self):
        _, _, generator = tiny_components()
        _, states = generator(torch.arange(2))
        loss = generator.reconstruction_loss(states, torch.arange(2))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(generator.slots.grad.abs().sum()), 0)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for n, p in generator.llm.named_parameters() if "lora_" in n))

    def test_joint_gradient_checkpointing(self):
        config, codes, generator = tiny_components()
        generator.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model = DescFuse(config, codes, 12, generator).train()
        batch = {"input_ids": torch.tensor([[1, 2]]), "mask": torch.ones(1, 2, dtype=torch.bool)}
        indices = torch.arange(2)
        logits, reconstruction = model(batch, indices, indices)
        loss = logits.square().mean() + 0.01 * reconstruction
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(generator.slots.grad.abs().sum()), 0)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for n, p in generator.llm.named_parameters() if "lora_" in n))

    def test_blocked_gradient_control(self):
        config, codes, generator = tiny_components()
        model = DescFuse(config, codes, 12, generator)
        batch = {"input_ids": torch.tensor([[1, 2]]), "mask": torch.ones(1, 2, dtype=torch.bool)}
        logits, _ = model(batch, torch.arange(2), stop_gradient=True)
        logits.sum().backward()
        self.assertIsNone(generator.slots.grad)

    def test_null_and_padding(self):
        fusion = EvidenceFusion(16, 24, dropout=0).eval()
        document, queries = torch.randn(1, 3, 16), torch.randn(2, 6, 16)
        mask = torch.tensor([[True, True, False]])
        facet_mask = torch.ones(2, 6, dtype=torch.bool)
        _, details = fusion(document, mask, queries, facet_mask, True)
        self.assertTrue(torch.equal(details["token_attention"][..., 2], torch.zeros(1, 2, 6)))
        self.assertTrue(torch.allclose(details["token_attention"].sum(-1) + details["null_probability"], torch.ones(1, 2, 6)))
        logits, empty = fusion(document, torch.zeros_like(mask), queries, facet_mask, True)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.equal(empty["null_probability"], torch.ones(1, 2, 6)))
        self.assertTrue(torch.equal(empty["evidence"], torch.zeros(1, 2, 6, 16)))

    def test_applicability_masks_cross_keys_and_pool(self):
        fusion = EvidenceFusion(16, 24, dropout=0).eval()
        facets = torch.tensor([[True, True, False, False, False, False]])
        _, details = fusion(torch.randn(1, 3, 16), torch.ones(1, 3, dtype=torch.bool),
                            torch.randn(1, 6, 16), facets, True)
        self.assertTrue((details["cross_attention"][..., 2:] == 0).all())
        self.assertTrue((details["facet_weights"][..., 2:] == 0).all())
        with self.assertRaises(ValueError):
            fusion(torch.randn(1, 3, 16), torch.ones(1, 3, dtype=torch.bool),
                   torch.randn(1, 6, 16), torch.zeros_like(facets))

    def test_full_label_block_equivalence(self):
        config, codes, generator = tiny_components()
        model = DescFuse(config, codes, 12, generator).eval()
        batch = {"input_ids": torch.tensor([[1, 2, 3]]), "mask": torch.ones(1, 3, dtype=torch.bool)}
        self.assertTrue(torch.allclose(model.predict_all(batch, 1), model.predict_all(batch, 2), atol=1e-6))

    def test_importance_estimator_exact_expectation(self):
        labels = torch.tensor([[1., 0., 0., 0.], [0., 0., 0., 0.]])
        logits = torch.tensor([[0.3, -1., 0.2, 2.], [-0.5, 1., -2., 0.7]])
        estimates = []
        for subset in itertools.combinations([1, 2, 3], 2):
            indices = torch.tensor([0, *subset])
            estimates.append(sampled_bce(logits[:, indices], labels, indices, torch.tensor([1., 1.5, 1.5])))
        actual = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        self.assertTrue(torch.allclose(torch.stack(estimates).mean(), actual))
        indices, weights = sample_candidates(labels, 1)
        self.assertIn(0, indices.tolist())
        self.assertEqual(len(indices), 2)
        self.assertEqual(float(weights[-1]), 3.)

    def test_metrics_and_threshold(self):
        labels = np.array([[1, 0, 0], [0, 1, 0]])
        probability = np.array([[0.9, 0.2, 0.1], [0.2, 0.8, 0.1]])
        result = f1_scores(labels, probability, 0.5)
        self.assertEqual(result["micro_f1"], 1.)
        self.assertAlmostEqual(result["macro_f1"], 2 / 3)
        self.assertEqual(f1_scores(labels, probability, select_threshold(labels, probability))["micro_f1"], 1.)
        self.assertAlmostEqual(equal_frequency_ece([1, 0], [1., 0.]), 0.)

    def test_bm25_and_reference_validation(self):
        index = BM25([{"id": "a", "text": "thyroid goiter"}, {"id": "b", "text": "bacterial pneumonia"}])
        self.assertEqual(index.search("thyroid", 1)[0]["id"], "a")
        self.assertEqual(index.search("unmatched", 3), [])
        row = load_codes(Path(__file__).resolve().parents[1] / "sample/codes.json")[0]
        response = {"facets": copy.deepcopy(row["facets"])}
        allowed = {source for f in row["facets"] for source in f["source_ids"]}
        validate_reference(response, allowed, row["code"])
        response["facets"][0]["source_ids"] = ["invented"]
        with self.assertRaises(ValueError):
            validate_reference(response, allowed, row["code"])


if __name__ == "__main__":
    unittest.main()
