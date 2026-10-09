import torch
import torch.nn.functional as F
from torch import nn

from descfuse import FACETS


INSTRUCTIONS = (
    "Describe the clinical definition.",
    "Describe symptoms and signs.",
    "Describe diagnostic evidence and investigations.",
    "Describe treatment and management.",
    "Describe complications and clinical consequences.",
    "Summarize the clinical profile.",
)


class FacetGenerator(nn.Module):
    def __init__(self, config, codes, llm=None, tokenizer=None):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if llm is None:
            dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
                     "float16": torch.float16}[config.get("llm_dtype", "bfloat16")]
            llm = AutoModelForCausalLM.from_pretrained(
                config["base_model"], torch_dtype=dtype,
                attn_implementation="eager", trust_remote_code=False)
            tokenizer = AutoTokenizer.from_pretrained(config["base_model"], trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer must define EOS and padding tokens.")
        tokenizer.padding_side = "right"
        self.tokenizer, self.codes = tokenizer, codes
        self.llm = get_peft_model(llm, LoraConfig(
            r=config.get("lora_rank", 2), lora_alpha=config.get("lora_alpha", 32),
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
            target_modules=config.get("lora_targets", ["q_proj", "v_proj"])))
        self.llm.config.use_cache = False
        if config.get("gradient_checkpointing", True):
            self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        width = llm.config.hidden_size
        self.slots = nn.Parameter(torch.empty(len(FACETS), width))
        nn.init.normal_(self.slots, std=0.02)
        self.project = nn.Sequential(nn.Linear(width, config["facet_dim"]), nn.LayerNorm(config["facet_dim"]))
        self.prefix_norm = nn.LayerNorm(width)
        self.gamma = nn.Parameter(torch.ones(()))
        self.max_prompt = config.get("max_prompt_tokens", 128)
        self.max_reference = config.get("max_reference_tokens", 160)
        self.micro_batch = config.get("facet_micro_batch", 12)

    @property
    def device(self):
        return self.slots.device

    def _encode_pairs(self, pairs):
        prompts = [f"ICD {self.codes[c]['code']}: {self.codes[c]['description']}\n{INSTRUCTIONS[k]}"
                   for c, k in pairs]
        batch = self.tokenizer(prompts, padding=True, truncation=True,
                               max_length=self.max_prompt, return_tensors="pt").to(self.device)
        embeds = self.llm.get_input_embeddings()(batch.input_ids)
        lengths = batch.attention_mask.sum(-1)
        # 槽位紧接有效提示词，避免右侧填充改变其位置。
        embeds = F.pad(embeds, (0, 0, 0, 1))
        rows = torch.arange(len(pairs), device=self.device)
        facet_ids = torch.tensor([k for _, k in pairs], device=self.device)
        embeds[rows, lengths] = self.slots[facet_ids].to(embeds.dtype)
        mask = torch.arange(embeds.shape[1], device=self.device)[None, :] <= lengths[:, None]
        # 直接读取骨干网络，编码阶段无需计算全词表 logits。
        backbone = self.llm.get_base_model().model
        output = backbone(inputs_embeds=embeds, attention_mask=mask.long(), use_cache=False, return_dict=True)
        return output.last_hidden_state[rows, lengths].float()

    def forward(self, code_indices):
        indices = [int(i) for i in code_indices]
        pairs = [(c, k) for c in indices for k in range(len(FACETS))]
        states = torch.cat([self._encode_pairs(pairs[i:i + self.micro_batch])
                            for i in range(0, len(pairs), self.micro_batch)])
        states = states.view(len(indices), len(FACETS), -1)
        return self.project(states), states

    def reconstruction_loss(self, states, code_indices):
        pairs = [(j, k) for j, c in enumerate(code_indices) for k in range(len(FACETS))
                 if self.codes[int(c)]["facets"][k]["supported"]]
        total, count = states.sum() * 0.0, 0
        for start in range(0, len(pairs), self.micro_batch):
            chunk = pairs[start:start + self.micro_batch]
            targets = []
            for j, k in chunk:
                text = self.codes[int(code_indices[j])]["facets"][k]["description"]
                ids = self.tokenizer.encode(text, add_special_tokens=False)[:self.max_reference - 1]
                targets.append(torch.tensor(ids + [self.tokenizer.eos_token_id], device=self.device))
            ids = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            lengths = torch.tensor([len(t) for t in targets], device=self.device)
            target_mask = torch.arange(ids.shape[1], device=self.device)[None, :] < lengths[:, None]
            prefix = self.gamma * self.prefix_norm(torch.stack([states[j, k] for j, k in chunk]))
            word_embeddings = self.llm.get_input_embeddings()(ids)
            inputs = torch.cat([prefix[:, None].to(word_embeddings.dtype), word_embeddings], dim=1)
            mask = torch.cat([torch.ones(len(chunk), 1, device=self.device, dtype=torch.bool), target_mask], dim=1)
            logits = self.llm(inputs_embeds=inputs, attention_mask=mask.long(), use_cache=False).logits[:, :-1]
            labels = ids.masked_fill(~target_mask, -100)
            total = total + F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1),
                                           ignore_index=-100, reduction="sum")
            count += int(target_mask.sum())
        return total / max(count, 1)

    @torch.no_grad()
    def decode(self, states, max_new_tokens=128):
        prefix = self.gamma * self.prefix_norm(states.reshape(-1, states.shape[-1]))
        outputs = []
        for i in range(0, len(prefix), self.micro_batch):
            embeddings = prefix[i:i + self.micro_batch, None].to(self.llm.get_input_embeddings().weight.dtype)
            ids = self.llm.generate(inputs_embeds=embeddings,
                                    attention_mask=torch.ones(embeddings.shape[:2], device=self.device, dtype=torch.long),
                                    max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                                    pad_token_id=self.tokenizer.pad_token_id,
                                    eos_token_id=self.tokenizer.eos_token_id)
            outputs.extend(self.tokenizer.batch_decode(ids, skip_special_tokens=True))
        return [outputs[i:i + len(FACETS)] for i in range(0, len(outputs), len(FACETS))]
