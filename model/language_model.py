from transformers import GPT2Model, GPT2Config,GPT2PreTrainedModel
from transformers.file_utils import  add_code_sample_docstrings, add_start_docstrings
import torch.nn as nn
import torch

class GPT2Reassemble(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config=config
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def init_embedding(self):
        embedding_matrix=torch.rand(self.config.vocab_size,self.config.n_embd)
        nn.init.xavier_uniform_(embedding_matrix)
        self.transformer.set_input_embeddings(embedding_matrix)

    def get_output_embeddings(self):
        return self.lm_head

    def prepare_inputs_for_generation(self, input_ids, past, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        return {"input_ids": input_ids, "past": past, "use_cache": kwargs["use_cache"]}

    def forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        outputs = (lm_logits,) + transformer_outputs[1:]


        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)
