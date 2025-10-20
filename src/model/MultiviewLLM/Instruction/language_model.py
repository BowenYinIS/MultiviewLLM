import torch


class MultiviewLLM(torch.nn.Module):
    def __init__(self, language_model, projector):
        super(MultiviewLLM, self).__init__()
        self.language_model = language_model
        self.projector = projector

    def forward(self, input_ids, is_graph, is_ts,
                graph_x, graph_x_pad,
                ts_x, ts_x_pad,
                attn_mask, labels=None):
        # Project multimodal embeddings
        embeds = self.projector(
            input_ids=input_ids,
            is_graph=is_graph,
            is_ts=is_ts,
            graph_x=graph_x,
            graph_x_pad=graph_x_pad,
            ts_x=ts_x,
            ts_x_pad=ts_x_pad,
        )

        # Integrate projected embeddings into the language model
        outputs = self.language_model(
            inputs_embeds=embeds,
            attention_mask=attn_mask,
            labels=labels,
            use_cache=False,
        )
        return outputs