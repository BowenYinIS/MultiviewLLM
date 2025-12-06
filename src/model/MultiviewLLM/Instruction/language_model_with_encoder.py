import torch


class MultiviewLLM(torch.nn.Module):
    def __init__(self, graph_model, ts_model, language_model, projector):
        super(MultiviewLLM, self).__init__()
        self.graph_model = graph_model
        self.ts_model = ts_model
        self.language_model = language_model
        self.projector = projector

    @staticmethod
    def pad_and_mask(tensor_list):
        B = len(tensor_list)
        max_len = max(t.size(0) for t in tensor_list)
        d = tensor_list[0].size(1)

        padded = torch.zeros(B, max_len, d, device=tensor_list[0].device)
        mask = torch.zeros(B, max_len, dtype=torch.bool, device=tensor_list[0].device)

        for i, t in enumerate(tensor_list):
            L = t.size(0)
            padded[i, :L] = t
            mask[i, :L] = 1
        return padded, mask

    def forward(self, data):
        embeds, attn_mask, labels = self.forward_embeddings(data)

        # Integrate projected embeddings into the language model
        outputs = self.language_model(
            inputs_embeds=embeds,
            attention_mask=attn_mask,
            labels=labels,
            use_cache=False,
        )
        return outputs

    def forward_embeddings(self, data):
        # Project multimodal embeddings
        input_ids = data['input_ids']
        labels = data['labels']
        graph_data = data['graph_data']
        is_graph = data['is_graph']
        ts_data = data['ts_data']
        is_ts = data['is_ts']
        attn_mask = data['attn_mask']
        transaction_num_lis = data['transaction_number_lis']

        # graph_embeddings
        # z, g = self.graph_model.forward_without_augment(graph_data.x, graph_data.edge_index, graph_data.edge_attr, graph_data.batch)
        # graph_sizes = torch.bincount(graph_data.batch)
        # z_pre_graph = list(torch.split(z, graph_sizes.tolist(), dim=0))
        # z_padding, z_mask = self.pad_and_mask(z_pre_graph)
        z_padding, z_mask = None, None

        # ts_embeddings
        ts = self.ts_model(ts_data)
        ts_padding = ts['token_embeddings']
        ts_mask = ~ts['attention_mask']
        # process ts embedding
        billing_cycle_num = transaction_num_lis.size(1)
        ts_flat = ts_padding[ts_mask]  # (total_valid_tokens, d_t)
        transaction_num_lis_flat = transaction_num_lis.reshape(-1)  # (B * billing_cycle_num,)
        ts_per_billing_cycle = list(torch.split(ts_flat, transaction_num_lis_flat.tolist(), dim=0))
        ts_per_billing_cycle_padded, ts_per_billing_cycle_mask = self.pad_and_mask(ts_per_billing_cycle)

        embeds = self.projector(
            input_ids=input_ids,
            is_graph=is_graph,
            is_ts=is_ts,
            graph_x=z_padding,
            graph_x_mask=z_mask,
            ts_x=ts_per_billing_cycle_padded,
            ts_x_mask=ts_per_billing_cycle_mask,
            billing_cycle_num=billing_cycle_num,
        )
        return embeds, attn_mask, labels
