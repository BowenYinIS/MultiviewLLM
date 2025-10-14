Findings 
---
1. Sample 500 数据，大约需要40-50 epochs 收敛

Data flow 
---
In model.py 

1. Raw data: [seq_len, 6]
2. Transaction embedding (MLP) [seq_len, hidden=128] -> padding [max_length, hidden]
3. mask (all ones (using bert)), [batch_size, max_len, d_model]
4. Add positional encoding (未check)
5. Transformer -> token_embeddings & attention_mask

Pretrain
---
0. apply_masking (in model.py) 每一条seq都15% mask (before padding $\checkmark$)
1. token embedding -> self.mcc_predictor (MLP)
2. self.amt_predictor (MLP)
3. loss computation (13 classes mcc, log(1+amt) -> N(0,1) MSE loss)
