# 以下是 `train_pretrain.py` 和 `train_full_sft.py` 兩個源碼檔的主要差異分析

---

## **1. 訓練目的**

- **`train_pretrain.py`**:
  - 用於 **預訓練（Pretraining）**，目標是讓模型在大量未標註數據上學習語言的通用特徵和結構。
  - 模型從頭開始訓練，學習如何預測下一個詞（或 token）。

- **`train_full_sft.py`**:
  - 用於 **全量監督微調（Full Supervised Fine-Tuning, SFT）**，目標是在標註數據上進一步調整模型，使其適應特定任務。
  - 模型會加載預訓練的權重，並在標註數據上進行微調。

---

## **2. 模型初始化**

- **`train_pretrain.py`**:
  - 模型從頭初始化，不載入任何預訓練權重。

  ```python
  model = MiniMindLM(lm_config).to(args.device)
  ```

- **`train_full_sft.py`**:
  - 模型會加載預訓練的權重，並在標註數據上進行微調。

  ```python
  model = MiniMindLM(lm_config)
  ckp = f'./out/pretrain_{lm_config.dim}{moe_path}.pth'
  state_dict = torch.load(ckp, map_location=args.device)
  model.load_state_dict(state_dict, strict=False)
  ```

---

## **3. 數據集**

- **`train_pretrain.py`**:
  - 使用 `PretrainDataset`，處理未標註的文本數據。

  ```python
  train_ds = PretrainDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
  ```

- **`train_full_sft.py`**:
  - 使用 `SFTDataset`，處理標註的監督數據。

  ```python
  train_ds = SFTDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
  ```

---

## **4. 學習率**

- **`train_pretrain.py`**:
  - 學習率較高，通常為 `5e-4`。

  ```python
  parser.add_argument("--learning_rate", type=float, default=5e-4)
  ```

- **`train_full_sft.py`**:
  - 學習率較低，通常為 `5e-5`，因為微調時不需要大幅調整模型參數。

  ```python
  parser.add_argument("--learning_rate", type=float, default=5e-5)
  ```

---

## **5. 梯度累積步數**

- **`train_pretrain.py`**:
  - 梯度累積步數較多，通常為 `8`。

  ```python
  parser.add_argument("--accumulation_steps", type=int, default=8)
  ```

- **`train_full_sft.py`**:
  - 梯度累積步數較少，通常為 `1`。

  ```python
  parser.add_argument("--accumulation_steps", type=int, default=1)
  ```

---

## **6. 模型保存名稱**

- **`train_pretrain.py`**:
  - 保存的模型名稱包含 `pretrain`。

  ```python
  ckp = f'{args.save_dir}/pretrain_{lm_config.dim}{moe_path}.pth'
  ```

- **`train_full_sft.py`**:
  - 保存的模型名稱包含 `full_sft`。

  ```python
  ckp = f'{args.save_dir}/full_sft_{lm_config.dim}{moe_path}.pth'
  ```

---

## **7. Wandb 專案名稱**

- **`train_pretrain.py`**:
  - Wandb 專案名稱為 `MiniMind-Pretrain`。

  ```python
  parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")
  ```

- **`train_full_sft.py`**:
  - Wandb 專案名稱為 `MiniMind-Full-SFT`。

  ```python
  parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT")
  ```

---

## **8. 數據路徑**

- **`train_pretrain.py`**:
  - 使用未標註的預訓練數據路徑 `pretrain_hq.jsonl`。

  ```python
  parser.add_argument("--data_path", type=str, default="./dataset/pretrain_hq.jsonl")
  ```

- **`train_full_sft.py`**:
  - 使用標註的監督數據路徑 `sft_mini_512.jsonl`。

  ```python
  parser.add_argument("--data_path", type=str, default="./dataset/sft_mini_512.jsonl")
  ```

---

## **總結**

| 特性                | `train_pretrain.py`               | `train_full_sft.py`               |
|---------------------|-----------------------------------|-----------------------------------|
| **訓練目的**        | 預訓練（Pretraining）             | 全量監督微調（Full SFT）          |
| **模型初始化**      | 從頭訓練                         | 加載預訓練權重並微調              |
| **數據集**          | `PretrainDataset`（未標註數據）   | `SFTDataset`（標註數據）          |
| **學習率**          | 較高（`5e-4`）                    | 較低（`5e-5`）                    |
| **梯度累積步數**    | 較多（`8`）                       | 較少（`1`）                       |
| **模型保存名稱**    | `pretrain_*.pth`                  | `full_sft_*.pth`                  |
| **Wandb 專案名稱**  | `MiniMind-Pretrain`               | `MiniMind-Full-SFT`               |
| **數據路徑**        | `pretrain_hq.jsonl`               | `sft_mini_512.jsonl`              |

這兩個腳本的主要差異在於 **訓練目的** 和 **數據處理方式**，`train_pretrain.py` 用於預訓練，而 `train_full_sft.py` 用於監督微調。
