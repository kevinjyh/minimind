
# 以下是 `train_pretrain.py: def train_epoch` 函式所運行的重要操作，以文字大綱格式列出

| 屬性 | 值 |
|------|------|
| 行數 | 34-95 |
| 類型 | 傳入變數 |

---

## **1. 初始化**

- 設定損失函數 `loss_fct` 為 `nn.CrossEntropyLoss(reduction='none')`。
- 記錄訓練開始時間 `start_time`。

---

## **2. 遍歷訓練數據 (for loop)**

- 從 `train_loader` 中獲取每個批次數據 `(X, Y, loss_mask)`。
- 將 `X`、`Y` 和 `loss_mask` 移動到指定設備（如 GPU）。

---

### **3. 學習率更新**

- 計算當前學習率 `lr`，並更新優化器 `optimizer` 的學習率。

---

### **4. 模型前向傳播與損失計算**

- 在混合精度訓練的上下文 `ctx` 中，進行模型前向傳播 `res = model(X)`。
- 計算損失 `loss`：
  - 使用 `loss_fct` 計算交叉熵損失。
  - 將損失與 `loss_mask` 相乘，並取平均值。
  - 加入輔助損失 `res.aux_loss`（如果存在）。
  - 將損失除以梯度累積步數 `args.accumulation_steps`。

---

### **5. 反向傳播與梯度累積**

- 使用 `scaler.scale(loss).backward()` 進行反向傳播。
- 如果達到梯度累積步數 `args.accumulation_steps`：
  - 取消梯度縮放 `scaler.unscale_(optimizer)`。
  - 裁剪梯度 `torch.nn.utils.clip_grad_norm_`。
  - 更新模型參數 `scaler.step(optimizer)`。
  - 更新縮放器 `scaler.update()`。
  - 清除梯度 `optimizer.zero_grad(set_to_none=True)`。

---

### **6. 記錄訓練日誌**

- 如果達到日誌記錄間隔 `args.log_interval`：
  - 計算訓練時間 `spend_time`。
  - 使用 `Logger` 函式記錄當前 epoch、step、損失、學習率等資訊。
  - 如果啟用 `wandb`，將日誌記錄到 `wandb`。

---

### **7. 保存模型檢查點**

- 如果達到保存間隔 `args.save_interval`：
  - 將模型切換為評估模式 `model.eval()`。
  - 根據是否使用 `MoE` 模型，生成保存路徑 `ckp`。
  - 保存模型狀態字典 `state_dict` 到指定路徑。
  - 將模型切換回訓練模式 `model.train()`。

---

### **總結**

`train_epoch` 函式的主要操作包括：

1. 初始化訓練環境。
2. 遍歷訓練數據並更新學習率。
3. 進行模型前向傳播與損失計算。
4. 執行反向傳播與梯度累積。
5. 記錄訓練日誌。
6. 保存模型檢查點。
