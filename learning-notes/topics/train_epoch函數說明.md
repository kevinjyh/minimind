# `train_pretrain.py: train_epoch` 函數解說

這是一個訓練語言模型的核心函數，以下分段說明其功能：

## 1. **初始設置**

    ```python
    def train_epoch(epoch, wandb):
        loss_fct = nn.CrossEntropyLoss(reduction='none')  # 設置損失函數
        start_time = time.time()  # 記錄開始時間
    ```

    - 建立交叉熵損失函數，設置 `reduction='none'` 以便後續自定義損失計算
    - 記錄訓練開始時間，用於計算訓練速度

### 交叉熵損失函數詳解

1. **關於 `fct` 縮寫**：

    - `fct` 是 "function" 的常見縮寫
    - 在深度學習程式碼中，常見的命名慣例是使用 `loss_fn` 或 `loss_fct` 來表示損失函數

2. **CrossEntropy 函數原理**：
交叉熵損失函數的計算原理如下：

$$ \text{CrossEntropy} = -\sum_{i=1}^{n} y_i \log(\hat{y_i}) $$

其中：

- $y_i$ 是真實標籤（one-hot 編碼）
- $\hat{y_i}$ 是模型預測的機率分布（經過 softmax）
- $n$ 是類別數量

工作流程：

    1. 模型輸出經過 softmax 轉換為機率分布
    2. 計算預測機率的對數
    3. 與真實標籤相乘並求和
    4. 取負值得到最終損失

3. **關於 `reduction` 參數**：
PyTorch 中 `CrossEntropyLoss` 的 `reduction` 參數有三個選項：

- `'none'`：不進行降維，返回每個樣本的損失值
- `'mean'`（默認）：返回所有樣本損失的平均值
- `'sum'`：返回所有樣本損失的總和

關於 `'none'` vs `None`：

- `'none'` 是字符串，這是 PyTorch API 的設計選擇
- `None` 是 Python 的空值
- 這裡必須使用 `'none'` 字符串，因為這是 PyTorch 的 API 規範

使用 `reduction='none'` 的效果：

    ```python
    # 假設有批次大小為 2 的數據
    outputs = torch.tensor([[0.2, 0.8], [0.7, 0.3]])
    targets = torch.tensor([1, 0])

    # 使用 reduction='none'
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(outputs, targets)
    print(loss)  # 輸出形如：tensor([0.2231, 0.3567])

    # 使用 reduction='mean'（默認）
    loss_fct = nn.CrossEntropyLoss(reduction='mean')
    loss = loss_fct(outputs, targets)
    print(loss)  # 輸出形如：tensor(0.2899)
    ```

在這個專案中使用 `reduction='none'` 的原因：

1. 允許對每個樣本的損失進行自定義處理
2. 配合 `loss_mask` 實現對填充位置的損失忽略
3. 可以根據需要進行更靈活的損失計算

這種設計在處理變長序列時特別有用，因為：

- ✅ 可以忽略填充位置的損失
- ✅ 可以對不同位置賦予不同的權重
- ✅ 可以實現更複雜的損失計算策略

## 2. **數據處理與前向傳播**

    ```python
        for step, (X, Y, loss_mask) in enumerate(train_loader):
            X = X.to(args.device)        # 輸入數據移至指定設備
            Y = Y.to(args.device)        # 標籤移至指定設備
            loss_mask = loss_mask.to(args.device)  # 遮罩移至指定設備
    ```

- 從數據加載器獲取批次數據
- 將數據移至指定計算設備（GPU/CPU）

## 3. **學習率調整**

    ```python
            lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    ```

- 根據訓練進度動態調整學習率
- 使用餘弦退火策略更新優化器的學習率

## 4. **模型推理與損失計算**

    ```python
            with ctx:  # 自動混合精度上下文
                res = model(X)  # 模型前向傳播
                loss = loss_fct(
                    res.logits.view(-1, res.logits.size(-1)),
                    Y.view(-1)
                ).view(Y.size())
                loss = (loss * loss_mask).sum() / loss_mask.sum()  # 計算遮罩後的平均損失
                loss += res.aux_loss  # 加入輔助損失
                loss = loss / args.accumulation_steps  # 梯度累積
    ```

- 在混合精度上下文中執行模型推理
- 計算交叉熵損失並應用遮罩
- 加入模型的輔助損失（如有）
- 進行梯度累積的損失縮放

## 5. **反向傳播與優化**

    ```python
            scaler.scale(loss).backward()  # 損失反向傳播

            if (step + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)  # 反縮放梯度
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪
                scaler.step(optimizer)  # 優化器步進
                scaler.update()  # 更新梯度縮放器
                optimizer.zero_grad(set_to_none=True)  # 清空梯度
    ```

- 執行反向傳播
- 在累積步數達到時：
- 進行梯度反縮放
- 執行梯度裁剪
- 更新模型參數
- 重置梯度

## 6. **日誌記錄與模型保存**

    ```python
            if step % args.log_interval == 0:
                # 記錄訓練進度、損失值、學習率等信息
                Logger(...)
                if wandb is not None:
                    wandb.log(...)  # 記錄到 Weights & Biases

            if (step + 1) % args.save_interval == 0:
                # 保存模型檢查點
                model.eval()
                # 保存模型狀態
                torch.save(state_dict, ckp)
                model.train()
    ```

- 定期輸出訓練進度和指標
- 可選的 Weights & Biases 記錄
- 定期保存模型檢查點

## 特別功能

- ✅ 支持分布式訓練（DDP）
- ✅ 支持梯度累積
- ✅ 支持混合精度訓練
- ✅ 動態學習率調整
- ✅ 自動模型檢查點保存
- ✅ 完整的訓練監控和日誌記錄

這個函數實現了現代深度學習訓練中的多個重要特性，是一個完整的訓練循環實現。
