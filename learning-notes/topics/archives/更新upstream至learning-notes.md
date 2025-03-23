# 更新上游源碼

要將 `upstream` 的源碼 fetch 到你專屬的 `learning-main` 分支，可以按照以下步驟進行：

1. **切換到 `learning-main` 分支**：

   ```bash
   git checkout learning-main
   ```

2. **從 `upstream` fetch 最新的變更**：

   ```bash
   git fetch upstream
   ```

3. **將 `upstream` 的變更合併到你的 `learning-main` 分支**：

   ```bash
   git merge upstream/master
   ```

或者，如果你希望直接將 `upstream` 的變更拉取到你的分支而不需要手動合併，可以使用以下指令：

```bash
git pull upstream master
```

這會自動 fetch 並合併 `upstream/master` 的變更到你的 `learning-main` 分支。
