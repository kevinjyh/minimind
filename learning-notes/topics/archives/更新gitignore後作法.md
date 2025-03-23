# 更新 .gitignore 後的說明

在使用 Git 進行版本控制時，有些輸出圖檔及文件已經 PUSH 至 GitHub，這是為了方便我可以用手機 GitHub App 觀看這些圖檔及文件，但後續繼續測試源碼檔時，我不想每次執行測試時都得更新輸出檔，所以我希望新的輸出檔不要再被 git 監控了，此時得更新 `.gitignore` 文件來實現的。然而，當您想要從此時開始忽略某個目錄下的文件時，您需要先清理 Git 的緩存，以便 Git 不再監控這些文件，但又不刪除本地端檔案，以及遠端舊commit的輸出檔，此時要執行以下指令。

## 使用 `git rm` 解除文件監控

使用 `git rm` 命令可以從 Git 的索引中移除文件，但不會刪除本地文件。這意味著文件仍然存在於您的工作目錄中，但不會被 Git 跟蹤。

例如，您可以使用以下命令來解除對 `visualization_output` 目錄的監控：

```bash
git rm -r --cached visualization_output/
```

這個命令中的 `-r` 選項代表 "recursive"（遞歸），表示 Git 將遞歸地移除指定目錄及其所有子目錄和文件。這對於刪除整個目錄及其內容非常有用。

## 提交更改

在解除監控後，您需要提交這些更改，以更新 Git 的狀態：

```bash
git commit -m "Update .gitignore to ignore visualization_output files"
```

## 確認忽略效果

您可以使用以下命令來確認 Git 現在是否忽略了 `visualization_output` 目錄下的文件：

```bash
git status
```

如果一切正常，您應該不會看到 `visualization_output` 目錄下的文件出現在未追蹤的文件列表中。

## 總結

- `git rm` 不會刪除實際的檔案，只是解除監控。
- `-r` 選項表示遞歸，適用於刪除目錄及其內容。
