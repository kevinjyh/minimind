## Q1:

1.Transformer 中的可訓練 Queries、Keys 和Values 矩陣從哪兒來？ Transformer 中為何會有Queries、Keys和
Values矩陣，只設置Values 矩陣本身來求Attention 不是更簡單嗎？

### A1:

Queries(查詢)，Keys(鍵)和Values(值)矩陣是通過線性變換行輸入的問詞嵌入向量得到的，這些矩陣顯通過認斷想致，它們的作用是將輸入的商造入向量安計的問員只要的空間
且通過學習工程中區，同里莫中的參品，以使模型能夠更好地消費“人序列中的房人人資訊即關
又是因為在巨注意力 機老 （Sell-Altention Mect anism 中，需要通過0000 和rene [Keyso相百關
聯度來計管注意力權重，不后再福回這些權重對valuesi進行加僅求和。 這件設計的此游在等
九件提型設計算注意力的同時考到不到不同仙桃之間的聽划關係，從而更好地虎捉到驗人序列中的
上下文##
至於為I什麼不只收 Velues起年來水Attenbon，而是是要同時使用QuenestDKaysiela體，原因開於
Queiesfokeystep）， 名福世界（·），從而便植型##龙心晚地址，主意力權手，只使用
Velues）티斗可能會제 初通型的表達能力，無法開分利用#人序列中的資訊。
2. Transformer 的 Feed Forward 層在訓練的時候到底在訓
練什麼？
Fead ForwandETranslomer中的訓練過程中，通過特定是取和中生性時快學習作入序列的
表示，從而力機型的下，並任各小更好的廣動人特徵，在訓練過程中，Feed Forkaid旨的參似是
通過反向傳推同法和精度下專優化方法幹學習的，通過最小化1坐在（再生上們為失監款，專業
會自動調整Feed Forwaid層中的僅事和看知道 以上手機型：更好地區公司合訓或居，並且在在來
見過的數道上具有良好的泛化能力。