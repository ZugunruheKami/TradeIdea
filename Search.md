时隔7年昨天回到了腾讯滨海大厦，滨海里人少了，爱马哥咖啡里依旧一堆可爱企鹅。关于搜索+AI在2026年的一些发展我最后讲了几个hot take都不是特别乐观，大家看个乐儿就行：

1. 我觉得2026年不是做搜索最好的一年，但也不算最差的一年。最好的一年是2025，DeepResearch系产品从deepseek-r1之后爆火，用大模型+搜索写带grounding的长篇大论成为最大的落地场景。无论是jina reader还是我们当时的国内外竞对：博查firecrawl serper-dev从25年三四月份开始都一下就起来了。最差的一年我觉得可能是2023年chatgpt刚出来不久：传统混合搜索pipeline面临淘汰frontier lab总爱拿RAG秀肌肉。市场对向量数据库充满了非理性认知。26年之所以不算差，是因为搜索已经不再是frontier labs的关注点了，大家关注更多的是long-horizon task下模型的性能和表现，关注百年未解数学题上的突破。搜索不够sexy太pedestrain。而且长程任务下搜索的比例其实被稀释了很多。如果说25年deepresearch中搜索重要性是80%，那么26年30小时的长程任务下搜索的重要性也就10%。好处就是当头部公司不再关注搜索时，市场可以对搜索的认知回归正常，想做搜索的人可以静下心来做。坏处就是当一个领域不够hot不够frontier 它很难吸引到顶级的人才，现存的high agency人才也会pivot到其他赛道。

2. 诸如embedding reranker这种小模型的训练到今年年底应该会被autoresearch吃掉（如果吃不掉，AR就是伪科学）2026年，在搜索的training-time 投入更多的精力和成本已经没有太多意义：除了被AR吃，训练时模型性能提升的回报在应用场景下在test-time被稀释或完全抹平。极端情况下你的embedding模型还要和grep在大模型面前争宠，大模型要是选择用grep不用你，那你把MTEB排行榜捅破了天也没有用。

3. 因此26年之后按照传统的方式去做搜索的研发会越来越无聊，可以尝试的新方向不多。我认为可以做一做的是把未来的搜索看作一个test-time compute/scaling的问题。简单来讲就是我能不能在固定模型权重的情况下，通过推理时间买相关性买新功能。比如大模型在test-time时grep->embed->read小pipeline的生成优化及这条小pipe的实时cache和复用，甚至一些test-time training的工作和把搜索和world model结合也许会有些火花产生。总之training-time的搜索没太多搞头了，test-time的搜索需要把搜索时间当作一种货币而不是系统的属性去买东西买相关性买新功能。

4. 搜索系统是一个overfit好于underfit的系统。也就是说一个企业如果有能力搭建自己的内部搜索系统，所有的索引召回排序行为都只适配那个企业内部的文档和数据管线，那么只要不拿出来当通用产品卖，这就不是一件坏事儿，反而是最优解。而今天搭建一个内部overfit的搜索需要什么呢？一个喜欢崩老登的00后+两罐红牛+一个max coding plan+一个周末。所以做通用搜索框架模型管线的公司每天都应该早上起来对着镜子问问自己：我一个underfit的系统凭什么和人家竞争，人家凭什么用你的系统。除非是某种护城河使得产品在信心层面有压垮式毁灭性的威慑，让喜欢崩老登的00后望而却步，但这种护城河绝不能只是PR稿和市场营销。嘴上都说尊敬老人，现实生活中看到登气太重的公司和产品谁还不想崩两个。

---

**不是「已經做完了」，也不是 KG／RAG 太淺，必須變成抽象理論才能投 ICLR。更關鍵的是：我們在推進中，把你原來的「從事後經驗學會更好地理解與決策」，逐漸替換成了「反向查詢變換與有限狀態抽象」。**

後者是前者的一種可能實現，但不是唯一實現。這個替換，確實讓研究離你的直覺越來越遠。

我也要修正上一輪的重心：**「可歸因、可重複」是證據標準；「有限預算優勢」只是可能的貢獻，不應成為你只能追求的研究目標。** 我們完全可以尋找更有洞察力的現象，再用嚴格實驗把它站穩。

## 1. 你的 Hindsight，和現在摘要中的 Backward，其實不同

我查到了你的 [Hindsight Preference Optimization 論文](https://arxiv.org/abs/2604.23988)，作者包含你，公開版本也註明被 ICLR 2026 TSALM Workshop 接收。

它的直覺很清楚：**在當時無法充分評價的建議，等結果發生後，可以獲得更好的訓練訊號。** 這與你現在描述的 Worldline 動機是一致的。

但需要分開四件事情：

| 概念 | 用熟悉的語言說 | 與你的研究的關係 |
|---|---|---|
| **Hindsight learning** | 結果出來後，重新評價先前的建議，改善下一次表現 | 你的 HPO |
| **Retrospective inference** | 後來知道的事情，幫助重新理解先前不確定的關係 | 你想要的「後知後覺」與 KG |
| **Backward query regression** | 為了知道最後會不會成功，倒推現在必須滿足什麼條件 | 現在登記摘要的主要內容 |
| **Counterfactual reasoning** | 理解實際發生了什麼，再問換一個行動會怎樣 | Worldline 的多世界決策 |

**你的興趣主要橫跨第一、二、四項，而我們最近的實驗高度集中在第三項。**

這不表示選題錯了，但解釋了為何你會覺得：「明明想研究有用的後見之明，怎麼變成了 bits、partitions、suffix 和 grammar？」

## 2. 成熟的是部分工具，不是整個問題

已有 regression、Bayesian smoothing、state abstraction，並不等於以下問題已經解決：

- 從雜亂的文件與事件中，學到正確的實體關係和行動語義。
- 從一個事後結果，分清哪些解釋得到支持，哪些只是合理故事。
- 把一個場景學到的關係，轉移到新公司、新流程、新行動組合。
- 利用多個可能世界改善決策，同時避免把模擬器的偏見當成真相。

理論通常回答的是：**在指定表示、轉移模型和資訊條件下，可以做什麼。** 你想做的系統，還需要學會取得、組織和使用那些條件。

而且 hindsight 現在仍在產生新工作。例如，2026 年的 [Hindsight Self-Distillation](https://arxiv.org/abs/2606.15576) 使用成功的完整延續來提供中間步驟訊號；[AHEAD](https://arxiv.org/abs/2608.24114) 則區分不同步驟應使用哪種事後指導。這既說明方向沒有「做完」，也提醒我們不能只以「使用未來資訊訓練」作為新意。

你的 workshop 經驗與此並不矛盾：**好的貢獻可以是把既有原理變成一個以前沒有被有效解決的學習問題，不必發明全新的數學宇宙。**

## 3. 我認為最值得追的「情理之中、意料之外」

我會把中心問題改成這一句：

> **事後回饋不應只教模型「哪個答案比較好」，還應教它「當時哪些關係值得記住」。**

再推進一步，可能有一個真正值得測的現象：

> **模型可以因為 hindsight 而更會預測已見情境的結果，卻更不會應對改變條件後的世界——因為它記住了與結果相關、但不可遷移的線索。**

這是**待驗證的假設**，不是我斷言你的 HPO 已經出現這個問題。

### 一個貼近實際的例子

在有明確規則的金融流程模擬中，兩筆付款都「成功」：

- 一筆具有有效的授權鏈；
- 另一筆授權不成立，但檢查環節恰好失效而放行。

最終 outcome 一樣，值得學習的關係卻不同。

只看成功結果，甚至看一段事後合理化的解釋，不一定能學到可靠的決策規則。當你換一個付款人、改變執行順序、修復檢查環節，差異才會暴露。

**多世界模擬器的價值，就不只是多生成幾條未來，而是提供「哪些關係改變後，結論就應該不同」的對照。**

這樣就能把幾條主線接起來：

- **HPO**：利用事後資訊產生學習訊號。
- **KG**：保存誰與誰、在什麼條件下有什麼關係。
- **Backward reasoning**：從未來查詢的差異，追溯需要保留的關係。
- **Worldline**：提供有明確語義的替代情境。
- **RAG**：在下一個問題出現時，找回真正影響結論的證據。

它比「反向搜尋快一些」更接近你要的研究故事。

但不能跳過兩個限制：

1. 如果當時的證據根本不足以區分兩種機制，模型不能靠 hindsight 訓練憑空知道答案；應保留不確定性。
2. LLM 編出的另一個故事，不自動成為有效的 counterfactual。初期仍需要受控流程、可執行動作或可核對的事件紀錄。

### 怎樣讓它超越一句漂亮話？

最關鍵的比較不是只對 outcome-only baseline，而是：

- 使用事後結果訓練；
- 使用同樣豐富的完整軌跡／分支資訊訓練；
- 使用這些資訊，**透過 backward refinement 改變關係記憶**。

如果第三種在未見行動組合中更可靠，而且不是因為它看到了更多答案，才建立了方法貢獻。

**可歸因與可重複仍然重要，但這次它們服務的是一個科學發現，而不是把論文縮成成本比較。**

## 4. KG／RAG 並不淺：我們需要把「顯微鏡」接回場景

直接的反例就有：

- **ToG 2.0，ICLR 2025**：核心是圖與文件檢索如何互相促進，不是純理論。[正式論文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/830b1abc6d2da85f23d41169fa44d185-Abstract-Conference.html)
- **G-reasoner，ICLR 2026**：把圖結構與文字語義結合，用於知識推理。[正式論文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/67e2da8af2476c7441c881fc253042a6-Abstract-Conference.html)

所以問題不是「金融 RAG 不夠高級」，而是能否提出超越單次產品改善的、可遷移的發現。

我們之前為了排除干擾，把問題縮成固定 signature 和有限動作，這作為診斷是合理的；但如果後續只圍繞這個簡化物件證明性質，就可能把研究工具當成了研究目的。

我現在建議：

**受控關係環境負責確認機制；文件連結的流程圖／事件圖負責確認實際價值。**

不必直接從五位元世界跳到完整聯儲分析，也不必永遠停留在五位元世界。

## 5. Graph／action／query 一定要固定類別嗎？不需要

你提出的 **graph encoder＋text** 完全可以成立。要區分不同層次的開放性：

- **新實體、新圖大小**：可以用共享 graph encoder 處理，不必為每個公司或人物預先建立固定分類頭。
- **新關係名稱**：可以用文字描述編碼。但「新名稱／同義詞」與「全新關係語義」是不同難度。
- **新動作組合**：可以把已知操作組合成新序列。
- **全新動作**：需要描述、示例或執行語義。給模型一個新動詞，並不能唯一決定它如何改變世界。
- **新 query**：可以由文字或結構化表達式輸入；同一個 probability head 回答不同 query，不等於只有固定的幾種問題。
- **輸出**：可以是概率、數值分布、圖更新、證據子圖或文字，不必都是固定 outcome classes。

尤其要注意：

**對每個查詢輸出「真／假概率」，不代表整個世界只有兩種 outcome。查詢本身可以變。**

目前的固定 alphabet 是實驗設計，不是 world model 的必然限制。不過，你登記的 **fixed grammar＋executable tests** 是實際方法承諾；如果全部改成自由文字生成，就會變成另一種研究。

### 小型 foundation model 可以嗎？

可以，而且已有很具體的尺度參照：

- **GFM-RAG** 的圖模型是 **8M 參數**，在多個 KG 上預訓練。
- **G-reasoner** 的圖模型是 **34M 參數**，融合圖與文字資訊。

這些是圖元件大小，不包含整個配套 LLM；GFM-RAG 的訓練資料也不小，涵蓋 60 個 KG。[GFM-RAG](https://arxiv.org/abs/2502.01113)、[G-reasoner](https://arxiv.org/abs/2509.24276)

對你，我更支持：

**重用文字模型，訓練小型關係記憶／圖推理元件，再用 hindsight 改善它保留和使用資訊的方式。**

8×A100 40GB 足以讓這個尺度值得認真考慮；但本次投稿不應突然增加「從頭做大規模跨域預訓練」這個依賴。

長期可以稱為 foundation-model 路線；短期先證明跨圖、跨關係配置、跨動作組合的 transfer。**Foundation 的重點是可重用的能力，不只是模型做過 pretraining。**

## 6. APC／BOWL／OpenVAE／DISC 還有用嗎？

有，但應根據問題分工，而不是四個都裝進來。

| 方法 | 真正有用的位置 | 不應期待它自動做到什麼 |
|---|---|---|
| **APC** | 從部分證據推斷概率表示；讓多個可能解釋共存；可作局部 hindsight teacher 的候選 | 不會自動理解任意圖／文字語義，也不保證概率已校準 |
| **BOWL** | 判斷新資料是否陌生、是否值得學，並支援持續更新 | 原方法利用 BatchNorm；不能把任意 embedding 擬合成 Gaussian 就當完整 BOWL |
| **OpenVAE** | 模型持續更新時，保留舊的有效知識與識別未知輸入 | 若只做一次離線訓練，replay 不一定需要 |
| **DISC** | 區分模型遇到的是哪類變化，而不只是報一個異常分數 | 原方法的 diffusion trajectory 不等於你的 action trajectory，不能直接混用 |

來源：[APC](https://arxiv.org/html/2507.04385v2)、[BOWL](https://proceedings.mlr.press/v330/kamath26a.html)、[OpenVAE 對應工作](https://arxiv.org/abs/1905.12019)、[DISC](https://arxiv.org/abs/2510.17381)。

其中，我會優先考慮 **APC 的部分證據／多解釋表示**，而不是先加入 replay。

另外，Martin 還有一條現在看來**更直接**的連接：**ConCon，ICML 2025**。它研究隨時間改變的虛假相關，指出持續學習的困難不只是忘記舊知識，還包括學到錯的線索。[論文](https://proceedings.mlr.press/v267/busch25a.html)

這與「hindsight 讓模型記住真正有用的關係，還是強化事後捷徑？」非常貼近。

也需要修正原 proposal 中的一種隱含期待：**PC 比 Gaussian 更有表達能力，不代表 PC likelihood 必然更能判斷回答可靠性。** 這仍然要測，不能預設為「升級後一定更安全」。

## 我的最新建議

我不會把你的研究收縮為「如何更便宜地搜尋 predicates」，也不建議再開一個與現有投稿脫節的 HPO 2.0。

我最支持的核心是：

> **Learning Relational Memory from Diverging Futures**  
> 利用事後經驗與不同未來的對照，學會哪些關係必須被記住，才能在新的情境下正確推理。

對本次投稿，保留原來的 action-conditioned queries、backward refinement 和 executable relational tests；把它們用來回答這個更直觀的問題。Graph＋text 可以作為介面與表示方式，小型預訓練模型則是後續可擴展的承載。

**HPO 教模型「哪個建議值得偏好」；下一步可以教模型「為了在另一個世界仍然做對，必須理解和保留什麼」。** 這是我認為最能連接你的直覺、既有工作、Worldline 和導師研究線的方向——尚未證明新穎或有效，但值得用一個真正貼近場景的實驗去破題。

---

我原來的想法是，backward reasoning比較契合worldline 這個多重世界模擬器，同時用後知後覺來輔助決策和kg的關係比較實用。這個理解促使我選擇了這個題目。但是爲何如此障礙重重，是因爲這個命題理論上依然過於intuitive所以早在過去幾年已經被很多人做完了嗎？真的做完沒有再挖掘的可能了嗎？因爲我今年早前提交過iclr 2026的workshop "Hindsight Preference Optimization for Financial Time Series Advisory" 入選poster,我以爲這是一個非常有趣的新穎命題，所以也增加了選題的信息。而且基於以上討論，似乎實驗和復現還有領域都進入了極其抽象的地步，而非我所認識熟悉的知識圖譜和rag方面，是因爲某些論述過於抽象？還是說因爲kg本身或者kg foundation model過於粗淺，所以爲了iclr 所以思路走向越來越理論，完全脫離一般實際和intuition？我想到好的論文一般都會有靈感乍現的“情理之中意料之外”，我們可能往這個方向破題嗎，而不是走向一個“可歸因、可重複的有限預算優勢  ”？以及，current graph + action(s) -> predict outcome classes . graph/action/outcome(query) 是不是一定假設為固定的types/classes, 還是可以pretrain 一個graph encoder+text token，像是Multi-modal. 還是說我們可以pretrain 小型的foundation ，model？最後，你上面期待很高的ramesh/martin 方法如apc/bowl/openvae/disc還有用嗎

--- 

**我的建議是：保留已登記的研究問題，但立即縮窄貢獻主張。不要再把「backward reasoning＋關係抽象＋概率模型」這個整體組合當成新意。**

你現在的困難有實質原因，不是讀得不夠。我也需要修正上一輪建議：**我把「與 ATLAS／導師主線吻合」判斷得比較清楚，但對「具體方法能否成立為新貢獻」收斂得太早。兩者不是同一回事。**

以下以你貼出的正式摘要和截圖為依據；GHES 的原始程式、結果與證明，我沒有獨立核驗。

## 1. 新意卡在哪裡：真正的對手不是 ToG、TRIX 或 Flock

你的摘要，實際上進入了 **generalized planning、predicate invention、state abstraction、automata learning** 的交叉區。這輪找到幾個非常接近的前例：

| 你摘要中的核心內容 | 已有的直接前例 |
|---|---|
| 把未來問題逐步反推為當前條件 | Boutilier 等人的 **first-order decision-theoretic regression**，已用反向運算構造與未來價值相關的關係區分。[原文](https://www.cs.toronto.edu/~cebly/Papers/dtregress-abs.html) |
| 從轉移樣本、固定 grammar 學習特徵與抽象動作 | **Learning Features and Abstract Actions for Computing Generalized Plans**，AAAI 2019。[論文](https://ojs.aaai.org/index.php/AAAI/article/view/4120) |
| 用下游任務目標挑選關係 predicates | **Predicate Invention for Bilevel Planning**，AAAI 2023，已對 grammar 中的 predicate sets 做目標導向搜尋。[論文](https://arxiv.org/abs/2203.09634) |
| 不重建整個世界，只保存回答問題所需的資訊 | **Value Equivalence／Proper Value Equivalence**。[VE](https://arxiv.org/abs/2011.03506)、[PVE](https://arxiv.org/abs/2106.10316) |
| 用 suffix 區分狀態，同時保留參數綁定 | **Scalable Tree-based Register Automata Learning**，TACAS 2024，連縮短 suffix、限制相關資料依賴以降低查詢成本都已有研究。[論文](https://link.springer.com/chapter/10.1007/978-3-031-57249-4_5) |
| world model 遇到反例後發明 predicates、修正表示 | 2026 年 **Continual learning and refinement of causal models through dynamic predicate invention** 已有 predict–verify–refine 方法。[方法全文](https://arxiv.org/html/2602.17217v1) |

最後一篇主要處理**確定性、完全可觀測**環境，因此不等於解決了你的概率查詢問題。但它足以否定「world model＋predicate invention＋refinement 本身就是新貢獻」的說法。

這些前例**沒有證明你的方法與它們等價**；它們說明，你必須找出一個更具體的算法差異。僅換名稱或加入 probabilistic circuit，不足以回答 reviewer。

## 2. 你的結果沒有否定原題，但否定了一條容易走偏的路

截圖中的固定表示實驗：

| 方法 | Exposed U MSE ↓ | Exposed V MSE ↓ |
|---|---:|---:|
| NLL | 0.089039 | 0.123109 |
| Workload | 0.107906 | 0.167181 |
| NLL＋Workload | 0.089775 | 0.124681 |

這輪結果不支持「換成 future-query loss 就更好」。而且它是已暴露的 development cohort、三個初始化，不是三次獨立資料實驗。

但你的正式摘要承諾的是：

> **學習新的 executable relational tests，改變表示本身。**

固定五位元表示後比較 loss，沒有測到這一點。

用直觀的例子說：如果表示只記得「有人持有資產」，沒有記得「誰持有哪項資產」，下游模型再怎樣訓練，也不能可靠回答某個指定人的轉讓操作是否成功。

因此：

- 五位元表示的限制，可以作為診斷。
- 256／512 類的計數結果，若證明成立，可以說明特定表示不夠。
- **它們不能直接證明 backward construction 比既有表示學習更好。**

同樣，TRIX／Flock 復現成功是實質進展，但屬於靜態關係預測；shared-suffix compiler 的數值一致性證明實現正確，也不是新學習機制的證據。

**現在欠缺的不是更多復現，而是一次真正改變表示的對照。**

## 3. 我最支持的收窄：反向訊號能否降低「找到好表示」的成本？

推薦把核心問題改成：

> **在相同資料、grammar、監督訊號與計算預算下，反向查詢引導是否能更有效地構造關係測試，使模型泛化到未見的動作組合？**

這其實就是你截圖最後的 **suffix-guided prefix construction vs completed-candidate search**。我不是建議你再開第六個方向，而是把這個尚未完成的比較提升為唯一主線。

它的優點是：

- 與已登記摘要高度一致。
- 不必聲稱發明了 regression 或 query closure。
- 不必證明比所有 NLL 模型更有表達能力。
- 可以靠**可歸因、可重複的有限預算優勢**建立貢獻。

但要注意：register-automata learning 也研究搜尋／查詢效率。因此，你還必須說清楚**關係 grammar、由資料學出的轉移模型，以及概率查詢目標，具體令算法多解決了什麼問題**。不能只是把既有技巧搬來後宣稱首次。

### 最小的決定性對照

保留四個核心組：

1. **One-step refinement**：根據一步轉移錯誤增加特徵。
2. **普通 grammar search**：使用相同 future-query 目標，搜尋完整候選特徵；不能只設一個很弱的 greedy baseline。
3. **你的 backward-guided construction**：利用訓練 suffix 和變量綁定引導部分表達式的擴展。
4. **相同監督的連續表示模型**：讓 relational encoder 自己學表示，檢驗離散測試構造是否真的有價值。

最重要的控制是：

**把不同方法找到的特徵集合，交給同一個 transition learner、同一個 objective 重新擬合。**

如果差異仍然存在，才比較能歸因於「找到的表示」。若更換 compiler、額外教師答案、不同 refit 次數就解釋了提升，那不是你想主張的機制。

最值得看的主圖是：

> **未見組合的 query error，對總 construction cost 的曲線。**

成本包括候選評估、teacher calls、refits 和快取，不只最後的模型大小。你不一定需要更高的最終極限性能；在合理範圍內，以更少成本找到同等有效的表示，也可能形成清楚的算法貢獻。

### 必須拆清的一個問題

**action semantics 到底是已知的，還是從經驗學的？**

- 若已知完整規則，反向條件可以由符號 regression 得到，必須比較這個強基線。
- 若由 circuit 從有限轉移資料學出，circuit 的查詢答案可能錯。其內部推理精確，不等於對真實環境的答案精確。

這個區別會決定你是在做 **symbolic compilation、表示搜尋，還是從有限經驗學 world model**。目前摘要把三者放得太近，容易讓貢獻邊界模糊。

## 4. 題目與摘要應怎樣改？

**現在不急著換題名。** 原名 *Learning Relational World Models through Backward Reasoning* 可以保留，先改摘要的貢獻重心。

建議做四個調整：

- 把「backward reasoning 是新原理」改為「backward guidance 是待驗證的表示構造策略」。
- 把 closure 理論放在問題定義與分析工具的位置；除非有超越既有結果的定理，不把它列為主要創新。
- 把 probabilistic circuit 定位為共享的條件查詢／轉移建模元件，單獨消融其作用。
- 把成功標準從籠統的 compositional transfer，改成**匹配成本下的表示構造效率與未見組合誤差**。

可採用的研究定位句是：

> We investigate whether backward-guided construction discovers executable relational tests more efficiently than equally supervised feature search and one-step refinement, enabling prediction on unfamiliar action compositions.

這句不替尚未完成的實驗預告勝利，但把你需要證明的東西說得很清楚。

### 備選：改成 characterization paper

如果構造算法沒有優勢，但你能得到有普遍性的結論，可以考慮：

**When Is Relational Memory Sufficient for Compositional Prediction?**

核心是區分：

- 表示丟失必要資訊；
- transition model 估計不準；
- 未見組合造成的泛化失敗。

這與 ATLAS 很吻合，但**目前一個固定編碼上的負結果，加上標準計數論證，還不足以支撐這條路**。需要跨模型、跨環境的可重複發現，而不是把失敗實驗換一個標題。

此外，官方明確要求修改後仍接近原摘要。因此 characterization 也應保留原來的 relational transition/query 任務與 backward refinement 評估，不能換成另一篇 RAG 或金融 foundation-model 論文。[ICLR 2027 規則](https://iclr.cc/Conferences/2027/AuthorGuidelines)

## 5. 剩下幾天，我會如何決策

全文截止是 **9 月 26 日北京／香港時間 19:59**。現在應採取一次有期限的研究判斷，而不是繼續擴展方法清單。[官方時間](https://iclr.cc/Conferences/2027/AuthorGuidelines)

- **今天**：寫清四組算法唯一差異，凍結可用資訊與新測試集；跑完一個真正 end-to-end 的小型 construction pilot。
- **明天**：得到匹配預算的核心比較。先判斷是否值得擴大，不把 pilot 當最終顯著性證據。
- **24 日**：若機制有區分度，擴到新世界／第二種有意義的動態條件；若沒有，判斷是否有足夠強的 characterization 發現。
- **25–26 日**：完成主張與結果對應、消融、論文和提交檢查。

暫停增加靜態 KG 復現、更多固定表示 loss、額外 backbone，以及與這個比較無關的性能優化。你的算力應投入**能改變研究判斷的實驗**。

我的最終判斷是：

**這個研究問題值得繼續，但目前還不是一個已找到新意、只差跑結果的 ICLR 方法題。最值得搶救的是「有限資源下如何學到必要的關係區分」，不是「反向推理優於前向建模」這個大敘事。**

它仍然能接上 Ramesh／Martin 的表示學習、概率推理和受控性能刻畫，也能成為 Worldline 的狀態抽象層；但導師主線是研究動機，不能替代方法證據。如果這個最小對照也無法建立區分，我不建議為了已登記而把成熟理論重新包裝成突破。

我已將文獻對照、實驗設計、停止條件及給導師的英文說明整理成：

[9 月 22 日：已登記論文的新意診斷與推進方案](/home/axin/dev/atlas/docs/research/2026-09-22-registered-paper-novelty-triage.md)
