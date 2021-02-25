# Machine Learning

[[日本語メモ](<Memo(JA).md>)]
[[English](<README.md>)]


Udemyのコース、([Machine Learning A-Z: Hands-On Python & R In Data Science](https://www.udemy.com/course/machinelearning/))で作ったファイルを管理するための個人的なレポジトリ。CSVや画像データ等は削除済み。

## 概要

- Association Rule Learning
  - Apriori
  - Eclat
- Classification
  - Logisitic Regression
  - K-NN
  - SVM
  - Kernel SVM
  - Naive Bayes
  - Decision Trees
  - Random Forest
- Clustering 
  - K-Means
  - Hierarchical Clustering
- Deep Learning (Keras with TensorFlow backend)
  - ANN
  - CNN 
- Dimensionality Reduction
  - PCA
  - LDA
  - Kernel PCA
- Model Selection & Boosting
  - Model Selection
  - XGBoost
- Natural Language Processing
  - NLP
- Regression
  - Simple Linear Regression
  - Multiple Linear Regression
  - Polynomial Regression
  - SVR
  - Decision Tree (Regression)
  - Random Forest (Regression)
- Reinforcement Learning
  - UCB
  - Thompson Sampling

## notes

## Association Rule Learning
### Apriori
### Eclat
## Classification
### Logisitic Regression

<a href="https://www.codecogs.com/eqnedit.php?latex=y&space;=&space;b_{0}&space;&plus;&space;b_{1}*x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y&space;=&space;b_{0}&space;&plus;&space;b_{1}*x" title="y = b_{0} + b_{1}*x" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=p&space;=&space;\frac{1}{1&space;&plus;&space;e^{-y}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p&space;=&space;\frac{1}{1&space;&plus;&space;e^{-y}}" title="p = \frac{1}{1 + e^{-y}}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=ln(\frac{p}{1-p})&space;=&space;b_{0}&space;&plus;&space;b_{1}*x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?ln(\frac{p}{1-p})&space;=&space;b_{0}&space;&plus;&space;b_{1}*x" title="ln(\frac{p}{1-p}) = b_{0} + b_{1}*x" /></a>

### K-NN
Data-pointに近いneighborを見て分類を行う手法。周りに"染まる"ようなイメージ。

**アルゴリズム**
1. neighbor数K(ハイパラ)を選択
2. 新しいデータを取り込む
3. For each データ
    3.1 他のデータとの距離を測定
    3.2 距離を配列等に収納
4. 距離を昇順に整理
5. 上位K個のデータを選択
6. 選択されたデータのラベルを取得
7. 分類の場合はラベルの最頻値を返す（回帰の場合は、ラベルの平均を返す）

**ハイパラの調整**
- Kの値をいじってエラーが最小なもの選ぶ
- Kの値が小さければ小さいほど予測が不安定になる

**利点**
- 構造がシンプル
- モデル構築の必要がない
- ハイパラが少ない
- 汎用性が高い

**欠点**
- データ数が大きくなるにつれ計算コストが上がる：O(n)

### SVM
分類の境界線に近い極端な例を見る手法。データを高次元に移すことでデータを分けるhyperplaneを探す。hyperplaneに強い影響を及ぼすデータポイントをsupport vectorと呼ぶ。

**アルゴリズム**
1. (低次元の)n次元のデータでスタートする
2. データをn + 1次元に移す(射影する)
3. そこでデータを上手く分けるSupport Vector Classifier^[1] を探す
    3.1 Polynomial Kernel　ハイパラの次数をCrossValidation等で探し当てる
    3.2 Radial Kernel infinite次元においてSVCを探す(実際はKernelTrickというもので次元数を削減して計算している)

^[1] (missclassificationを許す)Soft Marginを用いたthresholdのこと。例）2次元データのSVCは線、3次元データのSVCは平面、n次元データのSVCはn-1次元のsubspace(hyperplane)

**ハイパラ**
- Kernelの種類
- 普通はRBF(Radial Basis Function)

**利点**
- 高次元データに対しても使える
- support vectorだけを使うのでメモリを比較的食わない

**欠点**
- データ数が大きいと計算時間が長くなる
- ノイズが多い場合や2クラス間の境界線がはっきりとしていない場合は精度が落ちる

### Kernel SVM
Kernel functionを使うSVM

### Naive Bayes
ベイズの定理を用いた手法。分類クラスが3つ以上あった場合でも、確率は合計すれば必ず1になる。

**アルゴリズム**
年齢、居住地域を説明変数、徒歩出勤か車出勤かを被説明変数とした例を用いる

1. 新しいデータ、年齢x1、居住地域x2を読み込む
2. x1,x2の元でその人が徒歩出勤する確率を求める
3. x1,x2の元でその人が車出勤する確率を求める
4. 2つの確率を比較し、大きい方を選択する

ベイズの定理：
事後確率 = 尤度 x 事前確率 / 周辺尤度

step 2の具体的な計算方法
事前確率 P(徒歩) = 徒歩の人の人数 / データ数
周辺尤度 P(x1,x2) = 似た人の人数 / データ数
尤度 P(x1,x2 | 徒歩) = 似た人の内、徒歩の人人数 / 徒歩の人の人数

*事後確率の比較の際、分母はどの道消せるので周辺尤度は計算しなくてもいい

**ハイパラ**
- _

**利点**
- 計算速度は早い
- 高次元データでも使える

**欠点**
- 本来、説明変数が互いに独立であることを前提としている（ただし独立でなくとも高精度が出ることもある）
- 速度と引き換えに精度が犠牲になりやすい（世の中の説明変数が互いに独立であることは稀なため）

### Decision Trees
木構造の分類器。

**ハイパラ**
- 木の深さ

**利点**
- 可読性が高い
- 数値型、分類型のデータ両方扱える
- 推論速度が速い
- 多重共線性の影響が低い
- 外れ値の影響を受けづらい

**欠点**
- 過学習しやすい
- プルーニングする必要性がある
- 他手法に比べ精度が低い


### Random Forest
小さな木の集合体。アンサンブルラーニングの1種。XGBoost,LightGBM等の元。

**ハイパラ**
- 木の深さ

**利点**
- 他手法に比べ精度が高い
- 過学習しづらい
- 外れ値の影響を受けづらい
- 大きいデータセットでも

**欠点**
- 可読性が低い
- 訓練速度が遅い

## Clustering 
### K-Means
**アルゴリズム**
1. (elbow method等で)クラスター数Kを設定
2. K個のcentroidをランダムに選択
3. 各データポイントを最も近いcentroidに配属させ、K個のクラスターを作る
4. 各クラスターの重心に、新しいcentroidを設定する
5. step3~4を繰り返す。centroidが動かなくなり、データポイントの所属が変わらなくなったら終了。

**ハイパラ**
- クラスター数K

**利点**
- 導入が簡単
- 必ず収束する
- global optimumを目指す（全クラスターの分散を最小化しようとするため）

**欠点**
- Random Initialization Trapに注意。各クラスターの配置は初期値(初期のcentroid)に影響される。
- 外れ値に影響されやすい
- （素K-meansでは）分散やサイズの異なるクラスターをクラスタリングする際に苦戦する。(→次元ごとにクラスターごとの"幅"を変更することで改善可能)

### Hierarchical Clustering
名の通りhierarchicalなデータに対して使うのが一番。

**アルゴリズム(Agglomerative HC)**
1. 各データポイントを1つのクラスターとして定義する。（つまりデータ数n個分のクラスターができる。）
2. 最も近い2つのクラスターを融合し、新たなクラスターを作る。（n-1個のクラスターができる。）
3. step-2をクラスターが1つになるまで繰り返す。

クラスター間の距離について：
- 別々のクラスター中、最も近い2点
- 別々のクラスター中、最も遠い2点
- クラスターの点同士の平均距離
- 重心間の距離

**ハイパラ**
- 距離の定義
- 距離の計算手法(ユークリッド、マンハッタン)
- ＋αの手法

**利点**
- デンドログラムが生成でき、視覚的にクラスタリングの経緯がわかりやすい
- Ward法など外れ値に強いアルゴリズムもある(ただし計算過程で距離が歪む)

**欠点**
- greedy algorithmなので最適解ではないかもしれない
- 


## Deep Learning (Keras with TensorFlow backend)
### ANN
- 分野が広いので別途記述
### CNN 
- 分野が広いので別途記述

## Dimensionality Reduction

### PCA
教師なしの次元削減アルゴリズム
主な用途：
- ノイズのフィルタリング
- 可視化
- 特徴量の抽出
- 株価予測（←？）

d次元のデータをk次元に射影することで次元を減らす。

**アルゴリズム**
1. データを標準化する(平均0、分散1にする)
2. 共分散行列もしくは相関行列の固有値もしくは固有ベクトルを取得。または特異値分解を行う
3. 降順に固有値を整理し、k個の固有値に対応するk個の固有ベクトルを選択する。この際kは射影先次元数。
4. 選択したk個の固有ベクトルから射影行列Wをつくる
5. 元のデータセットXを射影行列Wで変換し、k次元のfeature subspaceを取得する

**利点**

**欠点**

### LDA
教師ありの次元削減アルゴリズム

**アルゴリズム**

**利点**

**欠点**

### Kernel PCA
非線形なPCAメソッド。

## Model Selection & Boosting
### Model Selection
### XGBoost
## Natural Language Processing
### NLP
## Regression
### Simple Linear Regression
### Multiple Linear Regression
### Polynomial Regression
### SVR
### Decision Tree (Regression)
### Random Forest (Regression)
## Reinforcement Learning
### UCB
### Thompson Sampling

