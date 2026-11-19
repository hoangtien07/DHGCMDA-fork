# THESIS_REPRO_LIMITATIONS_VI — 再現可能性の限界（論文用・日本語）

## 論文で主張できること（証明済み）
1. **評価メトリクス定義が最大のブロッカー**（WP1 で新規証明）: 公開ソースから導ける全ての評価規約（K5 スカラー truth の first/last/任意割当、pooled/fold-wise、multi-hot one-pred、K4 フィルタ）において、paper の P=0.7915 / R=0.9421 の組は厳密な上界（0.758〜0.917）を超える → 「再現失敗」ではなく「評価規約が公開情報から一意に定まらない」と記述すべき。
2. **リリース評価器は v3.2 を採点できない**（K4 評価器が K5 行を全てドロップ）— 著者コードリリースの欠陥。
3. **データセットは統計上一致**: 411×271×5、11,748 links、density 10.55% — paper 記述と一致（SPLD MDAv3.2-3 由来、L1 同一性）。したがって gap の主因はデータではなく evaluator + fold/feature 定義。
4. **独自実測の妥当な範囲**: U1 top1 harmonic 0.456–0.492、prior ≈0.45–0.53、Track N A1 は補正済み最良ベースラインに −0.014（棄却）。これらは「我々の解釈」ラベルの値であり paper 値と直接比較しない。

## 主張してはいけないこと
- paper の誤り・捏造 — 証明不能。スカラー truth の定義・フィルタ・micro 化・別集団の可能性が残る。
- 数値的に近い F1 が出る convention を「paper 相当」と呼ぶこと。
- author artifact 未検証のまま byte 一致を主張すること。

## 図表に使える証拠
- CONVENTION_MATRIX（上界 vs published R のギャップ表）
- FIDELITY_TRACE（predictor/loss/graph の adaptation 分類表）
- leakage の分離実験（v2.0 +0.049 / v3.2 ≈0 の paired contrast）
