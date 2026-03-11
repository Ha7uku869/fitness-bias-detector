"""エビデンスベースの筋トレ知識データ。

論文やレビュー記事の知見を構造化して格納する。
各エントリは category（カテゴリ）、text（知識の内容）、source（出典）を持つ。

--- 技術解説: 知識ベースの設計 ---
RAG で使う知識は「検索しやすい粒度」に分割する必要がある。
1エントリ = 1つのトピックに絞ることで、検索精度が上がる。
長い論文をそのまま1エントリにすると、検索時にノイズが混入しやすい。

--- 出典について ---
知識はスポーツ科学の主要なメタ分析・レビュー論文に基づく。
個人の経験則ではなく、複数の研究を統合した知見を優先している。
"""

KNOWLEDGE_ENTRIES: list[dict[str, str]] = [
    # === トレーニングボリュームと頻度 ===
    {
        "category": "トレーニングボリューム",
        "text": (
            "[training volume, sets, reps, hypertrophy] "
            "筋肥大には週あたりのトータルボリューム（セット数×レップ数×重量）が重要。"
            "1部位あたり週10〜20セットが推奨される。"
            "週10セット未満では効果が減少し、週20セット以上では回復が追いつかない可能性がある。"
        ),
        "source": "Schoenfeld et al., 2017 - Dose-response relationship between weekly resistance training volume and increases in muscle mass",
    },
    {
        "category": "トレーニング頻度",
        "text": (
            "[training frequency, muscle protein synthesis, split routine] "
            "同じ部位を週2回以上トレーニングすると、週1回よりも筋肥大効果が高い。"
            "これは筋タンパク質合成（MPS）が刺激後24〜48時間でピークに達するため。"
            "週の総ボリュームが同じなら、分割して頻度を上げた方が効果的。"
        ),
        "source": "Schoenfeld et al., 2016 - Effects of Resistance Training Frequency on Measures of Muscle Hypertrophy",
    },
    # === プログレッシブオーバーロード ===
    {
        "category": "プログレッシブオーバーロード",
        "text": (
            "[progressive overload, strength gains, weight increase] "
            "筋力・筋肥大の継続的な向上には漸進的過負荷（プログレッシブオーバーロード）が不可欠。"
            "重量、レップ数、セット数のいずれかを段階的に増やしていく。"
            "毎回重量を増やす必要はなく、レップ数を1回増やすだけでも過負荷になる。"
        ),
        "source": "ACSM Position Stand, 2009 - Progression Models in Resistance Training for Healthy Adults",
    },
    {
        "category": "停滞期の対処",
        "text": (
            "[plateau, stall, deload, stuck, not improving, strength plateau] "
            "筋力の停滞（プラトー）は正常な適応反応であり、才能がないことを意味しない。"
            "対策として: (1)ディロード週（負荷を40〜60%に落とす回復期間）を設ける、"
            "(2)トレーニング変数（種目、レップレンジ、テンポ）を変更する、"
            "(3)睡眠・栄養を見直す。停滞は成長の通過点。"
        ),
        "source": "Pritchard et al., 2015 - Tapering Practices of Strength Athletes",
    },
    # === 休息とリカバリー ===
    {
        "category": "休息の重要性",
        "text": (
            "[rest day, recovery, overtraining, skip gym] "
            "筋肉はトレーニング中ではなく休息中に成長する。"
            "同じ部位のトレーニング間に48〜72時間の休息を取ることが推奨される。"
            "オーバートレーニングは筋力低下、疲労蓄積、免疫低下を引き起こす。"
            "休息日を取ることはサボりではなく、成長のための戦略的な選択。"
        ),
        "source": "Kreher & Schwartz, 2012 - Overtraining Syndrome: A Practical Guide",
    },
    {
        "category": "睡眠と筋肉",
        "text": (
            "[sleep, growth hormone, testosterone, muscle recovery] "
            "睡眠中に成長ホルモンが分泌され、筋肉の修復と成長が促進される。"
            "睡眠不足（6時間未満）はテストステロンの低下や筋タンパク質合成の減少を招く。"
            "7〜9時間の睡眠が筋トレの効果を最大化するために推奨される。"
        ),
        "source": "Dattilo et al., 2011 - Sleep and muscle recovery",
    },
    # === 栄養 ===
    {
        "category": "タンパク質摂取",
        "text": (
            "[protein intake, nutrition, diet, muscle building] "
            "筋肥大を目的とする場合、体重1kgあたり1.6〜2.2gのタンパク質摂取が推奨される。"
            "一度に吸収できるタンパク質量に上限はあるが（約40g程度）、"
            "余剰分も時間をかけて利用されるため、1食の量を過度に気にする必要はない。"
            "タンパク質を3〜4回の食事に分散させるのが理想的。"
        ),
        "source": "Morton et al., 2018 - A systematic review of protein requirements for muscle mass",
    },
    {
        "category": "カロリーと体組成",
        "text": (
            "[calories, body recomposition, bulking, cutting, fat loss] "
            "筋肉を増やしながら脂肪を減らす（ボディリコンポジション）は、"
            "初心者やトレーニング再開者では可能だが、上級者では難しい。"
            "増量期（カロリー余剰）と減量期（カロリー不足）を分けるのが効率的。"
            "減量中も十分なタンパク質摂取と筋トレを継続することで筋量の維持が可能。"
        ),
        "source": "Barakat et al., 2020 - Body Recomposition: Can Trained Individuals Build Muscle and Lose Fat at the Same Time?",
    },
    # === フォームとケガ予防 ===
    {
        "category": "フォームの重要性",
        "text": (
            "[form, technique, range of motion, injury prevention] "
            "正しいフォームは怪我の予防と対象筋への効果的な刺激の両方に重要。"
            "ただし「完璧なフォーム」に固執しすぎると重量の進歩を妨げることがある。"
            "多少のフォームの崩れは許容範囲で、重要なのは安全な可動域の確保。"
            "フォームが大きく崩れる重量は扱うべきではない。"
        ),
        "source": "Schoenfeld & Grgic, 2020 - Evidence-Based Guidelines for Resistance Training",
    },
    {
        "category": "ケガへの対応",
        "text": (
            "[DOMS, soreness, pain, injury, muscle damage] "
            "軽度の筋肉痛（DOMS）は正常な反応であり、トレーニングの効果の指標ではない。"
            "筋肉痛がなくても筋肥大は起きる。筋肉痛が強すぎる場合はボリュームの増やしすぎ。"
            "関節の痛みや鋭い痛みは筋肉痛とは異なり、休息や医療相談が必要。"
        ),
        "source": "Schoenfeld & Contreras, 2013 - Is DOMS an accurate indicator of muscle damage?",
    },
    # === メンタルと継続性 ===
    {
        "category": "モチベーション",
        "text": (
            "[motivation, habit, consistency, lazy, skip, quit] "
            "モチベーションは波があるのが普通。やる気がない日もトレーニングを続けるには、"
            "「習慣化」が鍵。週3回など決まった日に行く仕組みを作る。"
            "完璧を求めず、「今日は軽くやるだけでもOK」というマインドセットが継続に繋がる。"
            "0か100かではなく、50%の日があっても全く問題ない。"
        ),
        "source": "Lally et al., 2010 - How are habits formed: Modelling habit formation in the real world",
    },
    {
        "category": "他者との比較",
        "text": (
            "[comparison, genetics, social media, beginner gains] "
            "SNSで見る筋トレ上級者と自分を比較するのは非生産的。"
            "筋肉の成長速度は遺伝（筋繊維組成、テストステロンレベル等）に大きく影響される。"
            "比較すべきは過去の自分のみ。1ヶ月前の自分より進歩していれば順調。"
            "初心者は最初の1年で最も大きな伸びが期待でき、その後は徐々に緩やかになる（対数的成長）。"
        ),
        "source": "Ericsson et al., 1993 - The Role of Deliberate Practice; Hubal et al., 2005 - Variability in muscle size and strength gain",
    },
    # === ベンチプレス特化 ===
    {
        "category": "ベンチプレス",
        "text": (
            "[bench press, chest, plateau, strength, improve bench] "
            "ベンチプレスの重量を伸ばすには: "
            "(1)週2回以上の頻度でベンチプレスを行う、"
            "(2)5×5や3×3などの低レップ高重量セットを組み込む、"
            "(3)補助種目（ナローベンチ、ディップス、ショルダープレス）で弱点を補強する、"
            "(4)パウズベンチ（胸の上で一時停止）でボトムの強さを鍛える。"
        ),
        "source": "Greg Nuckols, Stronger By Science - How to Bench Press",
    },
    # === スクワット特化 ===
    {
        "category": "スクワット",
        "text": (
            "[squat, legs, depth, full range, plateau, improve squat] "
            "スクワットの深さはフルレンジ（大腿が水平以下）が筋肥大に最も効果的。"
            "ただし個人の柔軟性に応じて深さを調整すべき。"
            "重量が停滞した場合はフロントスクワットやポーズスクワットなどの"
            "バリエーションを取り入れることで弱点を克服できる。"
        ),
        "source": "Schoenfeld, 2010 - Squatting kinematics and kinetics; Hartmann et al., 2012 - Analysis of the load on the knee joint",
    },
    # === デッドリフト特化 ===
    {
        "category": "デッドリフト",
        "text": (
            "[deadlift, back, conventional, sumo, lower back, belt] "
            "デッドリフトは全身の筋力を最も効率的に鍛えられる種目の一つ。"
            "コンベンショナルとスモウの両スタイルがあり、体格に応じて選ぶべき。"
            "腰を丸めないことが最重要で、ベルトの使用は高重量時に推奨される。"
            "週の総ボリュームを管理し、回復に十分な時間を取ること。"
        ),
        "source": "Cholewicki et al., 1991 - Lumbar spine loads during the deadlift",
    },
]
