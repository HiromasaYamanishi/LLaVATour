from easydict import EasyDict
from utils import pref_regions


prompt_spot = "このレビューがポジティブかネガティブか教えてください。\n\
    また、このレビューが述べるこの観光地のポジティブな部分、ネガティブな部分を要約されたキーワードで出力してください。\n\
    例1:\n\
    レビュー: 深夜に女性専用ラウンジを利用しました。リクライニングソファがあり寝る事が出来ます。静かで雰囲気は良かったけど暖房が効きすぎているのか暑くて辛かったです。\n\
    出力:\n\
        ポジティブな点: 「リクライニングソファで寝ることができる」「静かで雰囲気が良い」\n\
        ネガティブな点: 「暖房が効きすぎていて暑い」\n\
    例2:\n\
    レビュー: 函館山からの夜景を楽しみに行ったのに2回とも雲がかかり上からの景色は見ることが出来なかった。ロープウェイで下降する時少し見えただけでした。\n\
    出力:\n\
        ポジティブな点: なし\n\
        ネガティブな点: 「雲で夜景が見えなかった」\n\
    例3:\n\
    レビュー: 平日の昼過ぎに訪れました。駅からの道のりは少しわかりにくいですが迷うことなく到着。すぐに鳳凰堂内部見学のため観覧券販売所を見つけ行列の後ろに。１回に５０人程度が入れるのかな。並んだ時に次の時間を買うことができました。鳳凰堂の中に入ってビックリしたのはライトアップされた阿弥陀如来坐像の煌びやかさとたくさんの菩薩像です。２０分程度の説明があっという間でした。仏像を前に身が引き締まる思いです\n\
    出力:\n\
        ポジティブな点: 「ライトアップされた阿弥陀如来坐像の煌びやかさ」「たくさんの菩薩像」「説明があっというまに感じるほど面白い」\n\
        ネガティブな点: 「駅からの道が分かりにくい」\n\
それでは次のレビューに対して同様に出力してください。\n\
レビュー: {}。"

prompt_food = "このレビューがポジティブかネガティブか教えてください。\n\
    また、このレビューが述べるこの観光地のポジティブな部分、ネガティブな部分を要約されたキーワードで出力してください。\n\
    例1:\n\
    レビュー: おいしいせんべい,瓦せんべいで人気の和菓子店です。いろいろな大きさがあって、たくさんお土産に買って帰りました。癖になる美味しさです。\n\
    出力:\n\
        ポジティブな点: 「瓦せんべいで人気」「いろいろなお土産がある」「癖になる美味しさのせんべい」\n\
        ネガティブな点: なし\n\
    例2:\n\
    レビュー: お店の方のかけ声と音楽でとっても楽しかったです！すごく混んでました、予約必須です。うどん作りしてうどん食べてお土産もらって大満足です\n\
    出力:\n\
        ポジティブな点: 「お店の方のかけ声と音楽が楽しかった」「うどん食べてお土産もらって満足して」\n\
        ネガティブな点: 「すごく混んでいた」「予約必須」\n\
    例3:\n\
    レビュー: こしがしっかりあるのが好きな方にはもの足りないかな？サイドメニューは豊富でした。味と値段も納得でした\n\
    出力:\n\
        ポジティブな点: 「サイドメニューが豊富」「納得の値段と味」\n\
        ネガティブな点: 「こしが足りない」\n\
それでは次のレビューに対して同様に出力してください。\n\
レビュー: {}。"

prompt_hotel = "このレビューがポジティブかネガティブか教えてください。\n\
    また、このレビューが述べるこの観光地のポジティブな部分、ネガティブな部分を要約されたキーワードで出力してください。\n\
    例1:\n\
    レビュー:思ったより歓楽街より遠かったです。第一目的の場所は近かったのですが…。部屋もお風呂も朝食も価格対で考えると満足のいくものでした。が、コンビニくらいは近くに欲しいかなぁ。なんとなくロビーの雰囲気が寂しかったです\n\
    出力:\n\
        ポジティブな点: 「部屋もお風呂も朝食も価格対で満足」\n\
        ネガティブな点: 「思ったより歓楽街より遠い」「コンビニが近くにない」「ロビーの雰囲気が寂しい」\n\
    例2:\n\
    レビュー:一泊二食付きプランを利用させて頂きましたが、夕食を食べに行くのに車で１０分ほど掛かるのと、朝食の量が少しもの足りませんでした。良かったところは、ホテルのスポーツジムを利用出来たところです。また、利用したく思ってます。\n\
    出力:\n\
        ポジティブな点: 「ホテルのスポーツジムを利用出来た」\n\
        ネガティブな点: 「夕食を食べに行くのに車で10分ほど掛かる」「朝食の量が少し物足りない」\n\
    例3:\n\
    レビュー:フロントの方の対応がとてもにこやかで室内も気持ちよく、泊まってよかったねと話していました。朝のバイキングも内容が豊富でゆっくり食事ができて満足です。\n\
    出力:\n\
        ポジティブな点: 「フロントの方の対応がにこやか」「バイキングの内容が豊富」「朝食をゆっくり食事ができる」\n\
        ネガティブな点: なし\n\
それでは次のレビューに対して同様に出力してください。\n\
レビュー: {}。"

prompt_summarize_general = (
    "短文から多数派のトピックのみを抜き出し, 一文程度でまとめてください.次は例です.\n\
例1:\n\
入力:\n\
    石段がきつくてたいへんです。\n\
    混雑しており前へ進まず。階段をひたすら上がらなければならない。\n\
    階段数が多い。\n\
    800段を超える石段は少し疲れます。\n\
    こんぴらさんの階段はきつくて足にきます。\n\
    道のりが長く、きつい。\n\
    観光化されている。\n\
    この角を曲がったらそこかな。階段を登らないと行けない。\n\
    子供連れにはあまりオススメできません\n\
出力:\n\
    800段を超える階段がキツくて大変\n\
例2:\n\
入力:\n\
    混雑する時間帯 \n\
    混雑する可能性がある\n\
    混雑\n\
    渋滞が発生する\n\
    賑やかだった\n\
    GWで混んでいた\n\
    人がいっぱい\n\
    人が多すぎた\n\
    人が多かった\n\
出力:\n\
    GWや時間によっては混雑する\n\
例3:\n\
入力:\n\
    猿の子連れがいてチヨットびっくり\n\
    山頂で大勢のサルが寝転がっている\n\
    ウリ坊が出現したり、展望台で猿の大群が移動したりして嬉しいサプライズがあった\n\
    表は猿が多く、道には猿の糞が多く踏まないよう注意する必要がある\n\
    野生の猿がいて間近で見ることが出来ました\n\
    山の上にあり、野生の猿がいる\n\
    途中の道路\n\
出力:\n\
    ウリや猿など野生の動物を見ることができる\n\
\n\
それでは次の入力に対して同様に出力を行ってください\n"
)

prompt_summarize_detail = (
    "短文から多数派のトピックのみを選択し, 50文字から100文字でまとめてください. 次のように2段階で行ってください.\n\
例:\n\
第一段階では, まず多数派のトピックとそれに対応する文章を選びます.\n\
入力:\n\
\n\
    癒しの場所。\n\
    のどかな雰囲気. 広い空間でのんびり過ごせる\n\
    解放された建物\n\
    自然に溶け込む非日常の空間。\n\
    リフレッシュされる。\n\
    周囲から隔離されて落ち着いた雰囲気。\n\
    中国人の留学生と来た。\n\
    マイナスイオンで心や体も癒されます\n\
    ベンチがあり, 休憩できる\n\
    都会の喧騒を忘れることができる\n\
    日常の嫌なことを忘れています\n\
\n\
ここで多数派のトピックは「リラックス・リフレッシュできる」ということです。\n\
第一段階の出力は, 多数派のトピックと対応するトピックで\n\
\n\
    癒しの場所。\n\
    のどかな雰囲気. 広い空間でのんびり過ごせる\n\
    自然に溶け込む非日常の空間。\n\
    リフレッシュされる。\n\
    周囲から隔離されて落ち着いた雰囲気。\n\
    マイナスイオンで心や体も癒されます\n\
    ベンチがあり, 休憩できる\n\
    都会の喧騒を忘れることができる\n\
    日常の嫌なことを忘れています\n\
\n\
第二段階では第一段階の出力を複数の文で要約します.例の第一段階の短文の集合を要約すると\n\
\n\
「周囲から隔離されたリラックスできる雰囲気の場所です. 自然に溶け込む非日常の空間の中で都会の喧騒や日常を忘れ、リフレッシュすることができます.」\n\
\n\
となります.よって出力は\n\
\n\
出力:\n\
「周囲から隔離されたリラックスできる雰囲気の場所です. 自然に溶け込む非日常の空間の中で都会の喧騒や日常を忘れ、リフレッシュすることができます.」です\n\
\n\
それでは次の入力の短文の集合に対して同様に第一段階, 第二段階の処理を行い, 第二段階の次の最終的な要約された文章のみを出力してください.第一段階の処理結果は出力しないでください.\n\
入力:\n\
\n"
)

prompt_summarize_simple = "次に与える短文の集合から多数派のトピックに該当する短文の内容をつなぎ合わせて100文字程度で要約してください.\n"

prompt_refine = (
    "次の文章から内容の重複や繰り返しを取り除き，日本語として簡潔で自然な文章に変えてください.また，多数の文章からなる場合は読みやすくなるように要約してください．次は例です\n\
例1:\n\
入力:\n\
    「優しく丁寧に教えていただき、十分に楽しむことができました。楽しみ方のコツも教えていただき、大満足です。」<|im_end|>\n\
    丁寧な説明と指導のおかげで、のんびりと楽しい時間を過ごすことができました。<|im_end|>\n\
出力:\n\
    丁寧で優しい説明と指導によりのんびりと楽しめました．コツも教えていただき大満足です\n\
例2:\n\
入力:\n\
現代美術館では、斬新なデザインや趣のある建物が展示されており、モダンな空気が演出されています。モダンな作品が豊富に展示されており、こじんまりとした雰囲気も漂っています。<|im_end|>\n\
 芸術祭でたくさんの人が訪れて、様々な作品が展示されていました。展示品はアート性が高く、五感を刺激する一風変わった作品が多かった。また、ここでしか見られない資料や作品も展示されていました。展示作品は豊富で、これからの作品の数々が楽しみになるような場所でした。<|im_end|>\n\
出力:\n\
    現代美術館では、斬新なデザインや趣のある建物が展示されており、モダンな空気が演出されています．芸術祭では多くの人が訪れ，モダンな作品やアート性の高い，五感を刺激する一風変わった作品など豊富な作品が展示されています．\n\
それでは次の入力に対して同様に出力を行ってください\n\
\n\
入力:\n\
{review}"
)

prompt_negative = "次の文章がポジティブかネガティブか判定してください．次は例です\n\
例1:\n\
入力:\n\
    人気のスポットで混雑する場所にあります。\n\
出力:\n\
    ネガティブ\n\
例2:\n\
入力:\n\
    古民家風の展示物や外観が古民家であるなど、古民家を生かした独特な雰囲気が楽しめます。\n\
出力:\n\
    ポジティブ\n\
例3:\n\
入力:\n\
    猛暑日や暑い時期は注意が必要です。\n\
出力:\n\
    ネガティブ\n\
例4:\n\
入力:\n\
    水遊びができる場所がたくさんある夏です。\n\
出力:\n\
    ポジティブ\n\
それでは次の入力に対して同様に出力を行ってください\n\
{review}\n"

prompt_direct_step1 = (
    "これからレビューを与えます。これからポジティブな要素とネガティブな要素を抽出して要約してください.似たトピックはマージしてください.\n\
例1:\n\
入力:\n\
    ８００段を越える石段で有名なこんぴらさんですが、実は、幼稚園にも満たないお子さんでも参拝できます。奥宮までは、１３００段以上のなりますが、３つの女の子がお父さんと手をつないで楽しい層に登ってきたのが印象的でした。\n\
    当日雨予報でしたが現地着いた頃には上がっていたので傘を杖代わりに嫁と共に幸福の黄色御守りゲットの為参拝。帰りは雨が本降りになってきて足元かなり滑るのでまあまあ大変でした。\n\
    参道の階段を700数段上がったところの本宮は迫力で、展望テラス？もすごく見通しがよく遠くまで見えました。そこからさらに600数段上がったところに奥宮があり、奥宮限定のお守りが買えるということで頑張って奥宮まで登りました。奥宮は小さなお宮でしたが風情がありました。\n\
    久しぶりの金比羅さんでした。夕方4時前に宿に着き、御朱印を貰いたくて急いで登り16時半少し前に着き、御朱印をいただく事が出来ましたが汗だくでした。本宮も奥社も16時半までですが奥社は朝も9時からなので、気をつけて予定を立てて下さい。\n\
    少し風が冷たく、肌寒い感じはしましたが、階段を登りだせば汗だくになりました。途中、焼き立てのお煎餅をつまみ食い、とっても美味しかったです！\n\
    前回6月にお参りしましたが、暑くて本宮までしか行けませんでした。登っても登っても、先が見えず、かなりしんどかったです。天気は良く、ところどころ紅葉も見れました。\n\
    沢山の階段をお土産屋さんを見ながら登りようやくお守りを購入出来ました。帰りの際お土産をゆっくり見たり足湯も満喫できました。\n\
    大変ではありましたが行った達成感があるので行って下さいそこでしか買えない御守りや御朱印も頂けます途中の神椿でのパフェも美味しいので食べて良かったです\n\
    雨上がりに初めてお参りに行きました。靴底のデザインがツルツル系でしたが滑ってしまい派手に転倒してしまいました。階段が急な所も有り、靴の選択と濡れている場合は十分な注意が必要です。\n\
出力:\n\
    ポジティブな点:\n\
        ・焼きたてのお煎餅や神椿でのパフェが美味しい\n\
        ・階段を登った本宮の迫力がある\n\
        ・展望テラスからの見晴らしが良い\n\
        ・限定のお守りが買える\n\
        ・足湯が満喫できる\n\
    ネガティブな点:\n\
        ・雨が降ると滑って危険\n\
        ・階段が長くてしんどい\n\
        ・奥社の空いている時間に気をつける必要がある\n\
それでは次の入力に対して同様に出力を行ってください\n\
入力:\n"
)

prompt_direct_aspect = (
    "与えるレビューの一覧に対して・値段、・サービス(案内、接客、ガイド)、・飲食、・設備、・季節、・人気のあるトピック。についてそれぞれ抽出し要約してください\n\
次のような箇条書きの形式で出力を行なってください。\n\
・値段:\n\
・サービス: \n\
・飲食: \n\
・設備: \n\
・季節: \n\
・トピック: \n\
それでは次の入力に対して出力を行ってください\n\
入力:\n"
)

prompt_direct_aspect2 = (
    "与えるレビューの一覧に対して・値段、・サービス(案内、接客、ガイド)、・飲食、・設備、・季節、・人気のあるトピック、についてそれぞれ抽出し要約してください\n\
また、これまでの要約も与えるのでこれまでの要約結果と抽出した結果を結合して新しい結果を生成してください\n\
以下のように箇条書きの形式でそれぞれ要約文を出力してください。\n\
・値段:\n\
・サービス: \n\
・飲食: \n\
・設備: \n\
・季節: \n\
・トピック: \n\
それでは次の入力に対して同様に出力を行ってください\n\
入力:\n"
)

prompt_direct_step2 = (
    "これから施設の要約されたポジティブな文章の一覧とネガティブな文章の一覧を複数与えます.これらを要約して, 再び要約したポジティブの文章とネガティブな文章を出力してください.\n\
ただし、同じようなトピックを統合したり頻度の少ないトピックを削除し、再び要約したポジティブな文章とネガティブな文章はそれぞれ多くても10個程度にしてください.\n\
例1:\n\
入力:\n\
    要約1:\n\
    ポジティブな点:\n\
        手入れが行き届いた素敵な庭園\n\
        時間があればお食事やお茶を楽しめます\n\
        ライトアップも見ることができました\n\
    ネガティブな点:\n\
        時間がないと楽しめない\n\
    要約2:\n\
    ポジティブな点:\n\
        夜桜ライトアップが美しかった\n\
    ネガティブな点:\n\
        駐車料金が高い\n\
        人が多い\n\
    要約3:\n\
    ポジティブな点:\n\
        公園内に印象に残る撮影スポットがたくさんある\n\
        日本三名園より優れていると言われている\n\
        お抹茶を飲みながら眺める庭が素晴らしい\n\
    ネガティブな点:\n\
        混雑している\n\
        駐車場が混雑している\n\
        庭園が広く、歩くのに時間がかかる\n\
出力:\n\
    ポジティブな点:\n\
        手入れが行き届いた素敵な庭園\n\
        お食事やお茶を楽しめます\n\
        撮影スポットが多い\n\
        夜桜のライトアップが綺麗\n\
    ネガティブな点:\n\
        駐車場の料金の高さと混雑\n\
        園内の混雑\n\
        園内が広く回るのに時間がかかる\n\
\n\
それでは次の要約群を要約して，ポジティブな文章とネガティブな文章を要約してください\n\
入力:\n"
)

prompt_qa = "レビューから質問と回答のペアを作成してください。次は例です。\n\
例1.\n\
入力: 北海道旅行にいったときに夜景を見に行きました。展望台には観光バスで上るかロープウェイをつかって上ります。展望台から見る景色はとてもきらきらしていてとても綺麗です。絶対に夜にいくことをオススメします。\n\
出力: 「展望台にはどのように登れますか」「観光バスかロープウェイを使って登ります」\n\
例2.\n\
入力: 着いてからは人の山で流れもせず見られる場所を探すのに必死でした…。夕方から夜に変わる夕暮れで景色は最高でした！北海道の5月はやっぱりまだ冷んやりで夜は温かい格好がいいです。\n\
出力: 「レビューは何に満足しましたか」「夕方から夜に変わる夕暮れの景色」\n\
それでは次のレビューに対して同様に質問と回答のペアを作成してください。\n\
早朝に行くと空気が澄んでおり最高の景色を見ることができる。夜景とは違った表情もある。しかも人がいないのでゆっくり景色を堪能できる。\n"

prompt_topic_summary_neg = (
    "これから複数の文章を与えるので重複のあるものはマージするなどして3-7個前後のトピックに要約してください。文章の量に合わせて調節してください。\n\
通常のトピックの他トピックとして次の要素が含まれていればトピックとして作って欲しいです. ・値段、・サービス(案内、接客、ガイド)、・飲食、・設備、・季節ごとのイベントや特徴、\n\
またそれぞれの要約したトピックに対して具体例も含められるものは最大3個程度まで含めてください。具体的な要約例は次です\n\
出力例:\n\
1. トピック\n\
・具体例\n\
・具体例\n\
2. トピック\n\
・具体例\n\
・具体例\n\
3. トピック\n\
・具体例\n\
・具体例\n\
...\n\
\n\
それでは次の入力する文章に対して処理を行なってください。"
)

prompt_topic_summary = (
    "これから複数の文章を与えるので重複のあるものはマージするなどして7個前後のトピックに要約してください。文章の量に合わせて調節してください。\n\
通常のトピックの他トピックとして次の要素が含まれていればトピックとして作って欲しいです. ・値段、・サービス(案内、接客、ガイド)、・飲食、・設備、・季節ごとのイベントや特徴、\n\
またそれぞれの要約したトピックに対して具体例も含められるものは最大3個程度まで含めてください。具体的な要約例は次です\n\
出力例:\n\
1. トピック\n\
・具体例\n\
・具体例\n\
2. トピック\n\
・具体例\n\
・具体例\n\
3. トピック\n\
・具体例\n\
・具体例\n\
4. トピック\n\
・具体例\n\
・具体例\n\
5. トピック\n\
・具体例\n\
・具体例\n\
...\n\
\n\
それでは次の入力する文章に対して処理を行なってください。"
)

prompt_topic_summary__ = (
    "これから複数の文章を与えるので重複のあるものはマージするなどして7個前後のトピックに要約してください。文章の量に合わせて調節してください。\n\
通常のトピックの他トピックとして次の要素が含まれていればトピックとして作って欲しいです. ・値段、・サービス(案内、接客、ガイド)、・飲食、・設備、・季節ごとのイベントや特徴、\n\
またそれぞれの要約したトピックに対して具体例も含められるものは最大3個程度まで含めてください。具体的な要約例は次です\n\
出力例:\n\
1. アクセスと交通手段\n\
・駅から近く、公共交通機関が便利\n\
・停留所が観光スポットにあるため便利\n\
2. 展示物と歴史的な価値\n\
・展示品の豊富さと歴史的な学び歴史を感じることができる\n\
・武将や忍者の活躍を感じる展示\n\
3....\n\
4....\n\
5....\n\
6....\n\
7....\n\
\n\
それでは次の入力する文章に対して処理を行なってください。"
)


prompt_topic_summary_ = (
    "これから複数の文章を与えるので頻度の少ないものは削除する、重複のあるものはマージするなどして7個前後のトピックに要約してください。文章の量に合わせて調節してください。\n\
またそれぞれの要約したトピックに対して具体例も含められるものは最大3個程度まで含めてください。具体的な要約例は次です\n\
出力例:\n\
1. アクセスと交通手段\n\
・駅から近く、公共交通機関が便利\n\
・停留所が観光スポットにあるため便利\n\
2. 展示物と歴史的な価値\n\
・展示品の豊富さと歴史的な学び歴史を感じることができる\n\
・武将や忍者の活躍を感じる展示\n\
3. 歴史と展示物の魅力\n\
・展示品があり歴史の勉強にもなる\n\
・武将や忍者の活躍を感じる展示\n\
4. 名古屋城周辺の楽しみ\n\
・城の下には出店があり楽しい\n\
・みたらし団子が美味しい\n\
5. 名古屋城の食事と特産品\n\
・名物のきしめんやうどん屋さんがあり食事も楽しめます\n\
・売店の夏向きの商品\n\
……\n\
\n\
それでは次の入力する文章に対して処理を行なってください。"
)

prompt_review_summary_aspect = (
    "次にユーザーの訪問した全ての場所とそこで書いたレビューを渡します。ユーザーについて次の観点からプロファイルしてください。ただしそれぞれ箇条書きで出力してください\n\
・文体の特徴\n\
・ユーザの興味、関心があるトピック\n\
・ユーザーがポジティブに感じやすいこと\n\
・ユーザーがネガティブに感じやすいこと\n\
\n\
{}\n\
"
)

prompt_common_pvqa = (
    "5つの短文から過半数の文章に共通するトピックを抜き出し単語で抽出してください.次は例です\n\
例1:\n\
入力:\n\
文1:入園料も安く大満足でした！レッサーパンダが超可愛かった！\n\
文2:レッサーパンダが空中のはしごを歩く姿にも癒されます～\n\
文3:レッサーパンダが、頭上にいたり動物を間近でみる工夫が沢山されている動物園です\n\
文4:お子様連れで冬に行く場合は、気をつけられた方が良いかと思います\n\
文5:ただ、残念なのは字が小さいので読みにくい\n\
出力:\n\
レッサーパンダ\n\
それでは次の入力に対して同様に出力してください\n\
入力:\n"
)

prompt_summary_for_pvqa = (
    "次の文章からまず、それぞれの文章からキーワード 「{}」に対する観光地の事実を簡潔に抜き出してください。ただし、キーワードに関する事実の情報のみを含め、その他の観光地の情報や観光客の特有の情報、観光客の意見や感想は含めないでください。\n\
    次のステップとして、それらの抽出された事実を日本語として自然な形でまとまった文章で要約してください。箇条書きにはしないでください\n\
それでは、次の文章をまとめてください. 抽出の結果は出力せずに、最終的な要約の結果のみを出力してください。また、「要約すると」などの前置きはつけないでください\n"
)

prompt_summary_for_pvqa_ = (
    "次の文章からキーワード 「{}」に対する観光地の事実を簡潔に抜き出し、日本語として自然な形でまとまった文章で要約してください。ただし、キーワードに関する事実の情報のみを含め、その他の観光地の情報や観光客の特有の情報、観光客の意見や感想は含めないでください。箇条書きにはしないでください\n\
それでは、次の文章をまとめてください"
)


prompt_review_summary_general = (
    "次にユーザーの訪問した全ての場所とそこで書いたレビューを渡します。これらのレビューについて100文字から500文字程度で要約してください.\n\
{}\n"
)

prompt_fos_extract_old = (
    "please extract (feature, opinion, sentiment) pair from review please only output (feature, opinion, sentiment) pairs. sentiment can be 1 (positive) or 0 (neutral) or -1 (negative).\n\
These are some examples:\n\
Example1:\n\
Input:\n\
    Amazing lamb and great portion siZes.  Feta fries were great melted.  Cant wait to try baklava ice cream and rice pudding\n\
Output:\n\
    (place, awesome, +1), (size, small, -1), (atmosphere, loud, -1), (ambiance, bustling, +1), (availability of seats, not always easy to get a seat, -1), (food, interesting good, +1)\n\
Example2:\n\
Input:\n\
    Compare to the other Mexican restaurants in the dowtown area, this is one of the two places we like to go. Their food is unique. Great outdoor bar/patio. Service at bar varies. We ordered the pan seared red snapper. The head and bones were for display which could have been be edible chip if it was not a salt lick. \n\
Output:\n\
    (food, unique, +1), (outdoor bar/patio, great, +1), (service at bar, varies, 0), (pan seared red snapper display, could have been edible chip, -1),(the head and bones of pan seared red snapper, salt lick, -1)\n\
Please output similarly for following review:\n\
Input:\n\
    {}\n\
"
)

prompt_fos_extract = (
    "please extract (feature, opinion, sentiment) pair from the review. Please only output (feature, opinion, sentiment) pairs. sentiment can be 1 (positive) or 0 (neutral) or -1 (negative).\n\
\n\
Example:\n\
Input:\n\
    Amazing lamb and great portion sizes. Feta fries were great melted. Can not wait to try baklava ice cream and rice pudding.\n\
Output:\n\
    (lamb, amazing, +1), (portion sizes, great, +1), (feta fries, great melted, +1), (baklava ice cream, can not wait to try, +1), (rice pudding, can not wait to try, +1)\n\
\n\
Input:\n\
    Compare to the other Mexican restaurants in the downtown area, this is one of the two places we like to go. Their food is unique. Great outdoor bar/patio. Service at bar varies. We ordered the pan seared red snapper. The head and bones were for display which could have been edible chip if it was not a salt lick.\n\
Output:\n\
    (food, unique, +1), (outdoor bar/patio, great, +1), (service at bar, varies, 0), (pan seared red snapper display, could have been edible chip, -1), (the head and bones of pan seared red snapper, salt lick, -1)\n\
\n\
Please output similarly for the following review:\n\
\n\
Input:\n\
    {}\n\
Output:"
)


prompt_acos_yelp = (
    "please extract (category, feature, opinion, sentiment) pair from the review. Please only output (category, feature, opinion, sentiment) pairs. \n\
category should be in (Food Type, Food Quality, Service, Value, Atmosphere, Location, Specific Food, Convenience). feature is the detail of category. sentiment can be 1 (positive) or 0 (neutral) or -1 (negative).\n\
\n\
Example1:\n\
Input:\n\
    Country omelet, hash browns, stack were all very good.  The Bliody Mary was excellent,  huge piece of bacon! Amazing service, clean & fun/funky atmosphere.  Very reasonably priced . In the heart of neat Boulder city shops.   Read the rules- hysterical\n\
Output:\n\
    (Specific Food, Country omelet, very good, 1), (Specific Food, hash browns, very good, 1), (Specific Food, stack, very good, 1), (Specific Food, Bloody Mary, excellent, 1), (Specific Food, bacon, huge piece, 1), (Service, service, Amazing, 1), (Atmosphere, atmosphere, clean & fun/funky, 1), (Value, price, very reasonably priced, 1), (Location, Boulder city shops, In the heart of neat, 1)\n\
Example2:\n\
Input:\n\
    Compare to the other Mexican restaurants in the downtown area, this is one of the two places we like to go. Their food is unique. Great outdoor bar/patio. Service at bar varies. We ordered the pan seared red snapper. The head and bones were for display which could have been edible chip if it was not a salt lick.\n\
Output:\n\
    (Location, downtown area, Compare to the other Mexican restaurants, 0), (Food Quality, food, unique, +1), (Atmosphere, outdoor bar/patio, Great, +1), (Service, bar, varies, 0), (Specific Food, pan seared red snapper display, could have been edible chip, -1), (Specific Food, the head and bones of pan seared red snapper, salt lick, -1)\n\
\n\
Please output similarly for the following review:\n\
\n\
Input:\n\
    {}\n\
Output:\n\
"
)

prompt_pvqa = EasyDict({
"prompt1" : '{image_suffix}に関して{exp_suffix}してください',
"prompt2" : '{image_suffix}に関して「{keywords}」というキーワードを用いて{exp_suffix}してください',
"prompt3" : '{image_suffix}に関して「{reviews}」という文章を参照にして{exp_suffix}してください',
"prompt4" : "この{image_suffix}は{spot}の{image_suffix}です。{image_suffix}に関して{exp_suffix}してください",
"prompt5" : 'この{image_suffix}は{spot}の「{keyword}」に関する{image_suffix}です。{image_suffix}に関して{exp_suffix}してください',
"prompt6" : 'この{image_suffix}は{spot}の{image_suffix}です. この{image_suffix}に関する{exp_suffix}を「{keywords}」というキーワードを用いて{do_suffix}してください',
"prompt7" : 'この{image_suffix}は{spot}の「{keyword}」に関する{image_suffix}です。この{image_suffix}に関する{exp_suffix}を「{keywords}」というキーワードを用いて{do_suffix}してください',
"prompt8" : 'この{image_suffix}は{spot}に関する{image_suffix}です. この{image_suffix}に関する{exp_suffix}を「{reviews}」というレビューを元に{do_suffix}してください',
"prompt9" : 'この{image_suffix}は{spot}の「{keyword}」に関する{image_suffix}です. この{image_suffix}に関する{exp_suffix}を「{reviews}」というレビューを元に{do_suffix}してください'
})

prompt_acos_beauty = (
    "please extract (category, feature, opinion, sentiment) pair from the review. Please only output (category, feature, opinion, sentiment) pairs. \n\
category should be in (Specific Items, Product Performance, Usability, Packaging and Design, Price, Customer Experience, Customer Service, Durability and Reliability, Scent and Fragrance). feature is the detail of category. sentiment can be 1 (positive) or 0 (neutral) or -1 (negative).\n\
\n\
Example1:\n\
Input:\n\
    The smell is so sweet and delicious. I put some drops in my raw Shea butter and it worked great. It is not a very expensive product here and they even sent me a little gift! It was a neat surprise. Will definitely recommend and buy again. Thanks.\n\
Output:\n\
    (Scent and Fragrance, Smell, so sweet and delicious, 1), (Usability, Usage in raw Shea butter, worked great, 1), (Price, Cost, not a very expensive product, 1), (Customer Service, Gift, they even sent me a little gift, 1), (Customer Experience, Recommendation, will definitely recommend and buy again, 1)\n\
Example2:\n\
Input:\n\
    The designs in this set are gorgeous, the full nail images are much bigger--actually a bit big for my nails--and the backing does make them less sharp.  The quality is excellent, and it's a fantastic value.\n\
Output:\n\
    (Packaging and Design, Designs, gorgeous, 1), (Application and Usability, full nail images, a bit big, 0), (Application and Usability, Backing, the backing does make them less sharp, 1), (Product Performance, Quality, the quality is excellent, 1), (Price, Value, it's a fantastic value, 1)\n\
Example3:\n\
Input:\n\
    This polish is the exact shade of gold I was looking for (not too yellow). It applies evenly and dries to be very metallic and glossy. My only complaint is that it chips relatively easily without a top coat.\n\
Output:\n\
    (Product Performance, Shade, I was looking for and not too yellow, 1), (Usability, Application, applies evenly, 1), (Product Performance, polish, dries to be very metallic and glossy, 1), (Durability and Reliability, Chipping, chips relatively easily without a top coat, -1)\n\
\n\
Please output similarly for the following review:\n\
\n\
Input:\n\
    {}\n\
Output:\n\
"
)

prompt_acos_toys = (
    "please extract (category, feature, opinion, sentiment) pair from the review. Please only output (category, feature, opinion, sentiment) pairs. \n\
category should be in (Specific Item, Product Quality, Usability, Educational and Entertainment Value, Safety, Design and Packaging, Age Appropriateness). feature is the detail of category. sentiment can be 1 (positive) or 0 (neutral) or -1 (negative).\n\
\n\
Example1:\n\
Input:\n\
    My 12 year old son/coin enthusiast found these in his stocking on Christmas morning and was estatic. He immediately poured them out on the table and inspected them all. He found some he already had but also some that he said he'd never seen before. Very pleased, especially for the low price\n\
Output:\n\
    (Age Appropriateness, Suitability for 12-year-old, 12-year-old son/coin enthusiast, 1), (Product Quality, Coin Variety, found some he said he'd never seen before, 1), (Product Quality, Inspection, inspected them all, 1), (Specific Item, Coins, very pleased, 1), (Price, Value, especially for the low price, 1)\n\
Example2:\n\
Input:\n\
    Furbys entertained a few generations already and will keep going. It's interactive. Easy to deal with and to take care of. Furby Boom is a lot better than the previous editions. I guess they worked a lot more in making them talk in a way you can understand most of what they say.\n\
Output:\n\
    (Educational and Entertainment Value, Entertainment, entertained a few generations, 1), (Usability, Interactivity, interactive, 1), (Usability, Ease of Care, easy to deal with and take care of, 1), (Specific Item, Furby Boom, better than previous editions, 1), (Product Quality, Speech Improvement, talk in a way you can understand, 1)\n\
Example3:\n\
Input:\n\
    Even on the longest setting, the light and music don't stay on long enough. You have to keep pushing it over and over, and that basically defeats the point. I liked the Soothe and Glow Seahorses way better and so did my baby. That stays on for a long time.\n\
Output:\n\
    (Usability, Duration of Light and Music, don't stay on long enough, -1),(Specific Item, Comparison with Soothe and Glow Seahorses, I liked the Soothe and Glow Seahorses way better, -1),\n\
\n\
Please output similarly for the following review:\n\
\n\
Input:\n\
    {}\n\
Output:\n\
"
)

prompt_acos_sports = (
    "please extract (category, feature, opinion, sentiment) pair from the review. Please only output (category, feature, opinion, sentiment) pairs. \n\
category should be in (Specific Item, Functionality, Build Quality, Design, Versatility, Customer Service, Price, Portability, Comfort). feature is the detail of category. sentiment can be 1 (positive) or 0 (neutral) or -1 (negative).\n\
\n\
Example1:\n\
Input:\n\
    This tool is very compact, sleek, and durable. It also doesn't take up that much room in my pack either. It would be nice if the tools would lock into place when attempting to tighten or loosen a screw. But that is not too big of a deal right now.\n\
Output:\n\
    (Specific Item, Tool, Compact, sleek, and durable, 1), (Functionality, Accessibility, Easy to access, 1), (Build Quality, Durability, Durable, 1), (Design, Sleekness, Sleek, 1), (Portability, Size, Doesn't take up much room in the pack, 1)\n\
Example2:\n\
Input:\n\
    I currently have the Fobus paddle holsters for my handguns. This product is a welcome addition. Very comfortable and easy to access.\n\
Output:\n\
    (Specific Item, Holster, Welcome addition to Fobus paddle holsters, 1), (Functionality, Accessibility, Easy to access, 1), (Comfort, Fit, Very comfortable, 1), (Versatility, Compatibility, Works well with multiple handguns, 1)\n\
Example3:\n\
Input:\n\
    Received my canteen today. I'm happy it fits in my mesh pocket on my pack. I like the shape and feel of the bottle. Amazon shipped it in a timely manner. I'm also happy it is different than all the other round bottles I see. Makes it seem unique.\n\
Output:\n\
    (Specific Item, Canteen, Fits in mesh pocket on pack, 1), (Design, Shape and feel, Unique shape and feel, 1), (Customer Service, Delivery, Amazon shipped it in a timely manner, 1), (Portability, Fit, Fits in my mesh pocket on pack, 1)\n\
\n\
Please output similarly for the following review:\n\
\n\
Input:\n\
    {}\n\
Output:\n\
"
)
prompt_acos_extract = {
    "yelp": prompt_acos_yelp,
    "beauty": prompt_acos_beauty,
    "sports": prompt_acos_sports,
    "toys": prompt_acos_toys,
}

prompt_review_summary_general_en = (
    "Next, I will provide reviews written by the user at all the places they have visited. Please profile the user from the following perspectives. Make sure to output each point as a bullet point:\n\
- Stylistic features of the writing\n\
- Topics of interest to the user\n\
- Things that the user tends to feel positive about\n\
- Things that the user tends to feel negative about\n\
\n\
The output format should include 3 to 7 points for each topic as shown below:\n\
* Stylistic features of the writing\n\
・\n\
・\n\
* Topics of interest to the user\n\
・\n\
・\n\
* Things that the user tends to feel positive about\n\
・\n\
・\n\
* Things that the user tends to feel negative about\n\
・\n\
・\n\
\n\
please summarize the review below\n\
"
)

prompt_posneg_extract_en = (
    "Please extract the positive and negative aspects of this tourist spot mentioned in the review as summarized keywords.\n\
Example 1:\n\
Review: Location : Located in Town and is 5mins walk to Shopping malls (1st avenue/komtar) Which is very convenient. Hotel room : Newly renovated and very clean and comfortable. Pillow is too soft.Breakfast : Very good, lots of local varieties.\n\
    Facilities :  Basic gym and small pool.Free Wifi in Lobby & roomLate afternoon, Newlane street food appears just outside the Hotel where you can eat all the famous street food. No need to find, just at your doorstep! \n\
Output:\n\
    Positie aspects: 'Located in Town and is 5 mins walk to Shopping malls (1st avenue/komtar) which is very convenient', 'Newly renovated and very clean and comfortable (hotel room)', 'Very good breakfast, lots of local varieties', 'Basic gym and small pool', 'Free Wifi in Lobby & room', 'Newlane street food appears just outside the Hotel where you can eat all the famous street food'\n\
    Negative aspects: 'Pillow is too soft'\n\
Example 2:\n\
Review: This hotel is in a really great location in Bellevue, close to restaurants, shops, and great to access waterfront runs.  Room was average size.  Negative, they forgot the wake up call, so I had to forgo breakfast to make my meeting \n\
Output:\n\
    Positie aspects: 'Great location in Bellevue, close to restaurants, shops, and waterfront runs', 'Good service, provided a great running path that was serene and peaceful'\n\
    Negative aspects: 'Room was average size', 'Lack of benefits such as buffet breakfast and complimentary executive lounge access, compared to Marriott', 'Forgot the wake-up call, causing the reviewer to miss breakfast before the meeting'\n\
Now, please output similarly for the following review.\n\
{}"
)

prompt_posneg_extract_simple_en = (
    "Please extract the positive and negative aspects of this hotel mentioned in the review as short sentence. List the following points with consistent bullet points and spacing:\n\
the output form is as follows\n\
Positive aspects:\n\
・\n\
・\n\
Negative aspects:\n\
・\n\
・\n\
Now, please output similarly for the following review.\n\
{}"
)

prompt_aspect_summary_user_en = (
    "For the given list of short sentences, select the majority opinion for each category and summarize to know what type of aspect user like or dislike.\n\
category is Overall, Service, Check-In, Business, Location, Value, Cleanliness, Rooms, Sleep Quality,\n\
\n\
output format is as follows\n\
Overall:\n\
Service:\n\
Check-In:\n\
Business Service:\n\
Location:\n\
Value:\n\
Cleanliness:\n\
Rooms:\n\
Sleep Quality:\n\
Then, provide the output for the following input:\n\
Input:\n\
"
)

prompt_aspect_summary_item_tripadvisor = (
    "For the given list of short sentences, select the majority opinion for each category and summarize to know what good or bad aspects hotel have.\n\
category is Overall, Service, Check-In, Business, Location, Value, Cleanliness, Rooms, Sleep Quality,\n\
Additionally, for each topic, please include up to three specific and representative examples if possible. when there is no corresponding output, please omit it.\n\
\n\
output format is as follows\n\
Overall:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Service:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Check-In:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Business Service:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Location:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Value:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Cleanliness:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Rooms:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Sleep Quality:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Then, provide the output for the following input:\n\
Input:\n\
"
)

prompt_aspect_summary_user_yelp_degree = "For the given list of reviews,\n\
1. Please extract the food category and topics users like.\n\
2. For all reviews, for each aspect in {Food Quality, Service, Value, Atmosphere, Location, Specific Food, Convenience}, choose the degree of user mention from {Never, Rarely, Sometimes, Often, Always} overall.\n\
3. Also, please summarize the concrete features of each aspect users mention briefly.\n\
\n\
Output format:\n\
Food Quality:\n\
    Frequency:\n\
    Summary:\n\
Service:\n\
    Frequency:\n\
    Summary:\n\
Value:\n\
    Frequency:\n\
    Summary:\n\
Atmosphere:\n\
    Frequency:\n\
    Summary:\n\
Location:\n\
    Frequency:\n\
    Summary:\n\
Specific Food:\n\
    Frequency:\n\
    Summary:\n\
Convenience:\n\
    Frequency:\n\
    Summary:\n\
Food Quality:\n\
    Frequency:\n\
    Summary:\n\
Then, provide the output for the following input.\n\
Input:\n\
"
prompt_user_degree = {"yelp": prompt_aspect_summary_user_yelp_degree}


prompt_aspect_summary_user_yelp_rank = "For the given list of short sentences,\n\
1. Select the top three aspects that users like from the following set: {Food Type, Food Quality, Service, Value, Atmosphere, Location, Specific Food, Convenience}, based on frequency.\n\
2. Summarize the reviews for the top three aspects.\n\
\n\
output format is as follows\n\
\n\
Aspects User Concern about: Top1: Top2: Top3:\n\
Summary of overall attitude for Top1 aspect:\n\
Summary of overall attitude for Top2 aspect:\n\
Summary of overall attitude for Top3 aspect:\n\
\n\
Then, provide the output for the following input.\n\
Input:\n\
"
prompt_user_ranks = {"yelp": prompt_aspect_summary_user_yelp_rank}

prompt_aspect_summary_item_yelp = (
    "For the given list of short sentences, select the majority opinion for each category and summarize to know what good or bad aspects hotel have.\n\
category is Overall, Food Quality, Service, Value, Atmosphere, Location, Specific Food, Convenience\n\
Additionally, for each topic, please include up to three specific and representative examples if possible. when there is no corresponding output, please omit it.\n\
\n\
output format is as follows\n\
Overall:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Food Quality:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Service:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Value:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Atmosphere:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Location:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Specific Food:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Convenience:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Then, provide the output for the following input:\n\
Input:\n\
"
)

prompt_aspect_summary_item_beauty = (
    "For the given list of short sentences, select the majority opinion for each category and summarize to know what good or bad aspects hotel have.\n\
category is Overall, Product Quality, Application and Usability, Packaging and Design, Value and Pricing, Customer Experience, Health and Safety, Durability and Reliability\n\
Additionally, for each topic, please include up to three specific and representative examples if possible. when there is no corresponding output, please omit it.\n\
\n\
output format is as follows\n\
Overall:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Product Quality:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Application and Usability:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Packaging and Design:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Value and Pricing:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Customer Experience:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Health and Safety:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Durability and Reliability:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Then, provide the output for the following input:\n\
Input:\n\
"
)

prompt_aspect_summary_item_toys = (
    "For the given list of short sentences, select the majority opinion for each category and summarize to know what good or bad aspects hotel have.\n\
category is Specific Item Category, Product Quality, Usability, Educational and Entertainment Value, Safety, Design and Packaging, Age Appropriateness\n\
Additionally, for each topic, please include up to three specific and representative examples if possible. when there is no corresponding output, please omit it.\n\
\n\
output format is as follows\n\
Overall:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Product Quality:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Usability:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Educational and Entertainment Value:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Design:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Compatibility and Versatility:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Safety:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Performance:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Price and Packaging:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Then, provide the output for the following input:\n\
Input:\n\
"
)

prompt_aspect_summary_item_sports = (
    "For the given list of short sentences, select the majority opinion for each category and summarize to know what good or bad aspects hotel have.\n\
category is Overall, Functionality, Build Quality, Design, Versatility, Customer Service, Price, Safety, Portability\n\
Additionally, for each topic, please include up to three specific and representative examples if possible. when there is no corresponding output, please omit it.\n\
\n\
output format is as follows\n\
Overall:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Functionality:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Build Quality:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Design:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Versatility:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Customer Service:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Price:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Safety:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Portability:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Then, provide the output for the following input:\n\
Input:\n\
"
)

item_prompts = {
    "trip_advisor": prompt_aspect_summary_item_tripadvisor,
    "yelp": prompt_aspect_summary_item_yelp,
    "sports": prompt_aspect_summary_item_sports,
    "beauty": prompt_aspect_summary_item_beauty,
    "toys": prompt_aspect_summary_item_toys,
}

prompt_pseudo_review = (
    "次にユーザーが過去に書いたレビューとアイテムのプロファイルを渡すので, ユーザーがアイテムに対して書くレビューを予測して生成してください.\n\
ユーザーの過去のレビュー:\n\
{}\n\
アイテムのプロファイル:\n\
{}"
)

prompt_user_merge = (
    "You are provided with multiple profiles of a single user, each containing detailed information on their stylistic features of writing, topics of interest, positive feelings, and negative feelings. \n\
Your task is to merge and summarize these profiles into one cohesive and concise summary for each topic. \n\
Each summarized profile should contain less than five examples per topic. the output format should be same as input format and as follows\n\
* Stylistic features of the writing\n\
・\n\
・\n\
* Topics of interest to the user\n\
・\n\
・\n\
* Things that the user tends to feel positive about\n\
・\n\
・\n\
* Things that the user tends to feel negative about\n\
・\n\
・\n\
\n\
Below are the input profiles."
)


prompt_item_merge_yelp = (
    "You are provided with multiple summaries of reviews for a particular place, containing detailed information on overall experience, food quality, service, value, atmosphere, location, specific food items, and convenience. \n\
Your task is to merge and summarize these summaries into one cohesive and concise summary for each topic. \n\
Each summarized profile should contain less than five examples per topic. Below are the input profiles. the output format should be same as input format and as follows\n\
Overall:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Food Quality:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Service:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Value:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Atmosphere:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Location:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Specific Food:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Convenience:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Then, provide the output for the following input:\n\
Input:\n\
"
)

prompt_item_merge_beauty = (
    "You are provided with multiple summaries of reviews for a particular place, containing detailed information on overall experience, food quality, service, value, atmosphere, location, specific food items, and convenience. \n\
Your task is to merge and summarize these summaries into one cohesive and concise summary for each topic. \n\
Each summarized profile should contain less than five examples per topic. Below are the input profiles. the output format should be same as input format and as follows\n\
Overall:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Product Quality:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Application and Usability:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Packaging and Design:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Value and Pricing:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Customer Experience:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Health and Safety:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Durability and Reliability:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Then, provide the output for the following input:\n\
Input:\n\\n\
"
)

prompt_item_merge_sports = (
    "You are provided with multiple summaries of reviews for a particular place, containing detailed information on overall experience, food quality, service, value, atmosphere, location, specific food items, and convenience. \n\
Your task is to merge and summarize these summaries into one cohesive and concise summary for each topic. \n\
Each summarized profile should contain less than five examples per topic. Below are the input profiles. the output format should be same as input format and as follows\n\
Overall:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Functionality:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Build Quality:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Design:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Versatility:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Customer Service:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Price:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Safety:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Portability:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Then, provide the output for the following input:\n\
Input:\n\
"
)

prompt_item_merge_toys = (
    "You are provided with multiple summaries of reviews for a particular place, containing detailed information on overall experience, food quality, service, value, atmosphere, location, specific food items, and convenience. \n\
Your task is to merge and summarize these summaries into one cohesive and concise summary for each topic. \n\
Each summarized profile should contain less than five examples per topic. Below are the input profiles. the output format should be same as input format and as follows\n\
Overall:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Product Quality:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Usability:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Educational and Entertainment Value:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Design:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Compatibility and Versatility:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Safety:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Performance:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Price and Packaging:\n\
    Positive:\n\
        ・\n\
    Negative:\n\
        ・\n\
Then, provide the output for the following input:\n\
Input:\n\
"
)

item_merge_prompts = {
    "yelp": prompt_item_merge_yelp,
    "sports": prompt_item_merge_sports,
    "beauty": prompt_item_merge_beauty,
    "toys": prompt_item_merge_toys,
}


prompt_fos_japanese = "レビューから(feature, opinion, sentiment)の組み合わせを抽出してください。出力は(feature, opinion, sentiment)の組みのみで、sentimentは1（ポジティブ）、0（ニュートラル）、-1（ネガティブ）です。\n\
また, featureとopinionはなるべく簡潔にし, 分けられるものは分けてください。また、faetureとopinionは、表記揺れがないように一般的な単語に変換してください。\n\
例：\n\
入力：\n\
    セイウチのショーはとてもお勧めです。吉本のお笑いを見ているようで、爆笑の渦でした。動物園とか水族館のショーをこれまで色々見てきましたが、ここまで笑えたのは初めてです。\n\
出力：\n\
    (セイウチのショー, お勧め, 1), (セイウチのショー, お勧め, 1), (セイウチのショー, 笑える, 1)\n\
入力：\n\
    動くティラノサウルスが大迫力でした！動きも滑らかで、一瞬本物かと錯覚するよう(ﾟДﾟ)コロナウイルスの影響で入館予約制だったんですが、それでも結構混んでます。入館チケットと予約票を見せて入館になるので、チケットは先に引き換えすることをお勧めします。朝イチは混んでます。\n\
出力：\n\
    (ティラノサウルス, 大迫力, 1), (ティラノサウルス, 動きが滑らか, 1), (館内, 混雑, -1),(チケット, 先に引き換え推奨, 0),(朝, 混雑, -1)\n\
入力：\n\
    「方除の大社」として知られているので引越しや旅行の前に祈願に行くと良いです。造園家の中根金作さんが手掛けられたお庭が年中美しく茶室や池があり立派なお庭です。\n\
出力：\n\
     (祈願, おすすめ, 1), (庭, 年中美しい, 1), (庭, 立派, 1)\n\
入力：\n\
    規模は大きく素晴らしいと思いますが、何しろ料金が高い、室堂までの往復だと1人1万近くかかります。雪の大谷を見たかったので室堂まで行きましたが、室堂まで行きましたが、折角高い料金払ったのに強風で見れませんでした。。\n\
出力：\n\
    (規模, 大きい, 1) (規模, 素晴らしい, 1), (料金, 高い, -1), (室堂までの往復, 1人1万近く, -1), (雪の大谷, 強風で見られない, -1)\n\
\n\
以下のレビューに対して同様に出力してください：\n\
入力：\n\
    {}\n\
出力："


prompt_qa_multi = "レビューから観光地に関する情報を抽出し質問と回答のペアを複数作成してください。ユーザー特定の情報や感想は質問、回答に含めないでください。次は例です。\n\
例1.\n\
入力: 潮の満ち引きで、消えたり現れたりする珍しい仏様で、干満時にお参りすると、満潮の時に病苦を洗い流してくれるそうです。場所は、御座漁港近くにあります。「石仏地蔵尊」と書かれた小さくて黄色い案内板が道沿いに出ていました。入口は道が狭いですが、駐車場はきちんとしたものがありました。\n\
出力: \n\
    Q:「仏様の何が珍しいですか」A:「潮の満ち引きで消えたり現れたりします」\n\
    Q:「仏様にはどのようなご利益がありますか」A:「干潮時にお参りすると, 病苦を洗い流してくれます」 \n\
    Q:「場所はどこにありますか」A:「御座漁港近くにあります」 \n\
    Q:「目印はありますか」A:「「石仏地蔵尊」と書かれた小さくて黄色い案内板が道沿いに出ています」 \n\
    Q: 「駐車場はありますか」 A:「入り口は狭いですが, 駐車場はちゃんとしたものがあります」\n\
例2.\n\
入力: 東京出張の空き時間で立ち寄りました。無料だったので、時間つぶしに寄ったつもりでしたが、日本の貨幣が歴史とともに解説されており、非常に興味深く見学しました。\n\
出力: \n\
    Q:「料金はどのくらいですか」A:「無料です」\n\
    Q:「何が見れますか」A:「日本の貨幣が歴史とともに解説されています」\n\
例3.\n\
入力: 紫尾神社お隣にある公共浴場 朝５時から営業しており、神社境内から自噴することから「神の湯」 温度は大変熱く、浴槽ごとに温度が微妙に違う。\n\
出力: \n\
    Q:「どこにありますか」A:「紫尾神社の隣にあります」\n\
    Q:「何時から営業していますか」A:「朝５時から営業しています」\n\
    Q:「何と呼ばれていますか」A:「神社境内から自噴することから「神の湯」」\n\
    Q:「暑さはどれくらいですか」A:「温度は大変熱く、浴槽ごとに温度が微妙に違う。」\n\
それでは次のレビューに対して同様に質問と回答のペアを作成してください。\n\
入力: {input_review}\n\
出力: \n"

def make_marketing_prompt(row):
    review = row['review']
    spot = row['spot']
    age = row['age']
    sex = row['sex']
    tag = row['tag']
    prompt = f"次のレビューとそのレビューを書いた人の情報から, ユーザーが何に満足したのかを考えてください。ユーザーが満足したものを元に、同じセグメントの人を増やすにはどうすれば良いか考えてください\n\
回答フォーマットは[観光地]で[age]の[tag]の[gender]を増やすには「〇〇」が有効だと考えられます。というフォーマットにして\n\
\n\
観光地名: {spot}\n\
ユーザーの属性: {age}、{tag}、{sex}\n\
レビュー:{review}\n\
それでは出力してください"
    return prompt

def make_search_prompt(row):
    review = row['review']
    spot = row['spot']
    pref = row['pref']
    region = pref_regions[pref]
    prompt = f"次の観光地のレビューを「観光地を探すためのクエリープロンプト」と「そのクエリーに対する説明を含む回答」に変換してください. ただし、クエリ文が長くなりすぎないよう適当に情報は取捨選択してください\n\
例1.\n\
入力: 箱根ガラスの森美術館、神奈川県、関東地方、ベネチアをイメージした園内は、本当に日本じゃないみたいなロマンチックな雰囲気に包まれています。庭では中央にガラスのアーチがキラキラ輝いていて存在感がすごいです。クリスマスとかにカップルで来たら素敵だと思います！\n\
出力:\n\
    質問: 「神奈川県で外国感のあるロマンチックな場所を教えて」\n\
    回答: 「箱根ガラスの森美術館です. ベネチアをイメージした園内は、日本じゃないみたいなロマンチックな雰囲気に包まれています。庭では中央にガラスのアーチがキラキラ輝いていて存在感がすごいです。」\n\
例2.\n\
入力: 難波八阪神社、大阪府、関西地方、とっても大きな鬼？のような獅子頭にびっくりします！舞台に近づいてみると「歯」の大きさにまたびっくり、上を見上げると「鼻の穴」の大きさにまたまたびっくり！平日は訪れる人も少なく、ゆったりお参りできます。是非一度参拝してみてください。\n\
出力:\n\
    質問:「大阪府で独特な大きなオブジェがある神社を教えて」\n\
    回答:「難波八阪神社です。とっても大きな鬼のような獅子頭があり、その大きさに驚かされます。舞台に近づいてみると『歯』の大きさにまたびっくりし、上を見上げると『鼻の穴』の大きさにも驚かされます。平日は訪れる人も少なく、ゆったりとお参りできます。」\n\
例3.\n\
入力: 白浜温泉、千葉県、関東地方、潮の満ち引きで、消えたり現れたりする珍しい仏様で、干満時にお参りすると、満潮の時に病苦を洗い流してくれるそうです。場所は、御座漁港近くにあります。「石仏地蔵尊」と書かれた小さくて黄色い案内板が道沿いに出ていました。入口は道が狭いですが、駐車場はきちんとしたものがありました。\n\
出力:\n\
    質問:「関東地方で美しい散歩道や灯台がある観光スポットを教えて」\n\
    回答:「白浜温泉です。９月に訪れましたが、最高のスポットです。キレイな遊歩道や雄大な灯台があり、房総最南端の小高い岩の上にはベンチがあり、太平洋と東京湾の荒波がぶつかる様子を楽しめます。夜にはライトアップされる灯台もあり、そのスポットライトは実に圧巻です。都心から高速を使えば２時間弱でアクセスできる穴場です。」\n\
それでは次のレビューに対して同様に質問と回答を出力してください。\n\
入力: {spot}、{pref}、{region}、{review}\n\
出力:\n"
    return prompt


# prompt_fos_japanese = "レビューから(feature, opinion, sentiment)の組み合わせを抽出してください。出力は(feature, opinion, sentiment)の組みのみで、sentimentは1（ポジティブ）、0（ニュートラル）、-1（ネガティブ）です。\n\
# また, featureとopinionはなるべく簡潔にし, 分けられるものは分けてください。また、faetureとopinionは、表記揺れがないように一般的な単語に変換してください。\n\
# また観光地のfeature同士の関係性がある時もsentimentを2として抽出してください\n\
# 例：\n\
# 入力：\n\
#     セイウチのショーはとてもお勧めです。吉本のお笑いを見ているようで、爆笑の渦でした。動物園とか水族館のショーをこれまで色々見てきましたが、ここまで笑えたのは初めてです。\n\
# 出力：\n\
#     (セイウチのショー, お勧め, 1), (セイウチのショー, お勧め, 1), (セイウチのショー, 笑える, 1)\n\
# 入力：\n\
#     動くティラノサウルスが大迫力でした！動きも滑らかで、一瞬本物かと錯覚するよう(ﾟДﾟ)コロナウイルスの影響で入館予約制だったんですが、それでも結構混んでます。入館チケットと予約票を見せて入館になるので、チケットは先に引き換えすることをお勧めします。朝イチは混んでます。\n\
# 出力：\n\
#     (ティラノサウルス, 大迫力, 1), (ティラノサウルス, 動きが滑らか, 1), (館内, 混雑, -1),(チケット, 先に引き換え推奨, 1), (入館, 予約制, 2), (朝イチ, 混雑, -1)\n\
# 入力：\n\
#     「方除の大社」として知られているので引越しや旅行の前に祈願に行くと良いです。造園家の中根金作さんが手掛けられたお庭が年中美しく茶室や池があり立派なお庭です。\n\
# 出力：\n\
#     (観光地, 方除の大社, 2), (お庭, 年中美しい, 1), (お庭, 茶室, 2), (お庭, 池, 2), (お庭, 中根金作, 2), (お庭, 立派, 1)\n\
# 入力：\n\
#     秋田駅から歩いて行ける距離にあり、便利です。館内も綺麗で様々な展示を行っているので好きな美術館のひとつです。\n\
# 出力：\n\
#     (秋田駅, 歩いて行ける距離, 1), (館内, 綺麗, 1), (展示, 様々, 1)\n\
# \n\
# 以下のレビューに対して同様に出力してください：\n\
# 入力：\n\
#     {}\n\
# 出力："