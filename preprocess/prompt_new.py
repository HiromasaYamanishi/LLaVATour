import random
from easydict import EasyDict
import pandas as pd


def add_premise_prompt(spot, pattern="caption", gender=None, age=None, tag=None, month=None, season=None):
    prompt = ""
    season_random = random.random()
    if season_random < 0.2:
        season_suffix = ''
        season_pattern = 1
    elif 0.2<=season_random<0.6:
        if month is None or pd.isna(month):
            season_suffix = ''
        else:
            season_suffix = str(month) + 'に'
        season_pattern = 2
    else:
        if season is None or pd.isna(season):
            season_suffix = ''
        else:
            season_suffix = season + 'に'
        season_pattern = 3
    if pattern == "caption":
        spot_random = random.random()
        if spot_random < 0.3:
            prompt += f"この場所は{spot}です"
        elif 0.3 < spot_random < 0.6:
            prompt += f"この観光地は{spot}です"
        else:
            prompt += f"これは{spot}の写真です"
        data_type = 1
        
    elif pattern == "visitor":
        prompt += f"あなたは{season_suffix}{spot}"
        syugo_random = random.random()
        if syugo_random < 0.3:
            prompt += "に観光で訪れました"
        elif 0.3 < syugo_random < 0.6:
            prompt += "を訪問しました"
        else:
            prompt += "に旅行で来ました"
        data_type = 1 + season_pattern

    elif pattern == "context":
        prompt += f"あなたは{season_suffix}{spot}"
        syugo_random = random.random()
        if syugo_random < 0.3:
            prompt += "に観光で訪れた"
        elif 0.3 < syugo_random < 0.6:
            prompt += "を訪問した"
        else:
            prompt += "に旅行で来た"
        context_num = random.randint(1, 3)
        inds = random.sample([0, 1, 2], context_num)
        contexts = [gender, age, tag]
        for ind in inds:
            if pd.isna(contexts[ind]):
                continue
            prompt += f"{contexts[ind]}の"
        if context_num == 0:
            context_pattern = 1
        else:
            context_pattern = 2
            
        data_type = season_pattern * context_pattern
        
        syugo_random = random.random()
        if syugo_random < 0.3:
            prompt += "旅行客です"
        elif 0.3 < syugo_random < 0.6:
            prompt += "観光客です"
        else:
            prompt += "訪問客です"

    return prompt, data_type


def add_photo_prompt(prompt):
    input_random = random.random()
    if input_random < 0.3:
        prompt += "この"
    elif 0.3 < input_random < 0.6:
        prompt += "与えられた"
    else:
        prompt += "入力された"

    photo_random = random.random()
    if photo_random < 0.3:
        prompt += "画像について"
    elif 0.3 < photo_random < 0.6:
        prompt += "写真について"
    else:
        prompt += "画像を見て"
    return prompt


def add_review_prompt(prompt, short, pattern="caption"):
    premise_random = random.random()
    review_random = random.random()
    if pattern == "caption":
        if premise_random < 0.4:
            prompt += ""
        elif 0.4 < premise_random < 0.6:
            prompt += "訪問したつもりで"
        elif 0.6 < premise_random < 0.8:
            prompt += "観光客のつもりで"
        else:
            prompt += "観光客のように"

    if review_random < 0.7:
        prompt += "レビューを"
    else:
        prompt += "説明文を"

    if short:
        short_random = random.random()
        if short_random < 0.3:
            prompt += "簡単に"
        elif 0.3 < review_random < 0.6:
            prompt += "簡潔に"
        else:
            prompt += "手短に"

    action_random = random.random()
    if action_random < 0.3:
        prompt += "生成してください"
    elif 0.3 < action_random < 0.5:
        prompt += "書いてください"
    elif 0.5 < action_random < 0.7:
        prompt += "作成してください"
    else:
        prompt += "述べてください"
    return prompt


def add_pre_posneg_prompt(prompt, phrase):
    prompt += f"「{phrase}」という"
    keyword_random = random.random()
    if keyword_random < 0.2:
        prompt += "キーワードを含めて"
    elif 0.2 < keyword_random < 0.4:
        prompt += "フレーズを入れて"
    elif 0.4 < keyword_random < 0.6:
        prompt += "フレーズを含めて"
    elif 0.6 < keyword_random < 0.8:
        prompt += "一文を入れて"
    else:
        prompt += "キーワードを入れて"

    return prompt


def add_post_posneg_prompt(prompt, phrase):
    prompt += f"ただし，「{phrase}」という"
    keyword_random = random.random()
    if keyword_random < 0.2:
        prompt += "キーワードを含めてください"
    elif 0.2 < keyword_random < 0.4:
        prompt += "フレーズを入れて下さい"
    elif 0.4 < keyword_random < 0.6:
        prompt += "フレーズを含めて下さい"
    elif 0.6 < keyword_random < 0.8:
        prompt += "一文を入れてください"
    else:
        prompt += "キーワードを入れてください"

    return prompt


def make_caption_prompt(spot):
    # 例: この場所は函館山です. この画像について観光したつもりでレビューを簡単に生成してください
    prompt, data_type = add_premise_prompt(spot, pattern="caption")
    prompt = add_photo_prompt(prompt)
    prompt = add_review_prompt(prompt, short=True, pattern="caption")
    return prompt, data_type


def make_review_prompt(
    spot,
    short=False,
):
    # 例: あなたは函館山を訪れた観光客です. この画像についてレビューを書いてください
    prompt, data_type = add_premise_prompt(spot, pattern="visitor")
    prompt = add_photo_prompt(prompt)
    prompt = add_review_prompt(prompt, short=short, pattern="visitor")
    return prompt, data_type


def make_review_context_prompt(spot, tag, gender, age, month, season):
    # 例: あなたは函館山を訪れた30代家族連れの観光客です. この画像についてレビューを書いてください
    prompt_random = random.random()
    if prompt_random < 0.9:
        prompt, data_type = add_premise_prompt(
            spot, pattern="context", gender=gender, age=age, tag=tag, month=month, season=season
        )
    else:
        prompt, data_type = add_premise_prompt(spot, pattern="visitor")
    prompt = add_photo_prompt(prompt)
    prompt = add_review_prompt(prompt, short=False, pattern="visitor")
    return prompt, data_type


def make_review_context_posneg_prompt(spot, tag, gender, age, month, season,matches):
    # 例: あなたは函館山を訪れた30代家族連れの観光客です. 「夜景が綺麗」というキーワードを含めて，この画像についてレビューを書いてください
    prompt_random = random.random()
    if prompt_random < 0.9:
        prompt, data_type = add_premise_prompt(
            spot, pattern="context", gender=gender, age=age, tag=tag, month=month, season=season
        )
    else:
        prompt, data_type = add_premise_prompt(spot, pattern="visitor")
    if not len(matches):
        prompt = add_photo_prompt(prompt)
        prompt = add_review_prompt(prompt, short=False, pattern="visitor")

    else:
        phrase = random.sample(matches, 1)[0]
        place_random = random.random()
        if place_random < 0.3:
            prompt = add_pre_posneg_prompt(prompt, phrase)
            prompt = add_photo_prompt(prompt)
            prompt = add_review_prompt(prompt, short=False, pattern="visitor")
        elif 0.3 < place_random < 0.6:
            prompt = add_photo_prompt(prompt)
            prompt = add_pre_posneg_prompt(prompt, phrase)
            prompt = add_review_prompt(prompt, short=False, pattern="visitor")
        else:
            prompt = add_photo_prompt(prompt)
            prompt = add_review_prompt(prompt, short=False, pattern="visitor")
            prompt = add_post_posneg_prompt(prompt, phrase)
    data_type = data_type + 7
    return prompt, data_type


def make_spot_name_prompt(spot):
    prompt = ""
    input_random = random.random()
    if input_random < 0.3:
        prompt += "この場所の"
    elif 0.3 < input_random < 0.6:
        prompt += "この観光地の"
    else:
        prompt += "写真の観光地の"

    photo_random = random.random()
    if photo_random < 0.3:
        prompt += "名前を教えて"
    elif 0.3 < photo_random < 0.6:
        prompt += "名称を教えて"
    else:
        prompt += "名前はなんですか"
    return prompt


def make_tag_premise_prompt(spot_name, date, caption):
    photo_random = random.random()
    if photo_random < 0.5:
        suffix = "画像"
    else:
        suffix = "写真"
    if date is None and caption is None:
        prompt_type = random.sample([1,2], 1)[0]
    elif date is None and caption is not None:
        prompt_type = random.sample([1, 2, 3], 1)[0]
    elif date is not None and caption is None:
        prompt_type = random.sample([1, 2, 4], 1)[0]
    else:
        prompt_type = random.sample([1, 2,3, 4, 5], 1)[0]
    if prompt_type == 1:
        prompt = f"この{suffix}の"
    elif prompt_type == 2:
        prompt = f"この{suffix}は{spot_name}で撮られた{suffix}です。この{suffix}の"
    elif prompt_type == 3:
        prompt = f"この{suffix}は{spot_name}で撮られた{suffix}です。 キャプションは{caption}です。この{suffix}の"
    elif prompt_type == 4:
        prompt = f"この{suffix}は{spot_name}で{date}に撮られた{suffix}です。この{suffix}の"
    elif prompt_type == 5:
        prompt = f"この{suffix}は{spot_name}で{date}に撮られた{suffix}です。キャプションは{caption}です。この{suffix}の"
    return prompt, prompt_type

def make_category_prompt(sub=False):
    photo_random = random.random()
    if photo_random < 0.5:
        suffix = "画像"
    else:
        suffix = "写真"
    cat_random = random.random()
    if cat_random < 0.3:
        if sub:
            cat_suffix = '小分類'
        else:
            cat_suffix = '大分類'
    elif 0.3 <= cat_random < 0.7:
        if sub:
            cat_suffix = 'サブカテゴリ'
        else:
            cat_suffix = 'カテゴリ'
    else:
        if sub:
            cat_suffix = 'おおまかなカテゴリ'
        else:
            cat_suffix = '詳細なカテゴリ'
        
    prompt = f'{suffix}の{cat_suffix}を予測してください'
    return prompt

def make_tag_prediction_prompt(nice_count, spot_name, date, caption):
    prompt, data_type = make_tag_premise_prompt(spot_name, date, caption)
    task_type = random.randint(1, 3)
    if task_type == 1:
        prompt += "いいね数が5以上か予測してください"
        if nice_count >= 5:
            answer = "はい"
        else:
            answer = "いいえ"
    elif task_type == 2:
        prompt += "いいね数が10以上か予測してください"
        if nice_count >= 10:
            answer = "はい"
        else:
            answer = "いいえ"
    elif task_type == 3:
        prompt += "いいね数を予測してください"
        answer = nice_count
    return prompt, answer, task_type, data_type


def make_description_prompt(spot):
    prompt = ""
    spot_random = random.random()
    if spot_random < 0.2:
        prompt += f"この場所は{spot}です"
    elif 0.2 < spot_random < 0.4:
        prompt += f"この観光地は{spot}です"
    elif 0.4 < spot_random < 0.6:
        prompt += f"これは{spot}という観光地です"

    input_random = random.random()
    if input_random < 0.2:
        prompt += "この観光地の概要を教えてください"
    elif 0.2 < input_random < 0.4:
        prompt += "この観光地の基本情報を教えてください"
    elif 0.4 < input_random < 0.6:
        prompt += "この観光地について教えてください"
    elif 0.6 < input_random < 0.8:
        prompt += "この観光地について詳細に説明をしてください"
    else:
        prompt += "この観光地の詳細情報を教えてください"

    return prompt


def make_context_prompt(spot):
    prompt = add_premise_prompt(spot, pattern="caption")
    input_random = random.random()
    if input_random < 0.2:
        prompt += "この観光地を訪れる人の属性や季節の特徴を教えてください"
    elif 0.2 < input_random < 0.4:
        prompt += "この場所を訪れる人の年齢・性別やシーズンの特徴を教えてください"
    elif 0.4 < input_random < 0.6:
        prompt += "この場所を訪れる人や季節はどんなものが多いですか"
    elif 0.6 < input_random < 0.8:
        prompt += "この観光地を訪れる人の属性や季節について説明してください"
    else:
        prompt += "どのような属性の人がこの観光地を訪れますか．また，いつ訪れることが多いですか"
    return prompt


def make_gender_answer(gender_context_top):
    if gender_context_top[0] == 0:
        return "この場所を訪れる性別は女性が多いです"
    elif gender_context_top[0] == 1:
        return "この場所を訪れる性別はやや女性が多いです"
    elif gender_context_top[0] == 2:
        return "この場所を訪れる性別は男女が同じくらい訪れます"
    elif gender_context_top[0] == 3:
        return "この場所を訪れる性別はやや男性が多いです"
    else:
        return "この場所を訪れる性別は男性が多いです"


def make_age_answer(age_context_top):
    ages = ["10代", "20代", "30代", "40代", "50代以上"]
    return f"この場所を訪れることが多い年代は{ages[age_context_top[0]]}から{ages[age_context_top[1]]}が多いです"


def make_season_answer(season_context_top):
    months = [f"{i+1}月" for i in range(12)]
    return f"この場所が訪れられることが多い時期は{months[season_context_top[0]]}や{months[season_context_top[1]]}です"


def make_context_answer(
    spot_name,
    age_context_top,
    gender_context_top,
    season_context_top,
    people_context_top,
    time_context_top,
):
    answers = [
        make_gender_answer(gender_context_top),
        make_age_answer(age_context_top),
        make_season_answer(season_context_top),
    ]
    random.shuffle(answers)
    answer = ""
    for i, a in enumerate(answers):
        if i != 0:
            answer += "また，"
        answer += a
    return answer


def make_training_dict(id, image, instructions, texts):
    d = {}
    d["id"] = id
    if image is not None:
        d["image"] = image
    d["conversations"] = []
    for i, (instruction, text) in enumerate(zip(instructions, texts)):
        if i == 0 and image is not None:
            d["conversations"].append(
                {"from": "human", "value": f"<image>\n{instruction}."},
            )
        else:
            d["conversations"].append(
                {"from": "human", "value": f"{instruction}."},
            )
        d["conversations"].append({"from": "gpt", "value": f"{text}"})
    return d


prompt_qa = "レビューから質問と回答のペアを作成してください。次は例です。\n\
例1.\n\
入力: 北海道旅行にいったときに夜景を見に行きました。展望台には観光バスで上るかロープウェイをつかって上ります。展望台から見る景色はとてもきらきらしていてとても綺麗です。絶対に夜にいくことをオススメします。\n\
出力: 「展望台にはどのように登れますか」「観光バスかロープウェイを使って登ります」\n\
例2.\n\
入力: 着いてからは人の山で流れもせず見られる場所を探すのに必死でした…。夕方から夜に変わる夕暮れで景色は最高でした！北海道の5月はやっぱりまだ冷んやりで夜は温かい格好がいいです。\n\
出力: 「レビュアーは何に満足しましたか」「夕方から夜に変わる夕暮れの景色」\n\
それでは次のレビューに対して同様に質問と回答のペアを作成してください。\n\
\n"

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