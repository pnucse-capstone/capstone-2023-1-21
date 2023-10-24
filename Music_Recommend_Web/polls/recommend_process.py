import json
import pandas as pd
import math
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import konlpy
from konlpy.tag import Okt
import re
import nltk
from nltk.corpus import stopwords
from googletrans import Translator
import time


def get_tag_mean(input_songs, simi_songs, imb_mode, w2v, tag_weights):
    #print(input_songs)
    #print(simi_songs)
    if imb_mode == True:
        user_tag_mean = np.mean([w2v.wv[tag] * tag_weights[tag] for tag in input_songs if tag in w2v.wv], axis=0)
    else:
        user_tag_mean = np.mean([w2v.wv[tag] for tag in input_songs if tag in w2v.wv], axis=0)
    
    if np.any(np.isnan(user_tag_mean)):
        user_tag_mean = np.zeros(w2v.vector_size)
        
    song_tag_mean = []
    for tags in simi_songs['tags']:
        song_tags = [tag for tag in tags if tag in w2v.wv]
        
        if len(song_tags) > 0:
            temp_mean = np.mean([w2v.wv[tag] * tag_weights[tag] for tag in song_tags], axis=0)
            if np.any(np.isnan(temp_mean)):
                temp_mean = np.zeros(w2v.vector_size)
            song_tag_mean.append(temp_mean)
        else:
            song_tag_mean.append(np.zeros(w2v.vector_size))
    
    song_tag_mean = np.array(song_tag_mean)
    tag_mean_simi = cosine_similarity([user_tag_mean], song_tag_mean)
    simi_songs['tag_similarity'] = tag_mean_simi[0]
    rec_idx = tag_mean_simi[0].argsort()[::-1]
    return rec_idx

def get_embedding(songs, mode):
    songs['gnr_literal'] = songs['gnr'].apply(lambda x : (' ').join(x))
    
    if mode == 'cv':
        count_vect = CountVectorizer()
        gnr_mat = count_vect.fit_transform(songs['gnr_literal'])
    
    elif mode == 'tf':
        tfidf_vect = TfidfVectorizer()
        gnr_mat = tfidf_vect.fit_transform(songs['gnr_literal'])
        
    return gnr_mat

def get_sim(song_index, gnr_mat, sim, pea_cor):
    if sim == 'cos':
        gnr_sim = cosine_similarity(gnr_mat[song_index], gnr_mat)
                
    elif sim == 'pea':
        #gnr_sim = []
        #gnr_arr = gnr_mat.toarray()
        #print(gnr_arr)
        
        #for song in gnr_arr:
        #    cor = pearsonr(gnr_arr[song_index][0], song)[0]
        #    if cor == 1: # pearsonr 특성 상, 같은 벡터를 넣으면 분모가 0이 되는 문제를 방지
        #        cor = 0
        #    gnr_sim.append(1 - abs(cor))
        
        # NaN 값을 0으로 변경
        gnr_sim = pea_cor[song_index]
        gnr_sim = np.nan_to_num(gnr_sim, nan=0)

    return np.array(gnr_sim)

def get_like_weight(song_df, like_mode):
    # 스케일을 줄이기 위해 like_cnt_song 열에 로그 처리
    song_df['like_cnt_song'] = song_df['like_cnt_song'].apply(lambda x : math.log(x + 1))
    
    max_like = max(song_df['like_cnt_song'])
    len_like = len(song_df['like_cnt_song'])
    
    if like_mode == 0:
        song_df['weight'] = song_df['like_cnt_song'].apply(lambda x : 1 - (x / max_like))
    else:
        song_df['weight'] = song_df['like_cnt_song'].apply(lambda x : 1 - (x / len_like))
    
    for i in range(len(song_df)):
        song_df['similarity'].iloc[i] = song_df['similarity'].iloc[i] * song_df['weight'].iloc[i]
        
    return song_df

def apply_genre_weight(mat, mode):
    genre_nums = mat.getnnz(0)
    genre_num_max = max(genre_nums)
    if mode == 0:
        genre_weight = np.log(genre_num_max / genre_nums + 1)
    else:
        genre_weight = np.log(mat.shape[0] / genre_nums + 1)
    
    result_mat = mat.copy()
    
    for i in range(result_mat.shape[0]):
        for j in result_mat[i].indices:
            result_mat[(i, j)] *= genre_weight[j]
    
    return result_mat

def find_sim_song(song_df, sim, mat, weight_mat_cv, weight_mat_tf, songs, emb_mode, weight_mode, like_mode, genre_imb_mode=False, like_weight=False, top_n=0):
    simi = np.zeros(len(song_df['song_id']))
    minyear = 3000
    #print(f"User Plist: {songs}")
    
    for song in songs:
        title_song = song_df[song_df['song_id'] == song]
        if not title_song.empty:
            minyear = min(minyear, title_song['issue_date'].values[0]//10000)
    
    simi_dict = dict()

    pea_cor = []
    if sim == 'pea':
        if genre_imb_mode:
            if emb_mode == 'cv':
                pea_cor = np.corrcoef(weight_mat_cv[weight_mode].toarray())
            elif emb_mode == 'tf':
                pea_cor = np.corrcoef(weight_mat_tf[weight_mode].toarray())
        else:
            pea_cor = np.corrcoef(mat.toarray())
        pea_cor[pea_cor == 1] = 0
        pea_cor = 1 - abs(pea_cor)

    for song in songs:
        title_song = song_df[song_df['song_id'] == song]
        
        if title_song.empty:
            continue
            
        title_index = title_song.index.values
        
        if title_index[0] in simi_dict:
            sim_array = simi_dict[title_index[0]]
        
        else:
            if genre_imb_mode:
                if emb_mode == 'cv':
                    sim_array = get_sim(title_index, weight_mat_cv[weight_mode], sim, pea_cor)
            
                elif emb_mode == 'tf':
                    sim_array = get_sim(title_index, weight_mat_tf[weight_mode], sim, pea_cor)
            
                #elif emb_mode == 'aver':
                    #sim_array = get_sim(title_index, weight_mat_aver, sim)
        
            else:
                sim_array = get_sim(title_index, mat, sim, pea_cor)
            
            simi_dict[title_index[0]] = sim_array
            
        simi = simi + sim_array
    
    simi /= len(songs)
    
    # 유사도 값을 0~1 사이로 Scaling
    #simi *= (simi - min(simi)) / (max(simi) - min(simi))

    temp = song_df.copy()
    temp['similarity'] = simi.reshape(-1, 1)

    # 가중치 스위치가 켜져 있다면 가중치 적용
    if like_weight:
        temp = get_like_weight(temp, like_mode)
            
    temp = temp.sort_values(by="similarity", ascending=False)
    
    # for song in songs:
    #     title_song = df[df['song_id'] == song]
    #     title_index = title_song.index.values
        
    #     temp = temp[temp.index.values != title_index]
    
    temp = temp[temp['issue_date'] > minyear*10000]
        
    # 유사도가 0.5 이하인 경우는 제외
    #temp = temp[temp['similarity'] >= 0.5]
    if top_n < 1:
        temp = temp[temp['similarity'] >= 0.4]
        temp = temp.reset_index(drop=True)
        return temp
    else:
        temp = temp.reset_index(drop=True)
        return temp.iloc[ : top_n]
    
    # final_index = temp.index.values[ : top_n]


def get_recall_k(y_true, y_pred):
    recall_k = 0
    true_items = set(y_true)
    pred_items = set(y_pred)
    intersect_items = len(true_items.intersection(pred_items))
    recall = intersect_items / len(true_items) if len(true_items) > 0 else 0
    return recall


def get_precision_k(y_true, y_pred):
    precision_k = 0
    true_items = set(y_true)
    pred_items = set(y_pred)
    intersect_items = len(true_items.intersection(pred_items))
    precision = intersect_items / len(pred_items) if len(pred_items) > 0 else 0
    return precision

def song_recommend2(tags, songs, song_df, sim, tag_imb_mode, genre_imb_mode, like_weight, emb_mode, weight_mode, like_mode, w2v, weight_mat_cv, weight_mat_tf, tag_weights):
    # 기존 노래(히스토리)가 있는 경우 장르 유사도를 계산해
    #상위 100개의 노래를 찾아낸다
    if len(songs) > 0:
        # 장르 불균형데이터에 대해서는 상위 100개의 음악을 추출하는 것이 유효함
        #if genre_imb_mode:
        simi_songs = find_sim_song(song_df, sim, get_embedding(song_df, emb_mode), weight_mat_cv, weight_mat_tf, songs, emb_mode, weight_mode, like_mode, genre_imb_mode, like_weight, 100)
            
        #else:
        #simi_songs = find_sim_song(song_df, sim, get_cv(song_df), songs) #상위 100개가 아닌, 유사도가 0.4 이상인 음악만 추출
    
    # 기존 노래(히스토리)가 없는 경우 최신 노래(2018~2023년도)를 찾아낸다
    else:
        simi_songs = song_df
        simi_songs = simi_songs[simi_songs['issue_date'] > 20180000]
        simi_songs = simi_songs[simi_songs['issue_date'] < 20240000]
    
    #print(simi_songs.shape)
    ts = tags
    
    # 해당 태그가 존재하는 플레이리스트의 노래를 추출하고 등장 빈도수로 정렬한다
    tag_songs = dict()
    tag_simi_mean = []
    
    sorted_idx = get_tag_mean(ts, simi_songs, tag_imb_mode, w2v, tag_weights)
    tag_simi_mean = simi_songs.loc[sorted_idx]
    #print(tag_simi_mean)
    return tag_simi_mean.iloc[:10]

def music_prediction(test_my_songs, test_my_tags, song_tag_appended, simi_mode, tag_imb, genre_imb, like, mode, weight_mode, like_mode, tag_mode, w2v, weight_mat_cv, weight_mat_tf, tag_weights, tag_weights_all):
    pred_list = []
    tags = test_my_tags[-1]
    plist = test_my_songs[-1]
    if tag_mode == 0:
        recommended = song_recommend2(tags, plist, song_tag_appended, simi_mode, tag_imb, genre_imb, like, mode, weight_mode, like_mode, w2v,
                                     weight_mat_cv, weight_mat_tf, tag_weights)
    elif tag_mode == 1:
        recommended = song_recommend2(tags, plist, song_tag_appended, simi_mode, tag_imb, genre_imb, like, mode, weight_mode, like_mode, w2v,
                                     weight_mat_cv, weight_mat_tf, tag_weights_all)
            
    #pred_list.append(recommended['song_id'].tolist())
    #print(f"User Tags: {tags}")
    return recommended

def make_song_num_dict(data):
    song_ids = dict()
    song_num = dict()
    max_num = 0
    
    for i in range(len(data)):
        songs = data['song_id'][i]
        tags = data['tags'][i]
        for j in tags:
            if not j in song_ids:
                song_ids[j] = set(songs)
            else:
                song_ids[j].update(songs)
    
    for i in song_ids:
        song_num[i] = len(song_ids[i])
        max_num = max(song_num[i], max_num)
    
    return song_num, max_num

def main_process_no(sample_index, user_tags, user_songs, prepro_para):
    main_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(main_dir, 'Datasets/train_datas.json')
    song_dir = os.path.join(main_dir, 'Datasets/song_meta_with_likes.json')

    with open(train_dir, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    train_data = pd.DataFrame(json_data)
    train_data.rename(columns={'songs':'song_id'}, inplace=True)

    with open(song_dir, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    song_data = pd.DataFrame(json_data)
    song_data = song_data.drop(['album_name', 'song_gn_gnr_basket'], axis=1)
    song_data.rename(columns={'id':'song_id', 'song_gn_dtl_gnr_basket': 'gnr'}, inplace=True)
    song_data = song_data.astype({'issue_date':'int64'})
    
    train_data_origin = train_data.copy()
    train_data_origin = train_data_origin.loc[sample_index]
    train_data_origin = train_data_origin.reset_index(drop=True)
    
    train_data_appended = train_data_origin.copy()

    user_row = [user_tags, user_songs, 1]
    user_row_df = pd.DataFrame([user_row], columns=['tags', 'song_id', 'like_cnt'])
    train_data_appended = pd.concat([train_data_appended, user_row_df], ignore_index=True)
        
    train_data_sample = train_data_appended.copy()
    train_data_sample = train_data_sample.explode('song_id', ignore_index=True)
    train_dict = dict()

    for i in range(len(train_data_sample)):
        song = train_data_sample['song_id'][i]
        tag = train_data_sample['tags'][i]
        if song in train_dict:
            for j in tag:
                train_dict[song].add(j)
        else:
            train_dict[song] = set(tag)

    train_data_sample.drop_duplicates(subset='song_id', keep='first',inplace=True)

    for i in range(len(train_data_sample)):
        song = train_data_sample['song_id'].iloc[i]
        train_data_sample['tags'].iloc[i] = list(train_dict[song])

    song_tag_appended = pd.merge(train_data_sample, song_data)
    song_tag_appended = song_tag_appended.astype({'song_id':'int64'})
        
    w2v = Word2Vec(sentences = song_tag_appended['tags'], vector_size = 100, window = 5, min_count = 15, workers = 4, sg = 1)

    train_data_sample2 = train_data_appended.copy()
    train_data_sample2 = train_data_sample2.reset_index(drop=True)
        
    song_num_dict, song_num_max = make_song_num_dict(train_data_sample2)
    tag_weights = {tag: np.log(song_num_max / cnt + 1) for tag, cnt in song_num_dict.items()}
    tag_weights_all = {tag: np.log(len(song_tag_appended) / cnt + 1) for tag, cnt in song_num_dict.items()}

    song_tag_appended['gnr_literal'] = song_tag_appended['gnr'].apply(lambda x : (' ').join(x))

    test_data_sample = train_data_appended.copy()
    test_data_sample = test_data_sample.reset_index(drop=True)
    test_my_tags = test_data_sample['tags'].tolist()
    test_my_songs = test_data_sample['song_id'].tolist()

    weight_mat_cv = []
    weight_mat_tf = []

    weight_mat_cv.append(apply_genre_weight(get_embedding(song_tag_appended, 'cv'), 0))
    weight_mat_cv.append(apply_genre_weight(get_embedding(song_tag_appended, 'cv'), 1))

    weight_mat_tf.append(apply_genre_weight(get_embedding(song_tag_appended, 'tf'), 0))
    weight_mat_tf.append(apply_genre_weight(get_embedding(song_tag_appended, 'tf'), 1))
    
    print(test_data_sample.shape)
    result = music_prediction(test_my_songs, test_my_tags, song_tag_appended, 'cos', prepro_para[1], prepro_para[2], False, prepro_para[3], 0, 0, 0, w2v, weight_mat_cv, weight_mat_tf, tag_weights, tag_weights_all)
    return result

def main_process_kor(sample_index, user_tags, user_songs, prepro_para):
    main_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(main_dir, 'Datasets/train_datas.json')
    song_dir = os.path.join(main_dir, 'Datasets/song_meta_with_likes.json')

    with open(train_dir, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    train_data = pd.DataFrame(json_data)
    train_data.rename(columns={'songs':'song_id'}, inplace=True)

    with open(song_dir, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    song_data = pd.DataFrame(json_data)
    song_data = song_data.drop(['album_name', 'song_gn_gnr_basket'], axis=1)
    song_data.rename(columns={'id':'song_id', 'song_gn_dtl_gnr_basket': 'gnr'}, inplace=True)
    song_data = song_data.astype({'issue_date':'int64'})
    
    train_data_origin = train_data.copy()
    train_data_origin = train_data_origin.loc[sample_index]
    train_data_origin = train_data_origin.reset_index(drop=True)
    
    train_data_appended = train_data_origin.copy()

    user_row = [user_tags, user_songs, 1]
    user_row_df = pd.DataFrame([user_row], columns=['tags', 'song_id', 'like_cnt'])
    train_data_appended = pd.concat([train_data_appended, user_row_df], ignore_index=True)
        
    train_data_sample = train_data_appended.copy()

    okt = Okt()
    for i in range(len(train_data_sample)):
        preprocessed_tags = []
        for tag in train_data_sample['tags'][i]:
            normalized_tag = okt.normalize(tag)
            pos_tagging = okt.pos(normalized_tag)
            for word, pos in pos_tagging:
                if pos == 'Noun' or pos == 'Verb' or pos == 'Adjective':
                    preprocessed_tags.append(word)
                elif pos == 'Alpha':
                    preprocessed_tags.append(word.lower())
        if len(preprocessed_tags) == 0:
            train_data_sample['tags'][i] = preprocessed_tags
        else:
            preprocessed_tags = list(dict.fromkeys(preprocessed_tags))
            train_data_sample['tags'][i] = preprocessed_tags

    preprocessed_plist = train_data_sample.copy()
    
    train_data_sample = train_data_sample.explode('song_id', ignore_index=True)
    train_dict = dict()

    for i in range(len(train_data_sample)):
        song = train_data_sample['song_id'][i]
        tag = train_data_sample['tags'][i]
        
        if song in train_dict:
            for j in tag:
                train_dict[song].add(j)
        
        else:
            train_dict[song] = set(tag)
            
    train_data_sample.drop_duplicates(subset='song_id', keep='first',inplace=True)

    for i in range(len(train_data_sample)):
        song = train_data_sample['song_id'].iloc[i]
        train_data_sample['tags'].iloc[i] = list(train_dict[song])

    song_tag_appended = pd.merge(train_data_sample, song_data)
    song_tag_appended = song_tag_appended.astype({'song_id':'int64'})
    
    w2v = Word2Vec(sentences = song_tag_appended['tags'], vector_size = 100, window = 5, min_count = 15, workers = 4, sg = 1)
    
    song_num_dict, song_num_max = make_song_num_dict(preprocessed_plist)
    tag_weights = {tag: np.log(song_num_max / cnt + 1) for tag, cnt in song_num_dict.items()}
    tag_weights_all = {tag: np.log(len(song_tag_appended) / cnt + 1) for tag, cnt in song_num_dict.items()}

    song_tag_appended['gnr_literal'] = song_tag_appended['gnr'].apply(lambda x : (' ').join(x))
    
    test_data_sample = preprocessed_plist.copy()
    test_data_sample = test_data_sample.reset_index(drop=True)
    test_my_tags = test_data_sample['tags'].tolist()
    test_my_songs = test_data_sample['song_id'].tolist()

    weight_mat_cv = []
    weight_mat_tf = []

    weight_mat_cv.append(apply_genre_weight(get_embedding(song_tag_appended, 'cv'), 0))
    weight_mat_cv.append(apply_genre_weight(get_embedding(song_tag_appended, 'cv'), 1))

    weight_mat_tf.append(apply_genre_weight(get_embedding(song_tag_appended, 'tf'), 0))
    weight_mat_tf.append(apply_genre_weight(get_embedding(song_tag_appended, 'tf'), 1))
    
    print(test_data_sample.shape)
    result = music_prediction(test_my_songs, test_my_tags, song_tag_appended, 'cos', prepro_para[1], prepro_para[2], False, prepro_para[3], 0, 0, 0, w2v, weight_mat_cv, weight_mat_tf, tag_weights, tag_weights_all)
    return result

def trans_to_eng(tags):
    time.sleep(1)
    translator = Translator()
    translated = translator.translate(tags, src = 'ko', dest = 'en')
    return translated.text

def main_process_eng(sample_index, user_tags, user_songs, prepro_para):
    main_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(main_dir, 'Datasets/train_datas_eng.json')
    song_dir = os.path.join(main_dir, 'Datasets/song_meta_with_likes.json')

    with open(train_dir, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    train_data_eng = pd.DataFrame(json_data)
    train_data_eng = train_data_eng.loc[sample_index]
    train_data_eng = train_data_eng.reset_index(drop=True)

    with open(song_dir, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    song_data = pd.DataFrame(json_data)
    song_data = song_data.drop(['album_name', 'song_gn_gnr_basket'], axis=1)
    song_data.rename(columns={'id':'song_id', 'song_gn_dtl_gnr_basket': 'gnr'}, inplace=True)
    song_data = song_data.astype({'issue_date':'int64'})
    
    #------------------------------------- 이 부분에서 사용자 플레이리스트 영어 번역하기
    user_eng_tags = [trans_to_eng(tag) for tag in user_tags]
    user_tags = user_eng_tags
    
    user_row = [user_tags, user_songs, 1]
    user_row_df = pd.DataFrame([user_row], columns=['tags', 'song_id', 'like_cnt'])
    train_data_eng = pd.concat([train_data_eng, user_row_df], ignore_index=True)

    nltk.download('popular')
    pattern = re.compile('[^a-zA-Z0-9]')
    idx = 0
    for tags in train_data_eng['tags']:
        eng_tags = []
        for tag in tags:
            temp_tags = re.sub(pattern, ' ', tag).lower().split()
            for temp_tag in temp_tags:
                eng_tags.append(temp_tag)
        train_data_eng['tags'][idx] = eng_tags
        idx += 1

    #Stopwords
    stops = set(stopwords.words('english'))

    for i in range(len(train_data_eng)):
        eng_tags = [tag for tag in train_data_eng['tags'][i] if not tag in stops]
        train_data_eng['tags'][i] = eng_tags
        
    #Stemming
    stemmer = nltk.stem.SnowballStemmer('english')
    for i in range(len(train_data_eng)):
        eng_tags = [stemmer.stem(tag) for tag in train_data_eng['tags'][i]]
        train_data_eng['tags'][i] = eng_tags

    #중복 제거
    for i in range(len(train_data_eng)):
        eng_tags = list(dict.fromkeys(train_data_eng['tags'][i]))
        train_data_eng['tags'][i] = eng_tags
        
    #한 글자 제거
    for i in range(len(train_data_eng)):
        eng_tags = [tag for tag in train_data_eng['tags'][i] if len(tag) > 1]
        train_data_eng['tags'][i] = eng_tags


    # '록' 또는 '락' 이 'lock' 으로 번역되는 문제가 있어서, 'rock' 으로 일괄적으로 수정
    pattern = re.compile(r'\block\b')
    for i in range(len(train_data_eng)):
        eng_tags = [re.sub(pattern, 'rock', tag) for tag in train_data_eng['tags'][i]]
        train_data_eng['tags'][i] = eng_tags
        
    train_data_sample = train_data_eng.copy()
    train_data_sample = train_data_sample.reset_index(drop=True)
    train_data_sample = train_data_sample.explode('song_id', ignore_index=True)
    
    train_dict = dict()

    for i in range(len(train_data_sample)):
        song = train_data_sample['song_id'][i]
        tag = train_data_sample['tags'][i]
        
        if song in train_dict:
            for j in tag:
                train_dict[song].add(j)
        
        else:
            train_dict[song] = set(tag)
            
    train_data_sample.drop_duplicates(subset='song_id', keep='first',inplace=True)
    
    for i in range(len(train_data_sample)):
        song = train_data_sample['song_id'].iloc[i]
        train_data_sample['tags'].iloc[i] = list(train_dict[song])
        
    song_tag_appended = pd.merge(train_data_sample, song_data)
    song_tag_appended = song_tag_appended.astype({'song_id':'int64'})
    
    train_data_sample2 = train_data_eng.copy()
    train_data_sample2 = train_data_sample2.reset_index(drop=True)

    w2v = Word2Vec(sentences = song_tag_appended['tags'], vector_size = 100, window = 5, min_count = 15, workers = 4, sg = 1)
    
    song_num_dict, song_num_max = make_song_num_dict(train_data_sample2)
    tag_weights = {tag: np.log(song_num_max / cnt + 1) for tag, cnt in song_num_dict.items()}
    tag_weights_all = {tag: np.log(len(song_tag_appended) / cnt + 1) for tag, cnt in song_num_dict.items()}
    song_tag_appended['gnr_literal'] = song_tag_appended['gnr'].apply(lambda x : (' ').join(x))
    
    test_data_sample = train_data_eng.copy()
    test_data_sample = test_data_sample.reset_index(drop=True)
    test_my_tags = test_data_sample['tags'].tolist()
    test_my_songs = test_data_sample['song_id'].tolist()

    weight_mat_cv = []
    weight_mat_tf = []

    weight_mat_cv.append(apply_genre_weight(get_embedding(song_tag_appended, 'cv'), 0))
    weight_mat_cv.append(apply_genre_weight(get_embedding(song_tag_appended, 'cv'), 1))

    weight_mat_tf.append(apply_genre_weight(get_embedding(song_tag_appended, 'tf'), 0))
    weight_mat_tf.append(apply_genre_weight(get_embedding(song_tag_appended, 'tf'), 1))
    
    print(test_data_sample.shape)
    result = music_prediction(test_my_songs, test_my_tags, song_tag_appended, 'cos', prepro_para[1], prepro_para[2], False, prepro_para[3], 0, 0, 0, w2v, weight_mat_cv, weight_mat_tf, tag_weights, tag_weights_all)
    return result
