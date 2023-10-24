import json
import pandas as pd
import math
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from scipy import stats

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
    for tags in simi_songs:
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

def find_sim_song(df, sim, mat, weight_mat_cv, weight_mat_tf, songs, emb_mode, weight_mode, like_mode, genre_imb_mode=False, like_weight=False, top_n=0):
    simi = np.zeros(len(df['song_id']))
    minyear = 3000
    #print(f"User Plist: {songs}")
    
    for song in songs:
        title_song = df[df['song_id'] == song]
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
        title_song = df[df['song_id'] == song]
        
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

    temp = df.copy()
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
    recall = intersect_items / len(pred_items) if len(pred_items) > 0 else 0
    return recall

def get_ap_k(y_true, y_pred, k):
    pred_items = y_pred[:k]
    hits = []
    for item in pred_items:
        if item in y_true:
            hits.append(1)
        else:
            hits.append(0)
    precision_values = []
    for i in range(1, k+1):
        precision_values.append(sum(hits[:i]) / i)
        
    #print(precision_values)
    
    if len(precision_values) == 0:
        return 0
    else:
        return sum(precision_values) / len(precision_values)
    
#ap_k = get_ap_k(my_songs1, pred_list, 10)
#print("AP@K (K=10): {:.2f}".format(ap_k))

def get_map_k(y_true, y_pred, k):
    sum_ap = 0
    for true_item, pred_item in zip(y_true, y_pred):
        ap_k = get_ap_k(true_item, pred_item, k)
        sum_ap += ap_k
    if len(y_true) == 0:
        return 0
    else:
        return sum_ap / len(y_true)
    
#map_k = get_map_k(all_my_songs, all_pred_songs, 10)
#print("MAP@K (K=10): {:.2f}".format(map_k))

# 1차원 리스트에 대한 nDCG
def get_ndcg(y_true, y_pred, k):
    ndcg = 0
    ranking = []
    
    for i in range(len(y_pred)):
        if y_pred[i] in y_true:
            ranking.append(1)
        else:
            ranking.append(0)
        
    # Ideal ranking을 계산하기 위해 ranking을 내림차순으로 정렬한 별도의 리스트
    ideal_ranking = sorted(ranking, reverse=True)
    
    # DCG 계산
    dcg = ranking[0]
    for i in range(1, min(k, len(ranking))):
        dcg += ranking[i] / np.log2(i + 1)
    
    # Ideal DCG 계산
    ideal_dcg = ideal_ranking[0]
    for i in range(1, min(k, len(ideal_ranking))):
        ideal_dcg += ideal_ranking[i] / np.log2(i + 1)
    
    # nDCG 계산
    if ideal_dcg == 0:
        ndcg = 0
    else:
        ndcg = dcg / ideal_dcg
    
    return ndcg

# 2차원 리스트에 대한 nDCG
def ndcg_at_k(y_true, y_pred, k):
    ndcg = 0
    cnt = 0
    
    # 기존 플레이리스트와 추천 플레이리스트를 1:1로 비교
    for true_item, pred_item in zip(y_true, y_pred):
        if len(true_item) == 0 or len(pred_item) == 0:
            continue
        ndcg += get_ndcg(true_item, pred_item, k)
        cnt += 1
    
    # 모든 플레이리스트가 비어있을 경우 계산 불가능히므로 0을 반환
    if len(y_true) == 0:
        return 0
    else:
        return ndcg / cnt

def song_recommend(tags, songs, song_df, sim, tag_imb_mode, genre_imb_mode, like_weight, emb_mode, weight_mode, like_mode, w2v, weight_mat_cv, weight_mat_tf, tag_weights):
    ts = tags
    tag_simi_songs = []
    
    sorted_idx = get_tag_mean(ts, song_df['tags'], tag_imb_mode, w2v, tag_weights)
    tag_simi_songs = song_df.loc[sorted_idx]
    tag_songs = tag_simi_songs.iloc[:1000].copy()
    tag_songs = tag_songs.reset_index()
    
    # 기존 노래(히스토리)가 있는 경우 장르 유사도를 계산해
    #상위 100개의 노래를 찾아낸다
    if len(songs) > 0:
        if genre_imb_mode:
            w_mat = apply_genre_weight(get_embedding(tag_songs, emb_mode), weight_mode)
            simi_songs = find_sim_song(tag_songs, sim, w_mat,weight_mat_cv, weight_mat_tf, songs, emb_mode, weight_mode, like_mode, False, like_weight, 100)
                
        else:
            simi_songs = find_sim_song(tag_songs, sim, get_embedding(tag_songs, emb_mode), weight_mat_cv, weight_mat_tf, songs, emb_mode, weight_mode, like_mode, False, like_weight, 100)
    
    # 기존 노래(히스토리)가 없는 경우 최신 노래(2018~2023년도)를 찾아낸다
    else:
        simi_songs = tag_songs
        simi_songs = simi_songs[simi_songs['issue_date'] > 20180000]
        simi_songs = simi_songs[simi_songs['issue_date'] < 20240000]
    
    # 태그로 만들어낸 플레이리스트와 장르 유사도로 만들어낸 노래 목록
    # 둘 모두에 존재하는 노래 10개 추출한다
    recommended = []
    
    for rec in simi_songs['song_id']:
        title_song = tag_songs[tag_songs['song_id'] == rec]
        if not title_song.empty:
            title_index = title_song.index
            recommended.append(title_index[0])
    
    return tag_songs.iloc[recommended[:10]]

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
    
    sorted_idx = get_tag_mean(ts, simi_songs['tags'], tag_imb_mode, w2v, tag_weights)
    tag_simi_mean = simi_songs.loc[sorted_idx]
    
    return tag_simi_mean.iloc[:10]

def process_fuc(rec_num, test_my_songs, test_my_tags, song_tag_appended, simi_mode, tag_imb, genre_imb, like, mode, weight_mode, like_mode, tag_mode, w2v, weight_mat_cv, weight_mat_tf, tag_weights, tag_weights_all):
    pred_list = []
    if rec_num == 1:
        for plist, tags in zip(test_my_songs, test_my_tags):
            if tag_mode == 0:
                recommended = song_recommend(tags, plist, song_tag_appended, simi_mode, tag_imb, genre_imb, like, mode, weight_mode, like_mode, w2v,
                                     weight_mat_cv, weight_mat_tf, tag_weights)
            elif tag_mode == 1:
                recommended = song_recommend(tags, plist, song_tag_appended, simi_mode, tag_imb, genre_imb, like, mode, weight_mode, like_mode, w2v,
                                     weight_mat_cv, weight_mat_tf, tag_weights_all)
            
            pred_list.append(recommended['song_id'].tolist())
    elif rec_num == 2:
        for plist, tags in zip(test_my_songs, test_my_tags):
            if tag_mode == 0:
                recommended = song_recommend2(tags, plist, song_tag_appended, simi_mode, tag_imb, genre_imb, like, mode, weight_mode, like_mode, w2v,
                                     weight_mat_cv, weight_mat_tf, tag_weights)
            elif tag_mode == 1:
                recommended = song_recommend2(tags, plist, song_tag_appended, simi_mode, tag_imb, genre_imb, like, mode, weight_mode, like_mode, w2v,
                                     weight_mat_cv, weight_mat_tf, tag_weights_all)
            
            pred_list.append(recommended['song_id'].tolist())
    else:
        print("Process Error")
        return
    
    map_k = get_map_k(test_my_songs, pred_list, 10)
    ndcg = ndcg_at_k(test_my_songs, pred_list, 10)
    res = f"Recommend Model : {rec_num}" + f", Simi_mode : {simi_mode}" + f", Tag : {tag_imb}" + f", Genre : {genre_imb}" + f", Like : {like}" + ", Mode : " + mode +  ", Genre Weight Mode : " + str(weight_mode) +  ", Tag Weight Mode : " + str(tag_mode) + ", Like Weight Mode : " + str(like_mode) + "\n" + "MAP@K (K=10): {:.3f}".format(map_k) + "\n" + "nDCG: {:.3f}".format(ndcg)
    return res

# input 예시
# recnum(1, 2) / simi_mode(cos, pea, jac) / 태그 / 장르 / 좋아요 / mode(cv/tf) / 장르 가중치 모드(0/1) / 좋아요 가중치 모드(0/1) / 태그 가중치 모드(0/1)
# (1, 'cos', False, False, False, 'cv', 0, 0, 0)
# Rec : 1, Simi_mode : cos, Tag : False, Genre : False, Like : False, Mode : cv, Genre Weight Mode : 0, Tag Weight Mode : 0, Like Weight Mode : 0