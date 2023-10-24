from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.core.paginator import Paginator

#추가
from .recommend_process import *
import os

def get_songs(my_songs, tag_list, sample_index):
    if len(my_songs) == 0:
        my_songs_df = []
        return my_songs_df
    train_data_appended = settings.TRAIN_DATA.copy()
    train_data_appended = train_data_appended.loc[sample_index]
    train_data_appended = train_data_appended.reset_index(drop=True)
    user_row = [tag_list, my_songs, 1]
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

    train_data_sample.drop_duplicates(subset='song_id', keep='first', inplace=True)

    for i in range(len(train_data_sample)):
        song = train_data_sample['song_id'].iloc[i]
        train_data_sample['tags'].iloc[i] = list(train_dict[song])

    song_data = settings.SONG_DATA.copy()
    song_data.rename(columns={'id': 'song_id', 'song_gn_dtl_gnr_basket': 'gnr'}, inplace=True)
    song_data = song_data.astype({'issue_date': 'int64'})
    song_tag_appended = pd.merge(train_data_sample, song_data)
    song_tag_appended = song_tag_appended.astype({'song_id': 'int64'})
    my_songs_df = song_tag_appended[song_tag_appended['song_id'].isin(my_songs)]
    print(my_songs_df)
    return my_songs_df.values.tolist()
#추가
@csrf_exempt
def clear_session(request):
    if 'tag_clear' in request.POST and 'tag_list' in request.session:
        del request.session['tag_list']
    elif 'song_clear' in request.POST and 'selected_song_ids' in request.session:
        del request.session['selected_song_ids']
        #request.session['selected_song_ids'] = [394031,195524,540149,287984,440773,100335,556301,655561,534818,695032,516602,521739,97057,703323,295250,25155,24275,273672,334095]
    elif 'sample_clear' in request.POST:
        train_data = settings.TRAIN_DATA
        temp_df = train_data.sample(n=100)
        print('==============New Samples================')
        print(temp_df)
        print('=========================================')
        request.session['samples'] = temp_df.reset_index().to_json(orient='records')
    song_list = request.session.get('selected_song_ids', [])
    #print(song_list)
    tag_list = request.session.get('tag_list', [])

    train_data_json = request.session.get('samples')
    train_data_sample = pd.read_json(train_data_json, orient='records')
    sample_index = train_data_sample['index'].to_list()
    my_songs_df = get_songs(song_list, tag_list, sample_index)
    return render(request, 'polls/recommend.html', {'my_tags': tag_list, 'my_songs': song_list})

def recommend(request):
    if request.method == 'POST':
        tag_list = request.session.get('tag_list', [])
        song_list = request.session.get('selected_song_ids', [])
        song_data = settings.SONG_DATA
        train_data_json = request.session.get('samples')
        train_data_sample = pd.read_json(train_data_json, orient='records')
        sample_index = train_data_sample['index'].to_list()
        selected_checkboxes = request.POST.getlist('checkboxes')
        if len(selected_checkboxes) == 0:
            print("Empty Checkbox")
        prepro_para = [0, False, False, 'cv']
        para_output = ""
        for checkbox in selected_checkboxes:
            if checkbox == 'kor':
                para_output += "한국어 전처리, "
                prepro_para[0] = 1
            elif checkbox == 'eng':
                para_output += "영어 전처리, "
                prepro_para[0] = 2
            elif checkbox == 'tag_imb':
                para_output += "태그 불균형 처리, "
                prepro_para[1] = True
            elif checkbox == 'gnr_imb':
                para_output += "장르 불균형 처리, "
                prepro_para[2] = True
            elif checkbox == 'gnr_cv_tf':
                para_output += "장르 임베딩 방식 전환(TF), "
                prepro_para[3] = 'tf'
            print(f'Selected checkbox: {checkbox}')
        if len(para_output) != 0:
            para_output = para_output[:len(para_output) - 2]
        #recommend_res = song_data[song_data['id'].isin(pred_list)].values.tolist()
        if prepro_para[0] == 1:
            pred_df = main_process_kor(sample_index, tag_list, song_list, prepro_para)
        elif prepro_para[0] == 2:
            pred_df = main_process_eng(sample_index, tag_list, song_list, prepro_para)
        else:
            pred_df = main_process_no(sample_index, tag_list, song_list, prepro_para)
        print(pred_df)
        recommend_res = pred_df.values.tolist()
        eval_list = []
        precision_k = get_precision_k(song_list, pred_df['song_id'].tolist())
        eval_list.append(precision_k)
        my_songs_df = get_songs(song_list, tag_list, sample_index)
        #임시
        #recommend_res = song_data[song_data['id'].isin(song_list)].values.tolist()

    return render(request, 'polls/recommend.html', {'recommend_res': recommend_res, 'my_tags': tag_list, 'my_songs_df': my_songs_df, 'eval_list': eval_list, 'para_output': para_output})

@csrf_exempt
def recommend_music(request):
    tag_list = request.session.get('tag_list', [])
    song_list = request.session.get('selected_song_ids', [])

    train_data_json = request.session.get('samples')
    train_data_sample = pd.read_json(train_data_json, orient='records')
    sample_index = train_data_sample['index'].to_list()
    # 추가
    if request.method == 'POST':
        if 'tag_name' in request.POST:
            tag_name = request.POST.get('tag_name', '').split(',')
            tag_list = request.session.get('tag_list', [])
            tag_list.extend(tag_name)
            request.session['tag_list'] = list(set(tag_list))
            print(request.session.get('tag_list', []))
            my_songs_df = get_songs(song_list, tag_list, sample_index)
            return render(request, 'polls/recommend.html', {'my_tags': tag_list, 'my_songs_df': my_songs_df})


        song_name = request.POST.get('song_name', '')
        # 노래 이름을 사용하여 검색 결과를 데이터베이스에서 가져옴
        song_data = settings.SONG_DATA
        search_results = song_data[song_data['song_name'].str.contains(song_name, case=False, na=False)]
        #search_results = search_results.to_json(orient='records')
        print(search_results)
        selected_song_ids = request.session.get('selected_song_ids', [])  # 기존 선택한 노래 ID 리스트를 가져옴
        search_results = search_results.values.tolist()

        if 'selected_song' in request.POST:
            selected_song_id = int(request.POST['selected_song'])
            if selected_song_id in selected_song_ids:
                return render(request, 'polls/recommend.html' , {'my_tags': tag_list, 'my_songs': song_list})

            # 선택한 노래 ID를 리스트에 추가

            selected_song_ids.append(selected_song_id)

            # 세션에 선택한 노래 ID 리스트를 저장
            request.session['selected_song_ids'] = selected_song_ids
            print(selected_song_ids)

            # 여기에서 선택한 노래 ID를 사용하여 추가 작업을 수행
            # 선택한 노래 ID(selected_song_id)를 활용하여 추천된 음악을 찾거나 처리
            song_list = request.session.get('selected_song_ids', [])
            my_songs_df = get_songs(song_list, tag_list, sample_index)
            return render(request, 'polls/recommend.html', {'my_tags': tag_list, 'my_songs_df': my_songs_df})
        my_songs_df = get_songs(song_list, tag_list, sample_index)
        return render(request, 'polls/recommend.html', {'search_results': search_results, 'my_tags': tag_list, 'my_songs_df': my_songs_df})
    my_songs_df = get_songs(song_list, tag_list, sample_index)
    return render(request, 'polls/recommend.html', {'my_tags': tag_list, 'my_songs_df': my_songs_df})