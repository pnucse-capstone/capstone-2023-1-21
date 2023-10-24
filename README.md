[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/fnZ3vxy8)

### 1. 프로젝트 소개

수많은 데이터를 수집하고 가공하여 새로운 정보의 가치를 얻어내는 빅 데이터 시대를 뛰어넘어, 현재는 이러한 데이터를 기반으로 한 추천 시스템, 스마트 자동차와 같은 머신러닝 서비스 기술이 보편화되고 있습니다. 특히 특정 사용자가 관심을 가질만한 정보를 추천하는 추천 시스템은, 넷플릭스의 온라인 스트리밍 서비스나 멜론의 음악 스트리밍 서비스 등에서 필수적인 역할을 하고 있습니다. 하지만 낮은 추천 시스템의 성능은 사용자의 취향이 충분히 반영되지 않은 컨텐츠 제공을 유발할 수 있으며, 이는 사용자의 서비스 만족도 감소로 이어질 수 있습니다. 이러한 추천 시스템의 성능은 학습 데이터의 품질에 크게 의존하며, 데이터 전처리를 통해 머신 러닝 모델의 성능을 향상시킬 수 있습니다. 따라서 본 과제에서는 음악 추천 시스템을 대상으로 다양한 데이터 전처리 기법을 적용하고 성능을 비교하려 합니다.

본 졸업과제는 음악 추천 모델의 성능을 향상시키기 위한 효과적인 전처리 방법을 분석합니다. 음악 추천을 위한 플레이리스트 및 음악 데이터에 다양한 전처리를 적용하고, 각 전처리 방식에 따른 추천 시스템의 성능을 비교합니다. 플레이리스트 및 음악 데이터의 태그 및 장르는 자연어 처리, 불균형 데이터 처리 등 다양한 전처리를 통해 각 전처리 조합 별 데이터셋을 생성합니다. 이후 이러한 태그 및 장르 텍스트 데이터를 기계가 이해하고 처리할 수 있는 벡터 형태로 변환합니다. 본 과제의 음악 추천 모델은 앞서 생성한 두 벡터 값의 유사도를 계산하여 가장 유사한 10곡의 음악을 추천하고, 비지도 학습 평가 지표를 통해 모델의 성능을 평가합니다. 이 과정을 각 전처리 조합 및 유사도 측정 방식에 따라 서로 다른 조건으로 수행하여 그 결과 및 모델 성능을 비교합니다. 이를 통해 음악 데이터 전처리 별 효율성을 비교하고 음악 추천 모델의 성능을 보편적으로 향상시킬 수 있는 전처리 방식을 찾아내는 것이 본 과제의 최종 목표입니다.

### 2. 팀소개

진현, emil5322@pusan.ac.kr, 개발총괄

임우영, pigwoo98@pusan.ac.kr, 알고리즘 설계

김영수, kimyeong3732@pusan.ac.kr, 백앤드 개발

### 3. 시스템 구성도

![음악 추천 모델 구상도](https://ifh.cc/g/V4Ahj9.jpg)

### 4. 소개 및 시연 영상

[![부산대학교 정보컴퓨터공학부 소개](http://img.youtube.com/vi/zh_gQ_lmLqE/0.jpg)](https://youtu.be/zh_gQ_lmLqE)
추후 업데이트

### 5. 설치 및 사용법

```song_meta_with_likes.json```파일은 용량 제한으로 인해, 외부에서 다운받아 아래 지정한 경로에 추가해야합니다. [데이터셋 다운](https://drive.google.com/file/d/1762ZT67g2ZibxA3dl69tdnk4NsW3hEjQ/view?usp=drive_link)
- ```Recommend_Model``` : ```./Recommend_Model/Datasets/```
- ```Music_Recommend_Web``` : ```./Music_Recommend_Web/polls/Datasets/```

### 5.1 웹 페이지 및 실험 (Music_Recommend_Web)
1. 웹 페이지 실행을 위해 PyCharm에서 Django를 설치해야합니다. [How to Download Django](https://docs.djangoproject.com/ko/4.2/intro/install/)
2. PyCharm에서 프로젝트를 생성합니다.
3. PyCharm에서 추가적으로 설치해야 할 라이브러리 목록입니다.
    - ```pandas```
    - ```gensim```
    - ```scikit_learn```
    - ```google_trans```
    - ```nltk```
    - ```konlpy```
4. PyCharm의 ```./Music_Recommend_Web/``` 경로에서 ```python manage.py runserver``` 명령어를 입력해 서버를 실행합니다.
5. 웹에서 ```127.0.0.1:8000/polls/```을 입력해 음악 추천 페이지에 접근합니다.
