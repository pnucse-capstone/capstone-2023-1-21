from django.urls import path

from . import views

urlpatterns = [
    path('', views.recommend_music, name='recommend_music'),
    path('clear_session', views.clear_session, name='clear_session'),
    path('recommend', views.recommend, name='recommend'),
]