from django.contrib import admin
from django.urls import path
from myapp import views

urlpatterns = [
    path('', views.login),
    path('login/', views.login),
    path('data/', views.data),
    path('inference/', views.inference),
    path('model/train/', views.model_train),
    path('model/analyze/', views.model_analyze),
    path('about/', views.about, name='about'),
    path('target/', views.target, name='target'),
]
