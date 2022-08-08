from django.contrib import admin
from django.urls import path
from myapp import views

urlpatterns = [
    path('', views.login),
    path('about/', views.about, name='about'),
    path('target/', views.target, name='target'),
]
