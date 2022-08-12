from django.contrib import admin
from django.urls import path
from myapp import views

urlpatterns = [
    path('', views.login),
    path('login/', views.login),
    path('data/', views.data),
    path('train/', views.train),
    path('inference/', views.inference),
    path('models/', views.models, name='models'),
    
    path('dataUpload/', views.dataUpload),
    path('dataSelectAjax/', views.dataSelectAjax),
    path('modelPopup/',views.modelPopup),
]
