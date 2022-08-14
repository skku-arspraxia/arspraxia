from django.contrib import admin
from django.urls import path
from myapp import views

urlpatterns = [
    path('', views.login),
    path('login/', views.login),
    path('logout/',views.logout),
    path('data/', views.data),
    path('dataUpload/', views.dataUpload),
    path('dataFileUploadAjax/', views.dataFileUploadAjax),
    path('dataSelectAjax/', views.dataSelectAjax),
    path('train/', views.train),
    path('trainInsertAjax/', views.trainInsertAjax),
    path('inference/', views.inference),
    path('inferenceUpload/', views.inferenceUpload),
    path('inferenceFileUploadAjax/', views.inferenceFileUploadAjax),
    path('inferenceSA/', views.inferenceSA),
    path('models/', views.models, name='models'),
    path('modelPopup/',views.modelPopup),
]
