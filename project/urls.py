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
    path('dataDownloadAjax/', views.dataDownloadAjax),
    path('train/', views.train),
    path('trainStartAjax/', views.trainStartAjax),
    path('trainGetStatusAjax/', views.trainGetStatusAjax),
    path('inference/', views.inference),
    path('inferenceUpload/', views.inferenceUpload),
    path('inferenceFileUploadAjax/', views.inferenceFileUploadAjax),
    path('inferenceSA/', views.inferenceSA),
    path('inferenceStartAjax/', views.inferenceStartAjax),
    path('models/', views.models, name='models'),
    path('modelPopup/', views.modelPopup),

    path('tempmodeldown/', views.tempmodeldown),
    path('tempinference/', views.tempinference)
]
