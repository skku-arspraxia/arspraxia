from django.urls import path
from myapp import views
from django.conf.urls.static import static
from django.conf.urls import (handler404, handler500)
import project.settings

handler404 = 'myapp.views.page404'
handler500 = 'myapp.views.page500'

urlpatterns = [
    path('', views.login),
    path('login/', views.login),
    path('logout/',views.logout),
    path('data/', views.data),
    path('train/', views.train),
    path('trainStartAjax/', views.trainStartAjax),
    path('trainGetStatusAjax/', views.trainGetStatusAjax),
    path('inference/', views.inference),
    path('inferenceStartAjax/', views.inferenceStartAjax),
    path('models/', views.models),
    path('uploadFile/', views.uploadFile),
    path('uploadFileAjax/', views.uploadFileAjax),
    path("downloadFile/", views.downloadFile)
]
urlpatterns += static(project.settings.MEDIA_URL, document_root=project.settings.MEDIA_ROOT)
