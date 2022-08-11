from django.contrib import admin
from django.urls import path
from myapp import views

urlpatterns = [
    path('', views.login),
    path('login/', views.login),
    path('data/', views.data),
    path('inference/', views.inference),
    path('train/', views.train),
    # path('model/analyze/<model_type>', views.model_analyze, name='analyze'),
    path('models/', views.models, name='models'),
]
