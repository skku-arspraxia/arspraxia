from django.contrib import admin
from django.urls import path
from bootapp import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index),
    path('about/', views.about, name='about'),
    path('target/', views.target, name='target'),
]
