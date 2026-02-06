from django.urls import path
from . import views

urlpatterns = [
    path('', views.api_root, name='api_root'),
    path('translate/', views.translate, name='translate'),
]