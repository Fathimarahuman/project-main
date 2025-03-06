"""socialmedia URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.views.generic import TemplateView
from socialmediaapp.views import *

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',index,name='index'),
    path('index/',index,name='index'),

    path('loginaction/',loginaction,name='loginaction'),
    path('login/',login,name='login'),

    path('userpage/',userpage,name='userpage'),
    path('user/',user,name='user'),
    path('uploaddataset/',uploaddataset,name='uploaddataset'),
    path('uploadaction/',uploadaction,name='uploadaction'),
    path('accuracy/',accuracy,name='accuracy'),
    path('profile/',profile,name='profile'),
    path('profileaction/',profileaction,name='profileaction'),

    
    path('Rating/',Rating,name='Rating'),
    path('Ratingaction/',Ratingaction,name='Ratingaction'),

    path('',useraction,name='useraction'),
    path('useraction/',useraction,name='useraction'),


]
