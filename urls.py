from django.urls import path
from . import views
urlpatterns=[
    path('fakeornot',views.fakeornot,name='fakeornot'),
    path('',views.home,name='home')
]