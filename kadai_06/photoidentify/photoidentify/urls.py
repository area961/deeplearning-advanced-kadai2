from django.urls import path
from prediction.views import predict

urlpatterns = [
    path('', predict, name='predict'),
]
