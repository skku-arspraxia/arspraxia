from django.shortcuts import render
from .models import Project
from django.views.decorators.csrf import csrf_exempt

def login(request):
        return render(request, 'login.html')


def home(request):
        content = Project.objects.all()
        return render(request, 'home.html',{
                "content" : content
        })


def about(request):
        return render(request, 'about.html')

def index(request):
        content = Project.objects.all()
        # skku = SKKU_SENTIMENT()
        return render(request, 'index.html',{
                "content" : content
        })

@csrf_exempt
def target(request):
        content = Project.objects.all()
        return render(request, 'target.html',{
                "content" : content
        })
