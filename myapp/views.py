from django.shortcuts import render
from .models import Project
from django.views.decorators.csrf import csrf_exempt
#from bootapp.skkuu import SKKU_SENTIMENT

# Create your views here.

def home(request):
        content = Project.objects.all()
        # skku = SKKU_SENTIMENT()
        return render(request, 'home.html',{
                "content" : content,
		#'skku_device' : skku.device,
		#'skku_epochs' : skku.args.epochs
        })


def about(request):
        return render(request, 'about.html')

def index(request):
        content = Project.objects.all()
        # skku = SKKU_SENTIMENT()
        return render(request, 'index.html',{
                "content" : content,
		#'skku_device' : skku.device,
		#'skku_epochs' : skku.args.epochs
        })
@csrf_exempt
def target(request):
        content = Project.objects.all()
        # skku = SKKU_SENTIMENT()
        return render(request, 'target.html',{
                "content" : content
		#'skku_device' : skku.device,
		#'skku_epochs' : skku.args.epochs
        })
