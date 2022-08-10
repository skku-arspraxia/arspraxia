from django.shortcuts import render, redirect
from django.db import connection
from django.views.decorators.csrf import csrf_exempt

from myapp.ml_sa import SKKU_SENTIMENT

from django.contrib.auth import login as auth_login
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm

import boto3
import project.settings

#from .models import Member

@csrf_exempt
def login(request):
        if request.method == "GET":
                if request.session.session_key != None:
                        return redirect('/data/')     

        elif request.method == "POST":
                try:
                        cursor = connection.cursor()
                        query = "SELECT * FROM member"
                        result = cursor.execute(query)
                        item = cursor.fetchall()
                        connection.commit()
                        connection.close()

                        # DB
                        userid = item[0][0]
                        userpw = item[0][1]

                        # FORM
                        formid = request.POST["id"]
                        formpw = request.POST["pw"]
                        
                        if userid==formid and userpw==formpw:
                                request.session['key'] = userid
                                return redirect('/data/')      
                        
                except:
                        connection.rollback()
                        
        return render(request, 'login.html')


def data(request):
        if logincheck(request):
                return redirect('/login/')
        """
        s3r = boto3.resource(
                's3',
                aws_access_key_id=project.settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=project.settings.AWS_SECRET_ACCESS_ID
        )
        data = open('/home/user/Downloads/'+'36059_46941_4950'+'.jpg','rb')
        s3r.Bucket('arspraxiabucket').put_obejct(Key='second', body=data, ContentType='jpg')
        
        s3_client = boto3.client(
                's3',
                aws_access_key_id=project.settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=project.settings.AWS_SECRET_ACCESS_ID
        )
        s3_client.upload_fileobj(

        )
        """
        return render(request, 'data.html')


def model_train(request):
        if logincheck(request):
                return redirect('/login/')

        skku = SKKU_SENTIMENT()
        content = {
                'result' : skku("좋아요"),
                'device' : skku.device,
                'epochs' : skku.args.epochs
        }

        return render(request, 'model_train.html')


def model_analyze(request):
        if logincheck(request):
                return redirect('/login/')

        return render(request, 'model_analyze.html')


def logincheck(request):
        if request.session.session_key == None:
                return True


def about(request):
        return render(request, 'about.html')

@csrf_exempt
def target(request):
        #content = Project.objects.all()
        return render(request, 'target.html',{
         #       "content" : content
        })
