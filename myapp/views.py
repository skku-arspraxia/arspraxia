import os
import boto3
import project.settings
import pandas as pd
import json

from django.shortcuts import render, redirect
from django.db import connection
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse

#from myapp.ml_sa import SKKU_SENTIMENT

from django.contrib.auth import login as auth_login
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm

from .models import NLP_models

@csrf_exempt
def login(request):
        if request.method == "GET":
                if request.session.session_key != None:
                        return redirect('/data/?task=ner')     

        elif request.method == "POST":
                try:
                        # config
                        userid = project.settings.ADMIN_ACCESS_ID
                        userpw = project.settings.ADMIN_ACCESS_PW

                        # FORM
                        formid = request.POST["id"]
                        formpw = request.POST["pw"]
                        
                        if userid==formid and userpw==formpw:
                                request.session['key'] = userid
                                return redirect('/data/?task=ner')      
                        
                except:
                        connection.rollback()
                        
        return render(request, 'login.html')
        

def data(request):
        if logincheck(request):
                return redirect('/login/')

        s3c = boto3.client(
                's3',
                aws_access_key_id=project.settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=project.settings.AWS_SECRET_ACCESS_ID
        )

        s3r = boto3.resource(
                's3',
                aws_access_key_id=project.settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=project.settings.AWS_SECRET_ACCESS_ID
        )

        csvlist = []
        my_bucket = s3r.Bucket('arspraxiabucket')
        for my_bucket_object in my_bucket.objects.all():
                # Task 분류
                if my_bucket_object.key.split('/')[0] == request.GET["task"]:
                        # 파일 목록 출력
                        if len(my_bucket_object.key.split('.')) > 1:
                                csvlist.append(my_bucket_object.key)

        
        datapath = 'https://arspraxiabucket.s3.ap-northeast-2.amazonaws.com/sa/raw/goo.csv'
        df = pd.read_csv(datapath)
        
        json_records = df.reset_index().to_json(orient='records')
        data = []
        data = json.loads(json_records)

        """
        print("Bucket list")
        all_objects = s3c.list_objects(Bucket = 'arspraxiabucket')
        print(all_objects['Contents'])
        print(len(all_objects['Contents']))
        print("@@@@")

        for obj in all_objects['Contents']:
                print(obj['Key'])

        """
        #datapath = 'https://arspraxiabucket.s3.ap-northeast-2.amazonaws.com/sa/raw/goo.csv'
        #data = pd.read_csv(datapath).to_html(justify='center')
        #data = pd.read_csv(datapath).to_dict()

        #print(data_set)

        #for object in data['text']:
        #        print(object)

        #for i, row in data.iterrows():
        #        print(i, row)


        

        """
        # 다운로드
        prefix = '/sa'
        for object in bucket.objects.filter(Prefix = '/sa'):
                if object.key == prefix:
                        os.makedirs(os.path.dirname(object.key), exist_ok=True)
                        continue;
                bucket.download_file(object.key, object.key)
        """

        context = {
                "task" : request.GET["task"],
                "csvlist" : csvlist,
                "data" : data
        }

        return render(request, 'data.html', context)


def inference(request):
        if logincheck(request):
                return redirect('/login/')

        context = {
                "task" : request.GET["task"]
        }

        return render(request, 'inference.html', context)


def train(request):
        if logincheck(request):
                return redirect('/login/')


        """
        skku = SKKU_SENTIMENT()
        content = {
                'result' : skku("좋아요"),
                'device' : skku.device,
                'epochs' : skku.args.epochs
        }
        """

        context = {
                "task" : request.GET["task"]
        }

        return render(request, 'train.html', context)
        

def models(request):
        if logincheck(request):
                return redirect('/login/')

        # print(model_type)

        context = {
                "task" : request.GET["task"]
        }
        context['table_data'] = NLP_models.objects.filter(model_task=request.GET["task"])

        return render(request, 'models.html', context)


def logincheck(request):
        if request.session.session_key == None:
                return True


def dataSelectAjax(request):

        s3r = boto3.resource(
                's3',
                aws_access_key_id=project.settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=project.settings.AWS_SECRET_ACCESS_ID
        )

        datapath = ''
        datapath_url = 'https://arspraxiabucket.s3.ap-northeast-2.amazonaws.com/'
        data_src = request.GET["dataSrc"]
        datapath = datapath_url + data_src

        df = pd.read_csv(datapath)        
        json_records = df.reset_index().to_json(orient='records')
        data = []
        data = json.loads(json_records)

        context = {'data' : data}
        return JsonResponse(context)

