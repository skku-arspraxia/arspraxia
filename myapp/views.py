import os
import boto3
import project.settings
import pandas as pd
import json

from django.shortcuts import render, redirect
from django.db import connection
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .models import NLP_models

#from myapp.ml_sa import SKKU_SENTIMENT

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

        csvlist_train = []
        csvlist_inf = []
        my_bucket = s3r.Bucket('arspraxiabucket')
        for my_bucket_object in my_bucket.objects.all():
                task = my_bucket_object.key.split('/')[0]

                if task == request.GET["task"]:
                        if len(my_bucket_object.key.split('.')) > 1:
                                # 파일 목록 출력
                                if my_bucket_object.key.split('.')[1] == "csv":
                                        filetype = my_bucket_object.key.split('.')[0].split("/")[1]
                                        filename = my_bucket_object.key.split('.')[0].split("/")[2]
                                        
                                        if filetype == "train":
                                                csvlist_train.append(filename + ".csv")
                                        elif filetype == "inf":
                                                csvlist_inf.append(filename + ".csv")



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
                "csvlist_train" : csvlist_train,
                "csvlist_inf" : csvlist_inf
        }

        return render(request, 'data.html', context)


def train(request):
        if logincheck(request):
                return redirect('/login/')

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
                                

        context = {
                "task" : request.GET["task"],
                "csvlist" : csvlist
        }        

        return render(request, 'train.html', context)
        

def inference(request):
        if logincheck(request):
                return redirect('/login/')

        context = {
                "task" : request.GET["task"]
        }

        return render(request, 'inference.html', context)


def models(request):
        if logincheck(request):
                return redirect('/login/')

        context = {
                "task" : request.GET["task"]
        }
        context['table_data'] = NLP_models.objects.filter(model_task=request.GET["task"])

        return render(request, 'models.html', context)


def logincheck(request):
        if request.session.session_key == None:
                return True


def dataUpload(request):
        if logincheck(request):
                return redirect('/login/')

        context = {
                "task" : request.GET["task"]
        }

        return render(request, 'dataUpload.html', context)


@csrf_exempt
def dataFileUploadAjax(request):
        if logincheck(request):
                return redirect('/login/')

        s3c = boto3.client(
                's3',
                aws_access_key_id=project.settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=project.settings.AWS_SECRET_ACCESS_ID
        )

        file = request.FILES['file']
        filename = file.name
        task = request.POST["task"]
        fileuploadname = task + "/inf/" + filename
        s3c.upload_fileobj(
                file,
                'arspraxiabucket',
                fileuploadname
        )

        context = {
                "result" : "success"
        }

        return JsonResponse(context)

        
def dataSelectAjax(request):
        if logincheck(request):
                return redirect('/login/')

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
        data = json.loads(json_records)

        context = {
                'data' : data
        }

        return JsonResponse(context)


def modelPopup(request):
        if request.method == 'GET':
                
                model_input = NLP_models.objects.get(id=request.GET.get('id'))
                print(model_input.model_name)
                print(model_input.id)
                context = {
                        "model_name":model_input.model_name,
                        "model_id": model_input.id,
                        "model_task": model_input.model_task,
                        "epoch":model_input.epoch,
                        "batch_size": model_input.batch_size,
                        "learning_rate": model_input.learning_rate,
                        "accuracy": model_input.accuracy,
                        "f1":model_input.f1,
                        "speed":model_input.speed,
                        "volume":model_input.volume,
                        "date":model_input.date,
                        "desciption": model_input.description,
                
                }

                return render(request,'modelPopup.html',context)






