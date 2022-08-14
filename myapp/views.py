import os
import boto3
import project.settings
import pandas as pd
import json
import chardet

from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.core.paginator import Paginator
from django.db import connection
from .models import NLP_models

#from myapp.ml_sa import SKKU_SENTIMENT
from myapp.sa_test import SKKU_SENTIMENT

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
        
def logout(request):
        request.session.flush()
        return redirect('/login')
        

def data(request):
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
            filesrc = my_bucket_object.key.split('.')
            if len(filesrc) > 1:
                if filesrc[1] == "csv":
                    filepath = filesrc[0].split("/")
                    filetask = filepath[1]
                    filetype = filepath[2]
                    filename = filepath[3]

                    if filetask == request.GET.get("task"):
                        if filetype == "train":
                            csvlist.append(filename + "." + filesrc[1])
    
        context = {
            "task" : request.GET.get("task"),
            "csvlist" : csvlist
        }

        if request.GET.get("fileName"):
            datapath = ''
            datapath_url = 'https://arspraxiabucket.s3.ap-northeast-2.amazonaws.com/'
            data_src = "data/" + request.GET.get("task") + "/train/" + request.GET.get("fileName")
            datapath = datapath_url + data_src

            try:
                df = pd.read_csv(datapath, encoding="utf-8") 
            except:   
                df = pd.read_csv(datapath, encoding="cp949")       

            board_list = []
            for obj in df.values.tolist():
                board_list.append({'text':obj[0],'sentiment':obj[1]})
                
            page = request.GET.get('page', '1')
            paginator = Paginator(board_list, '10')
            page_obj = paginator.page(page)   

            page_numbers_range = 10
            max = len(paginator.page_range)
            current_page = int(page) if page else 1

            start = int((current_page - 1) / page_numbers_range) * page_numbers_range
            end = start + page_numbers_range
            if end >= max:
                    end = max

            context['fileName'] = request.GET.get("fileName")
            context['page_obj'] = page_obj
            context['page_range'] = paginator.page_range[start:end]

        return render(request, 'data.html', context)


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
        fileuploadname = "data/"+ task + "/train/" + filename
        s3c.upload_fileobj(
                file,
                'arspraxiabucket',
                fileuploadname
        )

        context = {
                "result" : "success"
        }

        return JsonResponse(context)


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
            filesrc = my_bucket_object.key.split('.')
            if len(filesrc) > 1:
                if filesrc[1] == "csv":
                    filepath = filesrc[0].split("/")
                    filetask = filepath[1]
                    filetype = filepath[2]
                    filename = filepath[3]

                    if filetask == request.GET["task"]:
                        if filetype == "train":
                            csvlist.append(filename + "." + filesrc[1])
                                
        context = {
                "task" : request.GET["task"],
                "csvlist" : csvlist,
                "inference_model" : NLP_models.objects.filter(model_task=request.GET["task"])
        }        

        return render(request, 'train.html', context)
        

def trainInsertAjax(request):
        if logincheck(request):
                return redirect('/login/')

        trainInsertAjax = NLP_models()
        trainInsertAjax.model_task = request.GET["task"]
        trainInsertAjax.model_name = request.GET["modelname"]
        trainInsertAjax.epoch = request.GET["modelepoch"]
        trainInsertAjax.learning_rate = request.GET["modellr"]
        trainInsertAjax.batch_size = request.GET["modelbs"]
        trainInsertAjax.description = request.GET["modeldes"]
        trainInsertAjax.accuracy = request.GET["modelacc"]
        trainInsertAjax.f1 = request.GET["modelf1"]
        trainInsertAjax.speed = request.GET["modelspeed"]
        trainInsertAjax.volume = request.GET["modelvolume"]
        trainInsertAjax.save()

        context = {
                'result' : 'success'
        }

        return JsonResponse(context)


def inference(request):
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
            filesrc = my_bucket_object.key.split('.')
            if len(filesrc) > 1:
                if filesrc[1] == "csv":
                    filepath = filesrc[0].split("/")
                    filetask = filepath[1]
                    filetype = filepath[2]
                    filename = filepath[3]

                    if filetask == request.GET["task"]:
                        if filetype == "inf":
                            csvlist.append(filename + "." + filesrc[1])

        context = {
                "task" : request.GET["task"],
                "csvlist" : csvlist,
                "inference_model" : NLP_models.objects.filter(model_task=request.GET["task"])
        }

        return render(request, 'inference.html', context)


def inferenceUpload(request):
        if logincheck(request):
                return redirect('/login/')

        context = {
                "task" : request.GET["task"]
        }

        return render(request, 'inferenceUpload.html', context)



def inferenceSA(request):
        if logincheck(request):
                return redirect('/login/')

        skku_sa = SKKU_SENTIMENT()

        sentence = "류도현 존나 잘생김"
        result = skku_sa(sentence)
        print("@@@@"+result)

        # 결과 json으로

        context = {
                "task" : request.GET["task"],
                #"result_json" : result_json
        }

        return render(request, 'inferenceSA.html', context)


def inferenceNER(request):
    pass


@csrf_exempt
def inferenceFileUploadAjax(request):
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
        fileuploadname = "data/" + task + "/inf/" + filename
        s3c.upload_fileobj(
                file,
                'arspraxiabucket',
                fileuploadname
        )

        context = {
                "result" : "success"
        }

        return JsonResponse(context)


def models(request):
        if logincheck(request):
                return redirect('/login/')

        board_list = list(NLP_models.objects.filter(model_task=request.GET["task"]))
        page = request.GET.get('page', '1')
        paginator = Paginator(board_list, '10')
        page_obj = paginator.page(page)


        context = {
                "task" : request.GET["task"],
                "page_obj" : page_obj
        }

        return render(request, 'models.html', context)


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
                        "description": model_input.description,
                
                }

                return render(request,'modelPopup.html',context)


def logincheck(request):
        if request.session.session_key == None:
                return True



"""
# 다운로드
prefix = '/sa'
for object in bucket.objects.filter(Prefix = '/sa'):
        if object.key == prefix:
                os.makedirs(os.path.dirname(object.key), exist_ok=True)
                continue;
        bucket.download_file(object.key, object.key)
"""