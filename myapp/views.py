import os
import boto3
import project.settings
import pandas as pd
import json
import shutil

from urllib.parse import quote
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.core.paginator import Paginator
from django.db import connection
from .models import NLP_models

from myapp.ml_sa import SKKU_SENTIMENT

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

        datalist = []
        my_bucket = s3r.Bucket('arspraxiabucket')
        for my_bucket_object in my_bucket.objects.all():
            filesrc = my_bucket_object.key.split('.')
            # 파일 여부 확인
            if len(filesrc) > 1:
                if filesrc[1] == "tsv" or filesrc[1] == "csv" or filesrc[1] == "xls" or filesrc[1] == "xlsx":
                    filepath = filesrc[0].split("/")
                    if len(filepath) == 4:
                        if filepath[0] == "data":
                            filetask = filepath[1]
                            filetype = filepath[2]
                            filename = filepath[3]

                            if filetask == request.GET.get("task"):
                                if filetype == "train":
                                    datalist.append(filename + "." + filesrc[1])

        context = {
            "task" : request.GET.get("task"),
            "datalist" : datalist
        }

        # 파일 조회 일 경우
        if request.GET.get("fileName"):
            datapath = ''
            datapath_url = 'https://arspraxiabucket.s3.ap-northeast-2.amazonaws.com/'
            data_src = "data/" + request.GET.get("task") + "/train/" + quote(request.GET.get("fileName"))
            datapath = datapath_url + data_src

            fileExtention = request.GET.get("fileName").split(".")[1]

            if fileExtention == "tsv":
                try:
                    df = pd.read_csv(datapath, encoding="utf-8", delimiter='\t') 
                except:   
                    df = pd.read_csv(datapath, encoding="cp949", delimiter='\t')   

            elif fileExtention == "csv": 
                try:
                    df = pd.read_csv(datapath, encoding="utf-8") 
                except:   
                    df = pd.read_csv(datapath, encoding="cp949")    
                     
            elif fileExtention == "xls" or fileExtention == "xlsx":
                    df = pd.read_excel(datapath)   

            board_list = []
            for obj in df.values.tolist():
                board_list.append({'text':obj[0], 'classification':obj[1]})
                
            page = request.GET.get('page', '1')
            paginator = Paginator(board_list, '30')
            page_obj = paginator.page(page)   

            page_numbers_range = 10
            max = len(paginator.page_range)
            current_page = int(page) if page else 1

            start = int((current_page - 1) / page_numbers_range) * page_numbers_range
            end = start + page_numbers_range
            if end >= max:
                    end = max

            startIdx = int(page_obj.paginator.per_page) * (int(page) - 1)

            context['fileName'] = request.GET.get("fileName")
            context['page_obj'] = page_obj
            context['page_range'] = paginator.page_range[start:end]
            context['startIdx'] = startIdx

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

        filelist = request.FILES.getlist('file')

        for file in filelist:
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


@csrf_exempt
def dataDownloadAjax(request):
        if logincheck(request):
                return redirect('/login/')

        jsonObj = json.loads(request.body)
        task = jsonObj.get('task', False)
        fileName = jsonObj.get('fileName', False)
        filePath = 'data/'+task+'/train/'
        fileSrc = filePath+fileName
        
        localrootPath = "C:/arspraxiabucket/"
        localfilePath = localrootPath+filePath
        localfileSrc = localfilePath+fileName
        if not os.path.exists(localfilePath):
           os.makedirs(localfilePath)
        
        s3c = boto3.client(
                's3',
                aws_access_key_id=project.settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=project.settings.AWS_SECRET_ACCESS_ID
        )

        s3c.download_file('arspraxiabucket', fileSrc, localfileSrc)

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

        datalist = []
        my_bucket = s3r.Bucket('arspraxiabucket')
        for my_bucket_object in my_bucket.objects.all():
            filesrc = my_bucket_object.key.split('.')
            if len(filesrc) > 1:
                if filesrc[1] == "tsv" or filesrc[1] == "csv" or filesrc[1] == "xls" or filesrc[1] == "xlsx":
                    filepath = filesrc[0].split("/")
                    if len(filepath) == 4:
                        if filepath[0] == "data":
                            filetask = filepath[1]
                            filetype = filepath[2]
                            filename = filepath[3]

                            if filetask == request.GET["task"]:
                                if filetype == "train":
                                    datalist.append(filename + "." + filesrc[1])
                                
        context = {
                "task" : request.GET["task"],
                "datalist" : datalist,
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

        datalist = []
        my_bucket = s3r.Bucket('arspraxiabucket')
        for my_bucket_object in my_bucket.objects.all():
            filesrc = my_bucket_object.key.split('.')
            if len(filesrc) > 1:
                if filesrc[1] == "tsv" or filesrc[1] == "csv" or filesrc[1] == "xls" or filesrc[1] == "xlsx":
                    filepath = filesrc[0].split("/")
                    if len(filepath) == 4:
                        if filepath[0] == "data":
                            filetask = filepath[1]
                            filetype = filepath[2]
                            filename = filepath[3]

                            if filetask == request.GET["task"]:
                                if filetype == "inf":
                                    datalist.append(filename + "." + filesrc[1])

        context = {
                "task" : request.GET["task"],
                "datalist" : datalist,
                "inference_model" : NLP_models.objects.filter(model_task=request.GET["task"])
        }

        # 220817 하계발표용 임시저장
        datapath = ''
        datapath_url = 'https://arspraxiabucket.s3.ap-northeast-2.amazonaws.com/'

        if request.GET["task"] == "ner":
            data_src = "data/" + request.GET.get("task") + "/result/" + quote("개체명인식결과.csv")
        elif request.GET["task"] == "sa":
            data_src = "data/" + request.GET.get("task") + "/result/" + quote("감성분석결과.csv")

        datapath = datapath_url + data_src

        try:
            df = pd.read_csv(datapath, encoding="utf-8") 
        except:   
            df = pd.read_csv(datapath, encoding="cp949")    

        board_list = []

        """
        filter1 = request.GET.get("filter1")
        filter2 = request.GET.get("filter2")
        """
        objIndex = 1
        for obj in df.values.tolist():        
            if request.GET["task"] == "ner":
                board_list.append({'text':obj[0], 'tagtoken':zip(obj[1].split(" "), obj[0].split(" ")), 'length':len(obj[1].split(" ")), 'index':objIndex })
                objIndex += 1
            elif request.GET["task"] == "sa":
                board_list.append({'text':obj[0], 'classification':obj[1], 'score':obj[2]})

                """
                if filter2:
                    if filter1 == ">":
                        if obj[2] > float(filter2):
                            board_list.append({'text':obj[0], 'classification':obj[1], 'score':obj[2]})
                    elif filter1 == ">=":
                        if obj[2] >= float(filter2):
                            board_list.append({'text':obj[0], 'classification':obj[1], 'score':obj[2]})
                    elif filter1 == "<":
                        if obj[2] < float(filter2):
                            board_list.append({'text':obj[0], 'classification':obj[1], 'score':obj[2]})
                    elif filter1 == "<=":
                        if obj[2] <= float(filter2):
                            board_list.append({'text':obj[0], 'classification':obj[1], 'score':obj[2]})
                else:
                    board_list.append({'text':obj[0], 'classification':obj[1], 'score':obj[2]})
                """

            
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

        startIdx = int(page_obj.paginator.per_page) * (int(page) - 1)

        context['page_obj'] = page_obj
        context['page_range'] = paginator.page_range[start:end]
        context['startIdx'] = startIdx 
        context['page'] = page
        """
        context['filter1'] = filter1
        context['filter2'] = filter2
        """
        # 여기까지

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

        return render(request, 'inferenceSA.html')


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

        filelist = request.FILES.getlist('file')

        for file in filelist:
            filename = file.name
            task = request.POST["task"]
            fileuploadname = "data/"+ task + "/inf/" + filename

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

        page_numbers_range = 10
        max = len(paginator.page_range)
        current_page = int(page) if page else 1

        start = int((current_page - 1) / page_numbers_range) * page_numbers_range
        end = start + page_numbers_range
        if end >= max:
                end = max

        page_range = paginator.page_range[start:end]

        startIdx = int(page_obj.paginator.per_page) * (int(page) - 1)

        context = {
                "task" : request.GET["task"],
                "page_obj" : page_obj,
                "page_range" : page_range,
                "startIdx" : startIdx
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


@csrf_exempt
def tempmodeldown(request):
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

        localrootPath = "C:/arspraxiabucket/"

        modelsrcList = []
        my_bucket = s3r.Bucket('arspraxiabucket')
        for my_bucket_object in my_bucket.objects.all():
            filesrc = my_bucket_object.key.split('.')
            
            if not os.path.exists(localrootPath+'model/1/'):
                os.makedirs(localrootPath+'model/1/')

            # 파일 여부 확인
            if len(filesrc) > 1:
                filepath = filesrc[0].split("/")
                if filepath[0] == "model":
                    modelsrcList.append(filesrc[0] + "." + filesrc[1])

        for modelsrc in modelsrcList:
            s3c.download_file('arspraxiabucket', modelsrc, localrootPath+modelsrc)

        context = {
                "result" : "success"
        }

        return JsonResponse(context)


@csrf_exempt
def tempinference(request):
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

    localrootPath = "C:/arspraxiabucket/"

    modelsrcList = []
    my_bucket = s3r.Bucket('arspraxiabucket')
    for my_bucket_object in my_bucket.objects.all():
        filesrc = my_bucket_object.key.split('.')
        
        if not os.path.exists(localrootPath+'model/1/'):
            os.makedirs(localrootPath+'model/1/')

        # 파일 여부 확인
        if len(filesrc) > 1:
            filepath = filesrc[0].split("/")
            if filepath[0] == "model":
                modelsrcList.append(filesrc[0] + "." + filesrc[1])

    for modelsrc in modelsrcList:
        s3c.download_file('arspraxiabucket', modelsrc, localrootPath+modelsrc)


    skku_sa = SKKU_SENTIMENT()

    sentence = "테스트가 잘됐으면 좋겠습니다."
    result = skku_sa(sentence)
    print("샘플 문장 : "+sentence)
    print("샘플 결과 : "+result)

    # 받은 임시 모델파일 삭제
    shutil.rmtree(localrootPath+'model/1')

    context = {
        "result" : result
    }

    return JsonResponse(context)

    