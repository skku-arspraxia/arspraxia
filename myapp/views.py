import os
import boto3
import project.settings
import pandas as pd
import shutil
import chardet
import urllib
import mimetypes
import schedule
import time
from django.http import HttpResponse
from urllib.parse import quote
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.core.paginator import Paginator
from django.db import connection
from .models import NLP_models

if project.settings.ISGPUON:
    from myapp.skku.ner.skku_ner_gpuon import SKKU_NER
    from myapp.skku.sa.skku_sa_gpuon import SKKU_SA
else:
    from myapp.skku.ner.skku_ner_gpuoff import SKKU_NER
    from myapp.skku.sa.skku_sa_gpuoff import SKKU_SA

s3r = boto3.resource(
    "s3",
    aws_access_key_id=project.settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=project.settings.AWS_SECRET_ACCESS_ID
)

s3c = boto3.client(
    "s3",
    aws_access_key_id=project.settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=project.settings.AWS_SECRET_ACCESS_ID
)

def logincheck(request):
    if request.session.session_key == None:
        return True


@csrf_exempt
def login(request):
    if request.method == "GET":
        if request.session.session_key != None:
            return redirect("/data/?task=ner")
    elif request.method == "POST":
        try:
            user_id = project.settings.ADMIN_ACCESS_ID
            user_pw = project.settings.ADMIN_ACCESS_PW
            form_id = request.POST["id"]
            form_pw = request.POST["pw"]
            
            if user_id==form_id and user_pw==form_pw:
                request.session["key"] = user_id
                return redirect("/data/?task=ner&data_type=train")                
        except:
            connection.rollback()                    
    return render(request, "login.html")

        
def logout(request):
    request.session.flush()
    return redirect("/login")
        

def data(request):
    if logincheck(request):
        return redirect("/login/")

    data_list = []
    task = request.GET.get("task")
    data_type = request.GET.get("data_type")

    if task == None:
        task = "ner"
    if data_type == None:
        data_type = "train"

    my_bucket = s3r.Bucket(project.settings.AWS_BUCKET_NAME)
    for my_bucket_object in my_bucket.objects.all():
        data_src = my_bucket_object.key.split(".")

        # Check if it is a file
        if len(data_src) > 1:
            data_extention = data_src[1]
            if data_extention == "csv" or data_extention == "tsv" or data_extention == "xls" or data_extention == "xlsx":
                data_path = data_src[0].split("/")
                if len(data_path) == 4:
                    if data_path[0] == "data" and data_path[1] == task and data_path[2] == data_type:
                        data_name = data_path[3]
                        data_list.append(data_name + "." + data_extention)

    context = {
        "task" : task,
        "data_type" : data_type,
        "data_list" : data_list,
        "page_title" : "Data",
        "page_no" : 1
    }

    # If a file is selected
    if request.GET.get("fileName"):
        file_name = request.GET.get("fileName")
        file_url = project.settings.AWS_URL
        file_path = "data/" + task + "/" + data_type + "/"
        file_src = file_url + file_path + quote(file_name)
        file_extention = file_name.split(".")[1]        

        if file_extention == "csv": 
            try:
                df = pd.read_csv(file_src, encoding="utf-8") 
            except:
                df = pd.read_csv(file_src, encoding="cp949")                  
        elif file_extention == "tsv":
            try:
                df = pd.read_csv(file_src, encoding="utf-8", delimiter="\t") 
            except:   
                df = pd.read_csv(file_src, encoding="cp949", delimiter="\t")   
        elif file_extention == "xls" or file_extention == "xlsx":
                df = pd.read_excel(file_src)

        board_list = []
        for board in df.values.tolist():
            if data_type == "train":
                board_list.append({"text":board[0], "classification":board[1]})
            elif data_type == "inf":
                board_list.append({"text":board[0]})
            elif data_type == "result":
                board_list.append({"text":board[0], "classification":board[1]})
            
        # board_list -> Paginator
        paginator = Paginator(board_list, "30")
        page = request.GET.get("page", "1")
        page_obj = paginator.page(page)
        page_numbers_range = 10
        max = len(paginator.page_range)
        current_page = int(page) if page else 1
        start = int((current_page - 1) / page_numbers_range) * page_numbers_range
        end = start + page_numbers_range
        start_idx = int(page_obj.paginator.per_page) * (int(page) - 1)
        if end >= max:
            end = max

        context["file_name"] = file_name
        context["page_obj"] = page_obj
        context["start_idx"] = start_idx
        context["page_range"] = paginator.page_range[start:end]

    return render(request, "data.html", context)


def train(request):
    if logincheck(request):
        return redirect("/login/")

    data_list = []
    task = request.GET.get("task")
    my_bucket = s3r.Bucket(project.settings.AWS_BUCKET_NAME)
    for my_bucket_object in my_bucket.objects.all():
        data_src = my_bucket_object.key.split(".")
        if len(data_src) > 1:
            data_extention = data_src[1]
            if data_extention == "csv" or data_extention == "tsv" or data_extention == "xls" or data_extention == "xlsx":
                data_path = data_src[0].split("/")
                if len(data_path) == 4:
                    if data_path[0] == "data":
                        data_task = data_path[1]    # ner/sa
                        data_type = data_path[2]    # train/inf/result
                        data_name = data_path[3]    # name

                        if data_task == task:
                            if data_type == "train":
                                data_list.append(data_name + "." + data_extention)
                            
    context = {
        "task" : task,
        "data_list" : data_list,
        "inference_model" : NLP_models.objects.filter(model_task=task),
        "page_title" : "Train",
        "page_no" : 2
    }
    return render(request, "train.html", context)
        

train_current_step = 0
train_current_epoch = 0
def trainGetStatusAjax(request):
    if logincheck(request):
        return redirect("/login/")

    global train_current_step
    global train_current_epoch
    
    context = {
        "train_current_step" : train_current_step,
        "train_current_epoch" : train_current_epoch
    }
    return JsonResponse(context)


def skku_sa_status(step, epoch):
    global train_current_step
    global train_current_epoch
    train_current_step = step
    train_current_epoch = epoch


def trainStartAjax(request):
    if logincheck(request):
            return redirect("/login/")
            
    params = {
        "pretrained_model" : request.GET.get("pretrained"),
        "train_data" : request.GET.get("dataSrc"),
        "modelepoch" : request.GET.get("modelepoch"),
        "modelbs" : request.GET.get("modelbs"),
        "modellr" : request.GET.get("modellr"),
    }

    global train_current_step
    global train_current_epoch
    train_current_step = 0
    train_current_epoch = 0
    tempCheck = False

    task = request.GET.get("task")
    if task == "sa":
        skku_sa = SKKU_SA()
        skku_sa.setTrainAttr(params)
        skku_sa.train()
        """
        schedule.every(1).seconds.do(skku_sa_status, skku_sa.getCurrentStep(), skku_sa.getCurrentEpoch())  
        while skku_sa.isTrainFinished() == False:
            schedule.run_pending()
            time.sleep(1)
            while tempCheck == False:
                skku_sa.setTrainAttr(params)
                skku_sa.train()
                tempCheck = True     
        """        
    elif task == "ner":
        skku_ner = SKKU_NER()
        skku_ner.setTrainAttr(params)
        skku_ner.train()   

    # DB 생성 및 저장
    trainStartAjax = NLP_models()
    trainStartAjax.model_task = task
    trainStartAjax.model_name = request.GET.get("modelname")
    trainStartAjax.epoch = request.GET.get("modelepoch")
    trainStartAjax.learning_rate = request.GET.get("modellr")
    trainStartAjax.batch_size = request.GET.get("modelbs")
    trainStartAjax.description = request.GET.get("modeldes")
    if task == "sa":
        trainStartAjax.precision = skku_sa.getPrecision()
        trainStartAjax.recall = skku_sa.getRecall()
        trainStartAjax.f1 = skku_sa.getF1score()
        trainStartAjax.volume = skku_sa.getModelsize()
    elif task == "ner":
        trainStartAjax.precision = skku_ner.getPrecision()
        trainStartAjax.recall = skku_ner.getRecall()
        trainStartAjax.f1 = skku_ner.getF1score()
        trainStartAjax.volume = skku_ner.getModelsize()
    trainStartAjax.save()

    context = {
        "result" : "success"
    }
    return JsonResponse(context)
        

def inference(request):
    if logincheck(request):
        return redirect("/login/")

    data_list = []
    task = request.GET.get("task")
    my_bucket = s3r.Bucket(project.settings.AWS_BUCKET_NAME)
    for my_bucket_object in my_bucket.objects.all():
        data_src = my_bucket_object.key.split(".")
        if len(data_src) > 1:
            data_extention = data_src[1]
            if data_extention == "csv" or data_extention == "tsv" or data_extention == "xls" or data_extention == "xlsx":
                data_path = data_src[0].split("/")
                if len(data_path) == 4:
                    if data_path[0] == "data":
                        data_task = data_path[1]    # ner/sa
                        data_type = data_path[2]    # train/inf/result
                        data_name = data_path[3]    # name

                        if data_task == task:
                            if data_type == "inf":
                                data_list.append(data_name + "." + data_extention)

    context = {
        "task" : task,
        "data_list" : data_list,
        "inference_model" : NLP_models.objects.filter(model_task=task),
        "page_title" : "Inference",
        "page_no" : 3
    }

    # Inference output file
    if request.GET.get("fileName"):
        file_name = request.GET.get("fileName")
        file_url = project.settings.AWS_URL
        file_path = "data/" + task + "/result/"
        file_src = file_url + file_path + quote(file_name)

        try:
            df = pd.read_csv(file_src, encoding="utf-8") 
        except:   
            df = pd.read_csv(file_src, encoding="cp949")    

        obj_index = 1
        board_list = []
        for board in df.values.tolist():        
            if task == "ner":
                board_list.append({"text":board[0], "tagtoken":zip(board[1].split(" "), board[0].split(" ")), "length":len(board[1].split(" ")), "index":obj_index })
                obj_index += 1
            elif task == "sa":
                board_list.append({"text":board[0], "classification":board[1], "score":"temp"})
            
        # board_list -> Paginator
        paginator = Paginator(board_list, "10")
        page = request.GET.get("page", "1")
        page_obj = paginator.page(page)
        page_numbers_range = 10
        max = len(paginator.page_range)
        current_page = int(page) if page else 1
        start = int((current_page - 1) / page_numbers_range) * page_numbers_range
        end = start + page_numbers_range
        start_idx = int(page_obj.paginator.per_page) * (int(page) - 1)
        if end >= max:
            end = max

        context["file_name"] = file_name
        context["page_obj"] = page_obj
        context["start_idx"] = start_idx 
        context["page_range"] = paginator.page_range[start:end]

    return render(request, "inference.html", context)
      

def inferenceStartAjax(request):
    if logincheck(request):
        return redirect("/login/")
        
    task = request.GET.get("task")    
    params = {
        "inference_data" : request.GET.get("dataSrc"),
        "inference_model" : request.GET.get("inference_model"),
    }

    result_file_name = ""
    if task == "sa":
        skku_sa = SKKU_SA()
        skku_sa.setInferenceAttr(params)
        skku_sa.inference()
        result_file_name = skku_sa.getResultFileName()
    elif task == "ner":
        skku_ner = SKKU_NER()
        skku_ner.setInferenceAttr(params)
        skku_ner.inference()
        result_file_name = skku_ner.getResultFileName()

    context = {
            "result" : "success",
            "result_file_name" : result_file_name
    }
    return JsonResponse(context)


def models(request):
    if logincheck(request):
        return redirect("/login/")

    task = request.GET.get("task")
    board_list = list(NLP_models.objects.filter(model_task=task))
    paginator = Paginator(board_list, "10")
    page = request.GET.get("page", "1")
    page_obj = paginator.page(page)
    page_numbers_range = 10
    max = len(paginator.page_range)
    current_page = int(page) if page else 1
    start = int((current_page - 1) / page_numbers_range) * page_numbers_range
    end = start + page_numbers_range
    start_idx = int(page_obj.paginator.per_page) * (int(page) - 1)
    if end >= max:
        end = max

    context = {
        "task" : task,
        "page_obj" : page_obj,
        "page_range" : paginator.page_range[start:end],
        "start_idx" : start_idx,
        "page_title" : "Models",
        "page_no" : 4
    }
    return render(request, "models.html", context)
    

def uploadFile(request):
    if logincheck(request):
        return redirect("/login/")

    task = request.GET.get("task")
    context = {
        "task" : task
    }
    return render(request, "uploadFile.html", context)


@csrf_exempt
def uploadFileAjax(request):
    if logincheck(request):
        return redirect("/login/")

    task = request.POST.get("task")
    data_type = request.POST.get("data_type")
    print(data_type)
    file_list = request.FILES.getlist("file")
    for file in file_list:
        file_name = file.name
        file_uploadname = "data/"+ task + "/" + data_type + "/" + file_name

        s3c.upload_fileobj(
            file,
            project.settings.AWS_BUCKET_NAME,
            file_uploadname
        )

    context = {
        "result" : "success"
    }
    return JsonResponse(context)

    
def downloadFile(request):
    task = request.GET.get("task")
    file_name = request.GET.get("fileName")
    data_type = request.GET.get("type")
    # AWS file path
    file_path = "data/" + task + "/" + data_type + "/"
    file_src = file_path + file_name

    # Local file path    
    local_file_path = os.path.join(project.settings.MEDIA_ROOT, file_path)
    if os.path.exists(local_file_path):
        shutil.rmtree(local_file_path)
        os.makedirs(local_file_path)
    else:
        os.makedirs(local_file_path)

    # Download from AWS
    local_file_src = local_file_path + file_name
    s3c.download_file(project.settings.AWS_BUCKET_NAME, file_src, local_file_src)
 
    if os.path.exists(local_file_src):
        file = open(local_file_src, "rb")
        response = HttpResponse(file.read(), content_type=mimetypes.guess_type(local_file_src)[0])
        #response["Content-Disposition"] = 'attachment; filename*=UTF-8\'\'%s' % urllib.parse.quote(file_name).encode('utf-8')
        #response = HttpResponse(file.read(), content_type="application/octet-stream; charset=utf-8")
        response["Content-Disposition"] = "attachment; filename=" + os.path.basename(local_file_src)
        return response
    else:
        return HttpResponse("<script>modalShow('파일 다운로드에 실패했습니다.');</script>")


def page404(request, exception):
    context = {}
    response = render(request, "404.html", context)
    response.status_code = 404
    return response


def page500(request):
    context = {}
    response = render(request, "500.html", context)
    response.status_code = 500
    return response