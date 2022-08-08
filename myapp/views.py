from django.shortcuts import render, redirect
from django.db import connection
from django.views.decorators.csrf import csrf_exempt

from django.contrib.auth import login as auth_login
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm

#from .models import Member

def logincheck(request):
        if request.session.session_key == None:
                return True


@csrf_exempt
def login(request):
        if request.method == "GET":
                if request.session.session_key != None:
                        return redirect('/home/')     

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
                                return redirect('/home/')      
                        
                except:
                        connection.rollback()
                        
        return render(request, 'login.html')


def home(request):
        if logincheck(request):
                return redirect('/login/')

        return render(request, 'home.html')


def about(request):
        return render(request, 'about.html')

@csrf_exempt
def target(request):
        #content = Project.objects.all()
        return render(request, 'target.html',{
         #       "content" : content
        })
