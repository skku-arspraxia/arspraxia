from django.db import models

# Create your models here.
class Member(models.Model):
    userid = models.CharField(max_length=20, db_column='userid')
    userpw = models.CharField(max_length=20, db_column='userpw')

    class Meta:
        db_table = 'member'

        