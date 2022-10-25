from django.db import models

class NLP_models (models.Model):
    id = models.AutoField(primary_key=True)
    model_name = models.CharField(max_length=45)
    model_task = models.CharField(max_length=45)
    epoch = models.IntegerField(null=False, blank=False)
    batch_size = models.IntegerField(null=False, blank=False)
    learning_rate = models.FloatField(max_length=45, null=False, blank=False)
    accuracy = models.FloatField(max_length=45, null=False, blank=False)
    f1 = models.FloatField(max_length=45, null=False, blank=False)
    speed = models.FloatField(max_length=45, null=False, blank=False)
    volume = models.FloatField(max_length=45, null=False, blank=False)
    date = models.DateTimeField(auto_now_add=True)
    description = models.TextField(help_text='ex) pretrained-model, used data-set etc...')

    class Meta:
        db_table = 'NLP_Models'