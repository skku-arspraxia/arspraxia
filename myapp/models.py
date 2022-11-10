from django.db import models

class NLP_models (models.Model):
    id = models.AutoField(primary_key=True)
    model_name = models.CharField(max_length=45, null=False)
    model_task = models.CharField(max_length=45, null=False)
    epoch = models.IntegerField(blank=False, null=False)
    batch_size = models.IntegerField(blank=False, null=False)
    learning_rate = models.DecimalField(max_digits=20, decimal_places=5, null=False)
    precision = models.DecimalField(max_digits=20, decimal_places=3, null=False)
    recall = models.DecimalField(max_digits=20, decimal_places=3, null=False)
    f1 = models.DecimalField(max_digits=20, decimal_places=3, null=False)
    volume = models.DecimalField(max_digits=20, decimal_places=3, null=False)
    date = models.DateTimeField(auto_now_add=True, null=False)
    description = models.TextField(help_text='ex) pretrained-model, used data-set etc...', null=False)

    class Meta:
        db_table = 'NLP_Models'