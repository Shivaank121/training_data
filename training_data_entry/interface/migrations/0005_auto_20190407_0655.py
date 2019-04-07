# Generated by Django 2.2 on 2019-04-07 06:55

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('interface', '0004_auto_20190405_2032'),
    ]

    operations = [
        migrations.AlterField(
            model_name='filter',
            name='name',
            field=models.CharField(choices=[('Adventure & Outdoors', 'Adventure & Outdoors'), ('Spiritual', 'Spiritual'), ('Chill & Relax', 'Chill & Relax'), ('Heritage', 'Heritage'), ('Travel & Learn', 'Travel & Learn'), ('Social Tourism (Volunteer & Travel)', 'Social Tourism (Volunteer & Travel)')], max_length=50, unique=True),
        ),
    ]