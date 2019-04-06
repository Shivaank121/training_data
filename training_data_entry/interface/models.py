from django.db import models
# Create your models here.


class Filter(models.Model):
    FILTER_CHOICES = (
        ('Adventure & Outdoors', 'Adventure & Outdoors'),
        ('Pilgrimage', 'Pilgrimage'),
        ('Chill & Relax', 'Chill & Relax'),
        ('Heritage', 'Heritage'),
        ('Travel & Learn', 'Travel & Learn'),
        ('Social Tourism (Volunteer & Travel)', 'Social Tourism (Volunteer & Travel)'),
    )
    name = models.CharField(max_length=50,choices=FILTER_CHOICES, unique=True)

    def __str__(self):
        return self.name


class POIClass(models.Model):
    city = models.CharField(max_length=50, null=False, blank= False)
    poi = models.CharField(max_length=60, null= False, blank= False)
    filters = models.ManyToManyField(Filter, help_text="Department of this user.")

    def filter(self):
        string = ''
        i=1
        for f in self.filters.all():
            string += str(i) + ': ' + f.name + '. '
            i += 1
        return string

    class Meta:
        db_table = 'POIClass'
        verbose_name = 'POIClass'
        verbose_name_plural = 'POIClasses'
