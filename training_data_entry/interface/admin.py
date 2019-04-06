from django.contrib import admin
from .models import POIClass, Filter
# Register your models here.
class POIClassAdmin(admin.ModelAdmin):

    list_display = (
        'id','city','poi', 'filter'
    )
    ordering = ['id', ]
    search_fields = ('city', 'poi', 'filters__name')


class FilterClassAdmin(admin.ModelAdmin):
    list_display = (
        'id', 'name'
    )
    ordering = ['id']
    search_fields = ['name']


admin.site.register(POIClass,POIClassAdmin)
admin.site.register(Filter,FilterClassAdmin)