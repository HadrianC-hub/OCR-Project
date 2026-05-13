from django.urls import path
from . import views

urlpatterns = [
    path('login/',                       views.login_view,     name='login'),
    path('logout/',                      views.logout_view,    name='logout'),
    path('users/',                       views.user_management, name='user_management'),
    path('users/add/',                   views.add_user,       name='add_user'),
    path('users/<int:user_id>/edit/',    views.edit_user_role, name='edit_user_role'),
    path('users/<int:user_id>/delete/',  views.delete_user,    name='delete_user'),
    path('password/change/',             views.change_password, name='change_password'),
]
