from pathlib import Path
import myconfig
import os

# AWS Info
AWS_ACCESS_KEY_ID = myconfig.MY_AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_ID = myconfig.MY_AWS_SECRET_ACCESS_KEY
ADMIN_ACCESS_ID = myconfig.ADMIN_ACCESS_ID
ADMIN_ACCESS_PW = myconfig.ADMIN_ACCESS_PW
AWS_URL = 'https://arspraxiabucket.s3.ap-northeast-2.amazonaws.com/'
AWS_BUCKET_NAME = 'arspraxiabucket'
AWS_STORAGE_BUCKET_NAME = 'arspraxiadata'
AWS_REGION = 'ap-northeast-2'
AWS_S3_CUSTOM_DOMAIN = '%s.s3.%s.amazonaws.com' % (
    AWS_STORAGE_BUCKET_NAME, AWS_REGION
)

# ML info
ISGPUON = False
NER_JSON_PATH = "./config/config_ner.json"
NER_CRF_JSON_PATH = "./config/config_ner_crf.json"
SA_JSON_PATH = "./config/config_sa.json"

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_URL = '/static/'
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'static'),
]
MEDIA_URL = '/bucket/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'bucket')

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = myconfig.SECRET_KEY

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True
#DEBUG = False
ALLOWED_HOSTS = myconfig.ALLOWED_HOSTS
SESSION_COOKIE_AGE = 7200
SESSION_SAVE_EVERY_REQUEST = True

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'myapp'
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'project.wsgi.application'

# Database
# https://docs.djangoproject.com/en/4.0/ref/settings/#databases

DATABASES = myconfig.DATABASES

# Password validation
# https://docs.djangoproject.com/en/4.0/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
# https://docs.djangoproject.com/en/4.0/topics/i18n/

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Default primary key field type
# https://docs.djangoproject.com/en/4.0/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'