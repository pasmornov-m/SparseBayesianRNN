FROM python:3.4-slim

# Заменяем все ссылки на архив
RUN sed -i 's|http://deb.debian.org|http://archive.debian.org|g' /etc/apt/sources.list && \
    sed -i 's|http://security.debian.org|http://archive.debian.org|g' /etc/apt/sources.list

# Удаляем строки с stretch-updates (они вызывают 404)
RUN sed -i '/stretch-updates/d' /etc/apt/sources.list

# Отключаем проверку срока действия и подписей
RUN echo 'Acquire::Check-Valid-Until false;' > /etc/apt/apt.conf.d/10no-check-valid-until

# Обновляем списки и устанавливаем зависимости
RUN apt-get update && apt-get install -y --allow-unauthenticated \
    build-essential \
    wget \
    libopenblas-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем Theano и Lasagne
RUN pip install \
    https://github.com/Theano/Theano/archive/master.zip \
    https://github.com/Lasagne/Lasagne/archive/master.zip

# Устанавливаем совместимые numpy
RUN pip install numpy==1.16.6

WORKDIR /app
