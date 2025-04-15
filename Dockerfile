FROM python:3.11

USER root

WORKDIR /home/trukhinmaksim

COPY requirements.txt /home/trukhinmaksim

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8888
EXPOSE 8000

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token='111'"]
