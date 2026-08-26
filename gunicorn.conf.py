import os


# A nagy .blend fájlok a webalkalmazáson át, közvetlenül az FTP NAS-ra
# streamelődnek. A Gunicorn alapértelmezett 30 másodperce ehhez kevés lehet.
timeout = max(30, int(os.getenv("GUNICORN_TIMEOUT", "1800")))
graceful_timeout = 30
