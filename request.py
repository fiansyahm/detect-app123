import requests

audiofile = "./Depression0.wav"
url = "http://127.0.0.1:5000/predict"
with open(audiofile, 'rb') as fobj:
    x = requests.post(url, files={'file': fobj})

    print(x.text)