import request

resp = request.post("https://getprediction-a56t3srpta-et.a.run.app", files={'file': open('glovi.jpeg', 'rb')})

print(resp.json())