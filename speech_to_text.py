import speech_recognition as sr
r=sr.Recognizer()
with sr.Microphone() as source:
    print("Please say something")
    audio = r.listen(source)
    print("Time over, thanks")
try:
    print("You said: "+r.recognize_google(audio,language = 'en-US'));
except:
     pass