from gtts import gTTS
input_text = "we will build until we get to the top"
convert = gTTS(text= input_text, lang='en', slow=False)

#Saving the converted audio into an mp3 file
convert.save('first_audio.mp3')