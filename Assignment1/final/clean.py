import sys, os, codecs
file = codecs.open("final.txt", "r", encoding="utf-8")
text = file.read()
text = text.replace("+ ", "")
text = text.replace("- ", "")
text = text.replace("\t", "")
text = text.replace(" ", "")
newFile = codecs.open("final-clean.txt", "a", encoding="utf-8")
newFile.write(text)
file.close()
newFile.close()