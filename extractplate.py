import numpy
import imutils
import cv2
import easyocr as ocr
from matplotlib import pyplot as pl

# Reading in the image and applying both Grayscale and Blur.

fname = input("Enter File Name: ")
fhandle = cv2.imread(fname)
grayscale = cv2.cvtColor(fhandle, cv2.COLOR_BGR2GRAY)
pl.imshow(cv2.cvtColor(grayscale, cv2.COLOR_BGR2RGB))

# Applying Noise reduction and edge detection.

filter = cv2.bilateralFilter(grayscale, 11, 17, 17)
edge = cv2.Canny(filter, 30, 200)
pl.imshow(cv2.cvtColor(edge, cv2.COLOR_BGR2RGB))

keypoints = cv2.findContours(edge.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break

mask = numpy.zeros(grayscale.shape, numpy.uint8)
new_image = cv2.drawContours(mask, [location], 0,255, -1)
new_image = cv2.bitwise_and(fhandle, fhandle, mask=mask)

pl.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
# un-comment if you want to check the output.
# pl.show()

(x,y) = numpy.where(mask==255)
(x1, y1) = (numpy.min(x), numpy.min(y))
(x2, y2) = (numpy.max(x), numpy.max(y))
cropped_img = grayscale[x1:x2+1, y1:y2+1]
pl.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
# pl.show()

lang = {"Abaza": "abq", "Adyghe": "ady","Afrikaans": "af", "Angika": "ang","Arabic":	"ar","Assamese":	"as","Avar":	"ava","Azerbaijani":	"az","Belarusian":	"be","Bulgarian":	"bg","Bihari":	"bh","Bhojpuri":	"bho","Bengali":	"bn"
,"Bosnian":	"bs","Simplified Chinese":	"ch_sim","Traditional Chinese":	"ch_tra","Chechen":	"che","Czech":	"cs","Welsh":	"cy","Danish":	"da","Dargwa":	"dar","German":	"de","English":	"en","Spanish":	"es","Estonian":	"et","Persian (Farsi)":	"fa"
,"French":	"fr","Irish":	"ga","Goan Konkani":	"gom","Hindi":	"hi","Croatian":	"hr","Hungarian":	"hu","Indonesian":	"id","Ingush":	"inh","Icelandic":	"is","Italian":	"it","Japanese":	"ja","Kabardian":	"kbd","Kannada":	"kn","Korean":	"ko","Kurdish":	"ku","Latin":	"la","Lak":	"lbe"
,"Lezghian":	"lez","Lithuanian":	"lt","Latvian":	"lv","Magahi":	"mah","Maithili":	"mai","Maori":	"mi","Mongolian":	"mn","Marathi":	"mr","Malay":	"ms","Maltese":	"mt","Nepali":	"ne","Newari":	"new","Dutch":	"nl","Norwegian":	"no"
,"Occitan":	"oc","Pali":	"pi","Polish":	"pl","Portuguese":	"pt","Romanian":	"ro","Russian":	"ru","Serbian (cyrillic)":	"rs_cyrillic","Serbian (latin)":	"rs_latin","Nagpuri":	"sck","Slovak":	"sk","Slovenian":	"sl","Albanian":	"sq"
,"Swedish":	"sv","Swahili":	"sw","Tamil":	"ta","Tabassaran":	"tab","Telugu":	"te","Thai":	"th","Tajik":	"tjk","Tagalog":	"tl","Turkish":	"tr","Uyghur":	"ug","Ukranian":	"uk","Urdu":	"ur","Uzbek":	"uz","Vietnamese":	"vi"}

x = sorted( [ (v,k) for k,v in lang.items() ] )
x = sorted(x)

for v, k in x[:100]:
    print(k)

inp = input("Choose the language you want to use with the easyOCR: ")

for k, v in lang.items():
    if k == inp:
        Value = v


reader = ocr.Reader([Value])
result = reader.readtext(cropped_img)

text = result[0][-2]
font = cv2.FONT_HERSHEY_SIMPLEX
res = cv2.putText(fhandle, text=text, org=(approx[0][0][1], approx[1][0][1]), fontFace=font, fontScale=1, color=(0,400,0), thickness=2, lineType=cv2.LINE_AA)
res = cv2.rectangle(fhandle, tuple(approx[0][0]), tuple(approx[2][0]), (0,400,0),3)
pl.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
pl.show()
