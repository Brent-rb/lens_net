import cv2
import json

imagePath = "output/gal_0.tif"
image = cv2.imread(imagePath)
jsonPath = "output/data_0.json"

with open(jsonPath, "r") as jsonFile:
    content = jsonFile.read()

data = json.loads(content)
print(data[0][0])

for row in range(10):
    for column in range(10):
        startY = row * 40
        endY = startY + 39
        startX = column * 40
        endX = startX + 39

        subImage = image[startY: endY, startX: endX]
        currentData = data[row][column]

        print(f"e1:{currentData['e1']:.3f} e2:{currentData['e2']:.3f} g1:{currentData['g1']:.3f} g2:{currentData['g2']:.3f}")
        print(subImage.flatten())