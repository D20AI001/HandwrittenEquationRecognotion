import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import load_model
from equationsolver import calculate

#import matplotlib.pyplot as plt
root = os.getcwd()
print(root)
OUTPUT_DIR = os.path.join(root, 'segmented')



"""
   Convert an image to a csv string for the EMNIST dataset.           
   @param filepath - the filepath of the image           
   @param char_code - the character code of the image           
   @return the csv string           
"""
def img2emnist(filepath, char_code):
    img = Image.open(filepath).resize((28, 28))
    inv_img = ImageOps.invert(img)
    flatten = np.array(inv_img).flatten() / 255
    flatten = np.where(flatten > 0.5, 1, 0)
    csv_img = ','.join([str(num) for num in flatten])
    csv_str = '{},{}'.format(char_code, csv_img)
    return csv_str



"""
    Given an image, find the upper and lower lines of the image.
    @param x - the image
    @return the upper and lower lines of the image
"""

def line_array(x):
	upper, lower = [], []
	for y in range(5, len(x) - 5):
		s_a, s_p = strtline(y, x)
		e_a, e_p = endline(y, x)
		if s_a >= 7 and s_p >= 5:
			upper.append(y)
		if e_a >= 5 and e_p >= 7:
			lower.append(y)
	return upper, lower

"""
    Given a y coordinate and an array, return the number of ahead and previous lines that are not empty.
    @param y - the y coordinate
    @param array - the array
    @return the number of ahead and previous lines that are not empty
    """
def strtline(y, array):

	prev,ahead = 0,0
	for i in array[y:y+10]:
		if i > 3:
			ahead += 1
	for i in array[y-10:y]:
		if i==0:
			prev += 1
	return ahead, prev

"""
    Given a y coordinate and an array, return the number of zeros ahead and behind the coordinate.
    @param y - the y coordinate
    @param array - the array
    @returns the number of zeros ahead and behind the coordinate
    """
def endline(y, array):


	ahead = 0
	prev = 0
	for i in array[y:y+10]:
		if i==0:
			ahead += 1
	for i in array[y-10:y]:
		if i > 3:
			prev += 1
	return ahead, prev

"""
    Given a y coordinate, an array of values, and a window size, return the number of values ahead and behind the current value.
    @param y - the y coordinate of the current value.
    @param array - the array of values.
    @param a - the window size.
    @return the number of values ahead and behind the current value.
    """
def endline_word(y, array, a):

	ahead = 0
	prev = 0
	for i in array[y:y+2*a]:
		if i < 2:
			ahead += 1
	for i in array[y-a:y]:
		if i > 2:
			prev += 1
	return prev ,ahead

"""
    Given an array of words and the size of the array, return the endlines of the array.
    @param array - the array of words
    @param a - the size of the array
    @return the endlines of the array
    """
def end_line_array(array, a):

	list_endlines = []
	for y in range(len(array)):
		e_p, e_a = endline_word(y, array, a)
		# print(e_p, e_a)
		if e_a >= int(1.5*a) and e_p >= int(0.7*a):
			list_endlines.append(y)
	return list_endlines

"""
    Given an array of numbers, return a list of numbers that are one number less than the next number in the array.
    @param array - the array of numbers
    @return the list of numbers that are one number less than the next number in the array
    """
def refine_endword(array):

	refine_list = []
	for y in range(len(array)-1):
		if array[y]+1 < array[y+1]:
			refine_list.append(array[y])

	if len(array) != 0:
		refine_list.append(array[-1])
	return refine_list

"""
    Given two arrays, refine the upper and lower bounds of the array.
    @param array_upper - the upper array
    @param array_lower - the lower array
    @returns the refined upper and lower bounds
    """
def refine_array(array_upper, array_lower):

	upper, lower = [], []
	for y in range(len(array_upper)-1):
		if array_upper[y] + 5 < array_upper[y+1]:
			upper.append(array_upper[y]-10)
	for y in range(len(array_lower)-1):
		if array_lower[y] + 5 < array_lower[y+1]:
			lower.append(array_lower[y]+10)
	if array_upper:
		upper.append(array_upper[-1]-10)
	if array_lower:
		lower.append(array_lower[-1]+10)
	return upper, lower

"""
    Calculate the average width of the letters in the image.
    @param contours - the contours of the letters in the image           
    @return the average width of the letters in the image           
    """
def letter_width(contours):

	letter_width_sum = 0
	count = 0
	for cnt in contours:
		if cv2.contourArea(cnt) > 20:
			x,y,w,h = cv2.boundingRect(cnt)
			letter_width_sum += w
			count += 1
	if count==0:
		return 0
	return letter_width_sum/count

"""
    Detect the end of a word. This is done by detecting the end of a word by detecting the end of a line.
    @param lines - the lines of the image           
    @param i - the index of the line being examined           
    @param bin_img - the binary image           
    @param mean_lttr_width - the mean letter width           
    @param total_width - the total width of the image           
    @return the end lines of the word           
    """
def end_wrd_dtct(lines, i, bin_img, mean_lttr_width, total_width):

	count_y = np.zeros(shape = total_width)
	for x in range(total_width):
		for y in range(lines[i][0],lines[i][1]):
			if bin_img[y][x] == 255:
				count_y[x] += 1

	end_lines = end_line_array(count_y, int(mean_lttr_width))
	endlines = refine_endword(end_lines)
	return endlines

"""
    Given a contour, find the bounding rectangle for the letter.
    @param k - the contour index           
    @param contours - the contours array           
    @returns the bounding rectangle for the letter
    """
def get_letter_rect(k, contours):

	valid = True
	x,y,w,h = cv2.boundingRect(contours[k])
	for i in range(len(contours)):
		cnt = contours[i]
		if i == k:
			continue
		elif cv2.contourArea(cnt) < 50:
			continue

		x1,y1,w1,h1 = cv2.boundingRect(cnt)
		
		if abs(x1 + w1/2 - (x + w/2)) < 50:
			if y1 > y:
				h = abs(y - (y1 + h1))
				w = abs(x - (x1 + w1))
			else:
				valid = False
			break

	return (valid,x,y,w,h)

"""
    Take the input image and split it into individual letters. Save the images in the output directory.
    @param lines_img - the image containing the letters           
    @param x_lines - the x coordinates of the letters           
    @param i - the index of the image           
    """
def letter_seg(lines_img, x_lines, i):

	copy_img = lines_img[i].copy()
	x_linescopy = x_lines[i].copy()
	
	letter_img = []
	letter_k = []
	
	contours, hierarchy = cv2.findContours(copy_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	for k in range(len(contours)):
		cnt = contours[k]
		if cv2.contourArea(cnt) < 50:
			continue
		
		valid,x,y,w,h = get_letter_rect(k, contours)
		if valid:
			letter_k.append((x,y,w,h))

	letter = sorted(letter_k, key=lambda student: student[0])
	
	word = 1
	letter_index = 0
    
	for e in range(len(letter)):
		if(letter[e][0]<x_linescopy[0]):
			letter_index += 1
			letter_img_tmp = lines_img[i][letter[e][1]-5:letter[e][1]+letter[e][3]+5,letter[e][0]-5:letter[e][0]+letter[e][2]+5]
			letter_img = letter_img_tmp#cv2.resize(letter_img_tmp, dsize =(28, 28), interpolation = cv2.INTER_AREA)
			cv2.imwrite(os.path.join(OUTPUT_DIR, str(i+1)+'_'+str(word)+'_'+str(letter_index)+'.jpg'), 255-letter_img)
		else:
			x_linescopy.pop(0)
			word += 1
			letter_index = 1
			letter_img_tmp = lines_img[i][letter[e][1]-5:letter[e][1]+letter[e][3]+5,letter[e][0]-5:letter[e][0]+letter[e][2]+5]
			letter_img = cv2.resize(letter_img_tmp, dsize =(28, 28), interpolation = cv2.INTER_AREA)
			cv2.imwrite(OUTPUT_DIR+str(i+1)+'_'+str(word)+'_'+str(letter_index)+'.jpg', 255-letter_img)
        

#filepath = "/media/sagar/ext_drive/01_Sagar/PG/Mtech/CV/Projects/Project-1/src-code/input_1.jpeg"
filepath = "/media/sagar/ext_drive/01_Sagar/PG/Mtech/CV/Projects/Project-1/src-code/input.png"

print("\n........Program Initiated.......\n")

# Read in the image and convert it to grayscale.
src_img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
orig_height, orig_width = src_img.shape

print("\n Resizing Image........")
width = 1320
height = int(width * orig_height / orig_width)
# resize the image to the desired width, while maintaining the aspect ratio of the image, and then write it to disk
src_img = cv2.resize(src_img, dsize=(width, height), interpolation=cv2.INTER_AREA)
RESIZE_DIR = os.path.join(root, 'segmented')
cv2.imwrite(root+'/Resized_Image.jpg',src_img)


print("#---------Image Info:--------#")
print("\tHeight =", height, "\n\tWidth =", width)
PIXEL_SET = 255
kernel_size = 21
normalized_mean = 20
# apply adaptive thresholding to the image, using the mean method, and then invert the resulting image.
bin_img = cv2.adaptiveThreshold(src_img, PIXEL_SET, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, kernel_size,
									normalized_mean)
cv2.imwrite(root+'/Binaized_Image.jpg',bin_img)

print("Noise Removal")
# Create a kernel to use for dilation. This is used to remove noise from the image.
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# Apply a closing morphology to the binary image to remove noise. Then apply a threshold to the image.
final_thr = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
contr_retrival = final_thr.copy()

cv2.imwrite(root+'/Noise_Removed_Image.jpg',contr_retrival)

print("Character Segmentation")
count_x = np.zeros(shape=(height))
# Count the number of pixels in each row.
for y in range(height):
	for x in range(width):
		if bin_img[y][x] == PIXEL_SET:
			count_x[y] += 1

# Given the number of lines, create an array of upper and lower lines.
upper_lines, lower_lines = line_array(count_x)
# Given the upper and lower lines, refine the lines to the correct position.
upperlines, lowerlines = refine_array(upper_lines, lower_lines)

# If the number of upper and lower lines are the same, then we know that we have a vertical line. We then set the pixels in the upper and lower lines to the pixel set
if len(upperlines) == len(lowerlines):
	lines = []
	for y in upperlines:
		final_thr[y][:] = PIXEL_SET
	for y in lowerlines:
		final_thr[y][:] = PIXEL_SET
	for y in range(len(upperlines)):
		lines.append((upperlines[y], lowerlines[y]))
else:
	print("Unable to process the noisy image")
""" 	showimages()
	k = cv2.waitKey(0)
	while 1:
		k = cv2.waitKey(0)
		if k & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			exit() """

lines = np.array(lines)
no_of_lines = len(lines)
print("\nLines :", no_of_lines)

# For each line, crop the image to the line itself. Then append it to a list.
lines_img = []
for i in range(no_of_lines):
	lines_img.append(bin_img[lines[i][0]:lines[i][1], :])

# Draw the contours of the contour retrieval on the source image.
contours, hierarchy = cv2.findContours(contr_retrival, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(src_img, contours, -1, (0, 255, 0), 1)

mean_lttr_width = letter_width(contours)
print("\nAverage Width of Each Letter:- ", mean_lttr_width)
x_lines = []

# For each line in the lines array, find the end of the word and then find the width of the word.
for i in range(len(lines_img)):
	x_lines.append(end_wrd_dtct(lines, i, bin_img, mean_lttr_width, width))
	
# For each line in the x_lines list, append the width of the image to the end of the list.
for i in range(len(x_lines)):
	x_lines[i].append(width)
	
# For each line, segment the letter and save the letter images.
for i in range(len(lines)):
	letter_seg(lines_img, x_lines, i)
	
# Find the contours in the binary image.
contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# For each contour, if the area is greater than 20, draw a rectangle around it.
for cnt in contours:
	if cv2.contourArea(cnt) > 20:
		x, y, w, h = cv2.boundingRect(cnt)
		cv2.rectangle(src_img, (x, y), (x + w, y + h), (0, 255, 0), 2)


# 'segmented' directory contains each mathematical symbol in the image
root = os.getcwd()
SEGMENTED_OUTPUT_DIR = os.path.join(root, 'segmented')
# trained model
MODEL_PATH = os.path.join(root, 'model.h5')
# csv file that maps numerical code to the character
mapping_processed = os.path.join(root, 'mapper.csv')
segmented_images = []
files = sorted(list(os.walk(SEGMENTED_OUTPUT_DIR))[0][2])
# writing images to the 'segmented' directory
for file in files:
    file_path = os.path.join(SEGMENTED_OUTPUT_DIR, file)
    segmented_images.append(Image.open(file_path))


#print(len(segmented_images))

files = sorted(list(os.walk(SEGMENTED_OUTPUT_DIR))[0][2])
for file in files:
    filename = os.path.join(SEGMENTED_OUTPUT_DIR, file)
    img = cv2.imread(filename, 0)
    #print(filename)
    kernel = np.ones((2, 2), np.uint8)
    dilation = cv2.erode(img, kernel, iterations=1)
    cv2.imwrite(filename, dilation)

segmented_characters = 'segmented_characters.csv'
#print(os.listdir())
if segmented_characters in os.listdir():
    os.remove(segmented_characters)
    # resize image to 48x48 and write the flattened out list to the csv file
with open(segmented_characters, 'a+') as f_test:
    column_names = ','.join(["label"] + ["pixel" + str(i) for i in range(784)])
    print(column_names, file=f_test)

    # Convert the segmented images to emnist format and write them to a file.
    files = sorted(list(os.walk(SEGMENTED_OUTPUT_DIR))[0][2])
    for f in files:
        file_path = os.path.join(SEGMENTED_OUTPUT_DIR, f)
        csv = img2emnist(file_path, -1)
        print(csv, file=f_test)


# Read in the dataframe and convert it to a numpy array. Also convert the dataframe to a numpy array.
test_df = data = pd.read_csv('segmented_characters.csv')
X_data = data.drop('label', axis = 1)
X_data = X_data.values.reshape(-1,28,28,1)
X_data = X_data.astype(float)

# Read in the mapping file and create a dictionary mapping each character to its code.
df = pd.read_csv(mapping_processed)
code2char = {}
for index, row in df.iterrows():
    code2char[row['id']] = row['char']


# predict each segmented character
model = load_model(MODEL_PATH)
results = model.predict(X_data)
results = np.argmax(results, axis = 1)
parsed_str = ""
for r in results:
    parsed_str += code2char[r]
print(parsed_str)

solution = calculate(parsed_str)

print(solution)