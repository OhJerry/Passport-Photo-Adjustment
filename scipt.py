import numpy as np
import cv2
import glob


def marginremove(img, rect=None, blur=(75,75)):
	orig = img.copy()
	img = cv2.GaussianBlur(img, blur, 0) 
	height, width = img.shape[:2]
	param = height // 40

	rect = rect or (param,param,width,height)

	# get grabcut mask for original image, will have left blank problem
	mask = np.zeros((img.shape[:2]),np.uint8)
	bgdModel = np.zeros((1,65),np.float64)
	fgdModel = np.zeros((1,65),np.float64)
	cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
	mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

	# get grabcut mask for horizontally flip image
	flip_img = np.fliplr(img)
	flip_mask = np.zeros((flip_img.shape[:2]),np.uint8)
	flip_bgdModel = np.zeros((1,65),np.float64)
	flip_fgdModel = np.zeros((1,65),np.float64)
	cv2.grabCut(flip_img,flip_mask,rect,flip_bgdModel,flip_fgdModel,5,cv2.GC_INIT_WITH_RECT)
	flip_mask2 = np.where((flip_mask==2)|(flip_mask==0),0,1).astype('uint8')
	mask2f = np.fliplr(flip_mask2)

	# OR combine two mask, to avoid left and right shoulder blanks
	mask_lr = mask2 | mask2f

	person = orig*mask_lr[:,:,np.newaxis]
	background = orig - person
	background[np.where((background > [0,0,0]).all(axis = 2))] =[255,255,255]
	final = background + person

	return final



def floodfill(img, floodpoints=None, threshold_sure_bg=200, threshold_sure_fg=190):
	# Threshold.
	# Set values equal to or above 220 to 0.
	# Set values below 220 to 255.
	height, width = img.shape[:2]
	floodpoints = floodpoints or [(width//10, width//10), (width//10, width - width//10)]
	
	th1, im_th1 = cv2.threshold(img, threshold_sure_bg, 255, cv2.THRESH_BINARY_INV);
	th2, im_th2 = cv2.threshold(img, threshold_sure_fg, 255, cv2.THRESH_BINARY_INV);
	 
	# Copy the thresholded image.
	im_floodfill1 = im_th1.copy()
	im_floodfill2 = im_th2.copy()
	 
	# Mask used to flood filling.
	# Notice the size needs to be 2 pixels than the image.
	h, w = im_th1.shape[:2]
	mask1 = np.zeros((h+2, w+2), np.uint8)
	mask2 = np.zeros((h+2, w+2), np.uint8)
	 
	# Floodfill from point (point_width, point_height)
	for point in floodpoints:
		cv2.floodFill(im_floodfill1, mask1, point, 255);
		cv2.floodFill(im_floodfill2, mask2, point, 255);
	 
	# Invert floodfilled image
	im_floodfill_inv1 = cv2.bitwise_not(im_floodfill1)
	im_floodfill_inv2 = cv2.bitwise_not(im_floodfill2)
	 
	# Combine the two images to get the foreground.
	im_sure_bg = cv2.bitwise_not(im_th1 | im_floodfill_inv1)
	im_sure_fg = cv2.bitwise_not(im_th2 | im_floodfill_inv2)


	return (im_sure_bg, im_sure_fg)



def grabcut(img, sure_bg, sure_fg, rect=None, blur=(15,15)):
	img = cv2.GaussianBlur(img, blur, 0) 
	height, width = img.shape[:2]
	param = height // 40

	rect = rect or (param,param,width,height)

	# get grabcut mask for original image, will have left blank problem
	mask = np.zeros((img.shape[:2]),np.uint8)
	bgdModel = np.zeros((1,65),np.float64)
	fgdModel = np.zeros((1,65),np.float64)
	cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
	mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

	# get grabcut mask for horizontally flip image
	flip_img = np.fliplr(img)
	flip_mask = np.zeros((flip_img.shape[:2]),np.uint8)
	flip_bgdModel = np.zeros((1,65),np.float64)
	flip_fgdModel = np.zeros((1,65),np.float64)
	cv2.grabCut(flip_img,flip_mask,rect,flip_bgdModel,flip_fgdModel,5,cv2.GC_INIT_WITH_RECT)
	flip_mask2 = np.where((flip_mask==2)|(flip_mask==0),0,1).astype('uint8')
	mask2f = np.fliplr(flip_mask2)

	# OR combine two mask, to avoid left and right shoulder blanks
	mask_lr = mask2 | mask2f

	# white: 255, black: 0
	# foreground: 1, background: 0
	mask_lr[sure_bg == 255] = 0
	mask_lr[sure_fg == 0] = 1
	mask_lr, bgdModel, fgdModel = cv2.grabCut(img,mask_lr,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
	mask_lr = np.where((mask_lr==2)|(mask_lr==0),0,1).astype('uint8')

	return mask_lr




def bg_cleaner(path, margin_exist=False, writeto=None, floodpoints=None, threshold_sure_bg=200, threshold_sure_fg=190, grabcut_rect=None, grabcut_blur=(15,15), marginremove_rect=None, marginremove_blur=(35,35)):
	test_images = glob.glob(path)

	for img_name in test_images:
		# floodfill the background first
		# foreground would be black (rgb value 0)
		# background would be white (rgb value 255)
		gray_img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
		# resize if too big
		if gray_img.shape[0] > 800:
			gray_img = cv2.resize(gray_img, (600, 800), interpolation = cv2.INTER_AREA)
		gray_sure_bg, gray_sure_fg = floodfill(gray_img, floodpoints=floodpoints, threshold_sure_bg=threshold_sure_bg, threshold_sure_fg=threshold_sure_fg)

		img = cv2.imread(img_name)
		# resize if too big
		if img.shape[0] > 800:
			img = cv2.resize(img, (600, 800), interpolation = cv2.INTER_AREA)
		# make floodfill checked background white, by adding pixel values directly
		mask = grabcut(img, gray_sure_bg, gray_sure_fg, rect=grabcut_rect, blur=grabcut_blur)
		person = img*mask[:,:,np.newaxis]
		background = img - person
		background[np.where((background > [0,0,0]).all(axis = 2))] =[255,255,255]
		final = background + person
		
		if margin_exist:
			final = marginremove(final, rect=marginremove_rect, blur=marginremove_blur)
		if writeto == None:
			cv2.imwrite("test" + "/" + img_name.split("/")[1], final)
		else:
			cv2.imwrite(writeto, final)



# if __name__ == "__main__":
# 	# In order to clean background, call function bg_cleaner()
# 	# There are many default variables, the simplest way to call it is to pass in the path of img:
# 	bg_cleaner("target.jpg")

# 	# Or it is also possible to do batch processing by giving path like this:
# 	bg_cleaner("path/*.jpg")
	
# 	# The default result path would be ./test/pathname.img, if need change, use writeto=
# 	bg_cleaner("target.jpg", writeto="./boom.jpg")

# 	# Floodpoints, threshold_sure_bg, threshold_sure_fg will be passed into floodfill function
# 	# It determines what is sure_background, and what is sure_foreground
# 	# If the result contains unwanted bg or fg, adjust threshold_sure_bg, threshold_sure_fg
# 	# The limit is (0, 255)
# 	bg_cleaner("target.jpg", threshold_sure_bg=220, threshold_sure_fg=180)

# 	# Grabcut_rect, grabcut_blur will be passed into grabcut function
# 	# It use opencv grabcut method to recgonize fg.
# 	# Normally you do not need to change this part
# 	# grabcut_blur must be two odd integers
# 	bg_cleaner("target.jpg", grabcut_rect=(0,0,200,200), grabcut_blur=(5,5))

# 	# If still exists dark margins, try to turn on margin_exist
# 	# And adjust marginremove_blur to remove the margin
# 	# Normally you do not need to change marginremove_rect
# 	# marginremove_blur must be two odd integers
# 	bg_cleaner("target.jpg", margin_exist=True, marginremove_rect=(0,0,200,200), marginremove_blur=(55,55))



def main():
	bg_cleaner("Archive/4.jpg", writeto='./boom.jpg')

main()







