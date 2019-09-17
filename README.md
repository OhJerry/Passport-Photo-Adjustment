# Passport-Photo-Adjustment
This project is designed to extract foreground and refine people images to meet valid passport requirements using OpenCV.

PassportPhoto.py stores the code. The file is highly documented, and examples are in the bottom of the file, helping to show the usage of the code.

Src folder stores sample photos, some of them are already clean enough, others are somehow 'polluted'.
SrcTest folder stores adjusted-sample-photos. You could have a glance on how well the algorithm is.


Below are some tohughts I kept down during the progress. I found them very interesting, so I am sharing them here.

Strategies:
1. Floodfill to ensure inside parts (eyes, white clothes) to be solid.
	Have to set up a threshold for discrimination
	* Problem, probably unable to deal with photo taken with margins.
2. Grabcut to seperate foreground from background.
	Able to deal with blur margin photoes.
	* Problem, hard to deal with grey background
	* left shoulder broken.
3. Face detection helps to calculate position of heads and shoulders.
	Able to give more specific instruction for grabcut.
	Give instruction of floodfill starting point.


Progress:
1. check grabcut left shoulder problem, no blur this time.
2. if needed, try flip graph and check left problem again.
3. if needed, combine two mask.
4. play around with face detection, draw rectangle around face.
5. get 4 points and try do calculation with it.
6. try draw recatangle around shoulder, draw floodfill point.
7. play around with floodfill.
8. cut edges: width // 20, height // 40
	floodpoint: width // 10, width // 10


Tricks:
1. Flip image horizontally:
	img = np.fliplr(img)

2. Change background from black to white:
	person = origin_img*mask[:,:,np.newaxis]
	background = origin_img - person
	background[np.where((background > [0,0,0]).all(axis = 2))] =[255,255,255]
	final = background + person

3. Resize image:
	dim = (width, height)
	resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

4. Blur and smoothing:
	blur = cv2.GaussianBlur(img, (15,15), 0) 

5. Grayscale image to rgb:
	cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
