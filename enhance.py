from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageStat
# flag = 0 for Red, Blue Compensation via green channel
# flag = 1 for Red Compensation via green channel

def compensate_RB(image, flag):
    # Splitting the image into R, G and B components
    imager, imageg, imageb = image.split()
    
    # Get maximum and minimum pixel value
    minR, maxR = imager.getextrema()
    minG, maxG = imageg.getextrema()
    minB, maxB = imageb.getextrema()
    
    # Convert to array
    imageR = np.array(imager,np.float64)
    imageG = np.array(imageg,np.float64)
    imageB = np.array(imageb,np.float64)
    
    x,y = image.size
    
    # Normalizing the pixel value to range (0, 1)
    for i in range(0, y):
        for j in range(0, x):
            imageR[i][j]=(imageR[i][j]-minR)/(maxR-minR)
            imageG[i][j]=(imageG[i][j]-minG)/(maxG-minG)
            imageB[i][j]=(imageB[i][j]-minB)/(maxB-minB)
    
    # Getting the mean of each channel
    meanR=np.mean(imageR)
    meanG=np.mean(imageG)
    meanB=np.mean(imageB)
    

    # Compensate Red and Blue channel
    if flag == 0:
        for i in range(y):
            for j in range(x):
                imageR[i][j]=int((imageR[i][j]+(meanG-meanR)*(1-imageR[i][j])*imageG[i][j])*maxR)
                imageB[i][j]=int((imageB[i][j]+(meanG-meanB)*(1-imageB[i][j])*imageG[i][j])*maxB)

        # Scaling the pixel values back to the original range
        for i in range(0, y):
            for j in range(0, x):
                imageG[i][j]=int(imageG[i][j]*maxG)
   
    # Compensate Red channel
    if flag == 1:
        for i in range(y):
            for j in range(x):
                imageR[i][j]=int((imageR[i][j]+(meanG-meanR)*(1-imageR[i][j])*imageG[i][j])*maxR)

        # Scaling the pixel values back to the original range
        for i in range(0, y):
            for j in range(0, x):
                imageB[i][j]=int(imageB[i][j]*maxB)
                imageG[i][j]=int(imageG[i][j]*maxG)
            
    # Create the compensated image
    compensateIm = np.zeros((y, x, 3), dtype = "uint8")
    compensateIm[:, :, 0]= imageR;
    compensateIm[:, :, 1]= imageG;
    compensateIm[:, :, 2]= imageB;
    
    # Plotting the compensated image
    plt.figure(figsize = (20, 20))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.title("RB Compensated Image")
    plt.imshow(compensateIm) 
    plt.show()
    compensateIm=Image.fromarray(compensateIm)
    
    return compensateIm

def gray_world(image):
    # Splitting the image into R, G, and B components
    imager, imageg, imageb = image.split()
    
    # Form a grayscale image
    imagegray = image.convert('L')

    # Convert to array
    imageR = np.array(imager, np.float64)
    imageG = np.array(imageg, np.float64)
    imageB = np.array(imageb, np.float64)
    imageGray = np.array(imagegray, np.float64)

    x, y = image.size

    # Get mean value of pixels     
    meanR = np.mean(imageR)
    meanG = np.mean(imageG)
    meanB = np.mean(imageB)
    meanGray = np.mean(imageGray)

    # Gray World Algorithm  
    for i in range(0, y):
        for j in range(0, x):
            imageR[i][j] = int(imageR[i][j] * meanGray / meanR)
            imageG[i][j] = int(imageG[i][j] * meanGray / meanG)
            imageB[i][j] = int(imageB[i][j] * meanGray / meanB)

    # Create the white balanced image
    whitebalancedIm = np.zeros((y, x, 3), dtype="uint8")
    whitebalancedIm[:, :, 0] = imageR
    whitebalancedIm[:, :, 1] = imageG
    whitebalancedIm[:, :, 2] = imageB

    return Image.fromarray(whitebalancedIm)
def sharpen(wbimage, original):
    # First find the smoothed image using Gaussian filter
    smoothed_image = wbimage.filter(ImageFilter.GaussianBlur)
    
    # Split the smoothed image into R, G and B channel
    smoothedr, smoothedg, smoothedb = smoothed_image.split()
    
    # Split the input image 
    imager, imageg, imageb = wbimage.split()
    
    # Convert image to array
    imageR = np.array(imager,np.float64)
    imageG = np.array(imageg,np.float64)
    imageB = np.array(imageb,np.float64)
    smoothedR = np.array(smoothedr,np.float64)
    smoothedG = np.array(smoothedg,np.float64)
    smoothedB = np.array(smoothedb,np.float64)
    
    x, y=wbimage.size
    
    # Perform unsharp masking 
    for i in range(y):
        for j in range(x):
            imageR[i][j]=2*imageR[i][j]-smoothedR[i][j]
            imageG[i][j]=2*imageG[i][j]-smoothedG[i][j]
            imageB[i][j]=2*imageB[i][j]-smoothedB[i][j]
    
    # Create sharpened image
    sharpenIm = np.zeros((y, x, 3), dtype = "uint8")         
    sharpenIm[:, :, 0]= imageR;
    sharpenIm[:, :, 1]= imageG;
    sharpenIm[:, :, 2]= imageB; 
    
    # Plotting the sharpened image
    plt.figure(figsize = (20, 20))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original)
    plt.subplot(1, 3, 2)
    plt.title("White Balanced Image")
    plt.imshow(wbimage)
    plt.subplot(1, 3, 3)
    plt.title("Sharpened Image")
    plt.imshow(sharpenIm) 
    plt.show()
    
    return Image.fromarray(sharpenIm)
def hsv_global_equalization(image):
    # Convert to HSV
    hsvimage = image.convert('HSV')
   
    # Plot HSV Image
    plt.figure(figsize = (20, 20))
    plt.subplot(1, 2, 1)
    plt.title("White balanced Image")
    plt.imshow(hsvimage)
    
    # Splitting the Hue, Saturation and Value Component 
    Hue, Saturation, Value = hsvimage.split()
    # Perform Equalization on Value Component
    equalizedValue = ImageOps.equalize(Value, mask = None)

    x, y = image.size
    # Create the equalized Image
    equalizedIm = np.zeros((y, x, 3), dtype = "uint8")
    equalizedIm[:, :, 0]= Hue;
    equalizedIm[:, :, 1]= Saturation;
    equalizedIm[:, :, 2]= equalizedValue;
    
    # Convert the array to image
    hsvimage = Image.fromarray(equalizedIm, 'HSV') 
    # Convert to RGB
    rgbimage = hsvimage.convert('RGB')
    
    # Plot equalized image
    plt.subplot(1, 2, 2)
    plt.title("Contrast enhanced Image")
    plt.imshow(rgbimage)
    
    return rgbimage

def average_fusion(image1, image2):
    # Split the images in R, G, B components
    image1r, image1g, image1b = image1.split()
    image2r, image2g, image2b = image2.split()
    
    # Convert to array
    image1R = np.array(image1r, np.float64)
    image1G = np.array(image1g, np.float64)
    image1B = np.array(image1b, np.float64)
    image2R = np.array(image2r, np.float64)
    image2G = np.array(image2g, np.float64)
    image2B = np.array(image2b, np.float64)
    
    x, y = image1R.shape
    
    # Perform fusion by averaging the pixel values
    for i in range(x):
        for j in range(y):
            image1R[i][j]= int((image1R[i][j]+image2R[i][j])/2)
            image1G[i][j]= int((image1G[i][j]+image2G[i][j])/2)
            image1B[i][j]= int((image1B[i][j]+image2B[i][j])/2)
    
    # Create the fused image
    fusedIm = np.zeros((x, y, 3), dtype = "uint8")
    fusedIm[:, :, 0]= image1R;
    fusedIm[:, :, 1]= image1G;
    fusedIm[:, :, 2]= image1B;
    
    # Plot the fused image
    plt.figure(figsize = (20, 20))
    plt.subplot(1, 3, 1)
    plt.title("Sharpened Image")
    plt.imshow(image1)
    plt.subplot(1, 3, 2)
    plt.title("Contrast Enhanced Image")
    plt.imshow(image2)
    plt.subplot(1, 3, 3)
    plt.title("Average Fused Image")
    plt.imshow(fusedIm) 
    plt.show()
    
    return Image.fromarray(fusedIm)
    


if __name__ == "__main__":
    image1 = Image.open("/kaggle/input/2nd-test/images.jpeg")
    
    compensatedimage1=compensate_RB(image1, 0)
    whitebalanced1=gray_world(compensatedimage1)
    sharpenedimage1=sharpen(whitebalanced1, image1)
    contrastenhanced1 = hsv_global_equalization(whitebalanced1)
    averagefused1 =  average_fusion(sharpenedimage1, contrastenhanced1)
    # Save or display the result
    averagefused1.save("output_image.jpg")
    plt.imshow(averagefused1)
    plt.title("Final Output Image")
    plt.axis("off")
    plt.show()
