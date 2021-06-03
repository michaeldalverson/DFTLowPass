import cv2 as cv
import numpy as np

def Fourier(frame, ham2d):
    f = cv.dft(np.float32(frame), flags=cv.DFT_COMPLEX_OUTPUT)
    f_shift = np.fft.fftshift(f)
    f_complex = f_shift[:, :, 0] + 1j*f_shift[:, :, 1]

    f_abs = np.abs(f_complex) + 1
    f_bounded = 20*np.log(f_abs)
    f_img = 255*f_bounded/np.max(f_bounded)
    f_img = f_img.astype(np.uint8)

    f_filtered = ham2d * f_complex
    f_filtered_shifted = np.fft.fftshift(f_filtered)
    inv_img = np.fft.ifft2(f_filtered_shifted)
    filtered_img = np.abs(inv_img)
    filtered_img -= filtered_img.min()
    filtered_img = filtered_img*255/filtered_img.max()
    filtered_img = filtered_img.astype(np.uint8)

    return f_img, filtered_img

cam=cv.VideoCapture(0)

while(True):
    ret, frame = cam.read()
    frame = cv.resize(frame, None, fx = .9, fy = 1.24, interpolation = cv.INTER_AREA)
    cv.imshow("Camera", frame)

    r = 50
    ham = np.hamming(570)[:, None]
    ham2d = np.sqrt(np.dot(ham, ham.T))**r
    frame = frame[:570, :570]

    f_imgR, f_imgRS = Fourier(frame[:, :, 0], ham2d)
    f_imgG, f_imgGS = Fourier(frame[:, :, 1], ham2d)
    f_imgB, f_imgBS = Fourier(frame[:, :, 2], ham2d)
    lowPass = np.zeros((570, 570, 3))
    lowPass[:, :, 0] = f_imgRS
    lowPass[:, :, 1] = f_imgGS
    lowPass[:, :, 2] = f_imgBS
    lowPass = cv.flip(lowPass/255, 0)
    cv.imshow("Frequency Domain", f_imgR)
    cv.imshow("Low Pass Filter", lowPass)


    c = cv.waitKey(1)
    if c == 27:
        break

cam.release()
cv.destroyAllWindows()