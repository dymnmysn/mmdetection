def yukle(img,impath='/ari/users/madiyaman/mmdetection/temp/silimg0.png'):
    im2write = img.cpu().numpy()[0]
    ust = im2write.max()
    alt = im2write.min()
    olcek = 250/(ust-alt)
    im2write = im2write + abs(alt)
    im2write = im2write * olcek
    im2write = im2write.transpose((1,2,0))
    im2write = im2write.astype('uint8')
    import matplotlib.image
    matplotlib.image.imsave(impath,im2write)

def yukletek(img,impath='/ari/users/madiyaman/mmdetection/temp/silyukletek.png'):
    im2write = img.cpu().numpy()
    ust = im2write.max()
    alt = im2write.min()
    olcek = 250/(ust-alt)
    im2write = im2write + abs(alt)
    im2write = im2write * olcek
    im2write = im2write.transpose((1,2,0))
    im2write = im2write.astype('uint8')
    import matplotlib.image
    matplotlib.image.imsave(impath,im2write)

def yuklemask(gt_masks,impath='/ari/users/madiyaman/mmdetection/temp/silmask'):
    import matplotlib.image
    mask2write = gt_masks[0].masks
    count=1
    for m in mask2write:
        matplotlib.image.imsave(impath+str(count)+'.png',m)
        count+=1

def gor2d(img,impath='/ari/users/madiyaman/mmdetection/temp/sil.png'):
    im2write = img.cpu().numpy()
    ust = im2write.max()
    alt = im2write.min()
    olcek = 250/(ust-alt)
    im2write = im2write + abs(alt)
    im2write = im2write * olcek
    l = len(im2write[0])
    im2write[0:10] = [[i*250/l for i in range(l)] for _ in range(10)]
    im2write = im2write.astype('uint8')
    import matplotlib.image
    matplotlib.image.imsave(impath,im2write)

def gor2dabs(img,impath='/ari/users/madiyaman/mmdetection/temp/silabs.png'):
    im2write = img.cpu().numpy()
    ust = im2write.max()
    alt = im2write.min()
    olcek = 250/(ust-alt)
    #im2write = im2write + abs(alt)
    im2write = im2write * 64
    im2write = im2write + 125
    import numpy as np
    im2write = np.clip(im2write, 0, 250)
    l = len(im2write[0])
    im2write[0:10] = [[i*250/l for i in range(l)] for _ in range(10)]
    im2write = im2write.astype('uint8')
    import matplotlib.image
    matplotlib.image.imsave(impath,im2write)

def gor3c(img,impath='/ari/users/madiyaman/mmdetection/temp/gor3c.png'):
    im3d = img.cpu().detach().numpy()[0]
    im2write1 = im3d[0]
    ust = im2write1.max()
    alt = im2write1.min()
    olcek = 250/(ust-alt)
    im2write1 = im2write1 + abs(alt)
    im2write1 = im2write1 * olcek
    l = len(im2write1[0])
    im2write1[0:10] = [[i*250/l for i in range(l)] for _ in range(10)]
    im2write1 = im2write1.astype('uint8')
    im2write2 = im3d[1]
    ust = im2write2.max()
    alt = im2write2.min()
    olcek = 250/(ust-alt)
    im2write2 = im2write2 + abs(alt)
    im2write2 = im2write2 * olcek
    l = len(im2write2[0])
    im2write2[0:10] = [[i*250/l for i in range(l)] for _ in range(10)]
    im2write2 = im2write2.astype('uint8')
    im2write3 = im3d[2]
    ust = im2write3.max()
    alt = im2write3.min()
    olcek = 250/(ust-alt)
    im2write3 = im2write3 + abs(alt)
    im2write3 = im2write3 * olcek
    l = len(im2write3[0])
    im2write3[0:10] = [[i*250/l for i in range(l)] for _ in range(10)]
    im2write3 = im2write3.astype('uint8')
    import matplotlib.image
    import numpy as np
    im2write = np.stack((im2write1,im2write2,im2write3), axis=2)
    matplotlib.image.imsave(impath,im2write)

def gor3cnb(img,impath='/ari/users/madiyaman/mmdetection/temp/gor3cnb.png'):
    im3d = img.cpu().detach().numpy()
    im2write1 = im3d[0]
    ust = im2write1.max()
    alt = im2write1.min()
    olcek = 250/(ust-alt)
    im2write1 = im2write1 + abs(alt)
    im2write1 = im2write1 * olcek
    l = len(im2write1[0])
    im2write1[0:10] = [[i*250/l for i in range(l)] for _ in range(10)]
    im2write1 = im2write1.astype('uint8')
    im2write2 = im3d[1]
    ust = im2write2.max()
    alt = im2write2.min()
    olcek = 250/(ust-alt)
    im2write2 = im2write2 + abs(alt)
    im2write2 = im2write2 * olcek
    l = len(im2write2[0])
    im2write2[0:10] = [[i*250/l for i in range(l)] for _ in range(10)]
    im2write2 = im2write2.astype('uint8')
    im2write3 = im3d[2]
    ust = im2write3.max()
    alt = im2write3.min()
    olcek = 250/(ust-alt)
    im2write3 = im2write3 + abs(alt)
    im2write3 = im2write3 * olcek
    l = len(im2write3[0])
    im2write3[0:10] = [[i*250/l for i in range(l)] for _ in range(10)]
    im2write3 = im2write3.astype('uint8')
    import matplotlib.image
    import numpy as np
    im2write = np.stack((im2write1,im2write2,im2write3), axis=2)
    matplotlib.image.imsave(impath,im2write)

def gor1c(img,ch=0,impath=None):
    if not impath:
        impath=f'/ari/users/madiyaman/mmdetection/temp/silgorch{ch}.png'
    im3d = img.cpu().detach().numpy()[0]
    im2write1 = im3d[ch]
    ust = im2write1.max()
    alt = im2write1.min()
    olcek = 250/(ust-alt)
    im2write1 = im2write1 + abs(alt)
    im2write1 = im2write1 * olcek
    l = len(im2write1[0])
    im2write1[0:10] = [[i*250/l for i in range(l)] for _ in range(10)]
    im2write1 = im2write1.astype('uint8')
    import matplotlib.image
    matplotlib.image.imsave(impath,im2write1)

def gor1cgt(img,ch=0,impath=None):
    if not impath:
        impath=f'/ari/users/madiyaman/mmdetection/temp/silgorchgt{ch}.png'
    im3d = img[0].cpu().numpy()
    im2write1 = im3d[ch]
    ust = im2write1.max()
    alt = im2write1.min()
    olcek = 250/(ust-alt)
    im2write1 = im2write1 + abs(alt)
    im2write1 = im2write1 * olcek
    l = len(im2write1[0])
    im2write1[0:10] = [[i*250/l for i in range(l)] for _ in range(10)]
    im2write1 = im2write1.astype('uint8')
    import matplotlib.image
    matplotlib.image.imsave(impath,im2write1)

def gor3cnb(img,impath='/ari/users/madiyaman/mmdetection/temp/gor3cnb.png'):
    im3d = img.cpu().detach().numpy()
    im2write1 = im3d[0]
    ust = im2write1.max()
    alt = im2write1.min()
    olcek = 250/(ust-alt)
    im2write1 = im2write1 + abs(alt)
    im2write1 = im2write1 * olcek
    l = len(im2write1[0])
    im2write1[0:10] = [[i*250/l for i in range(l)] for _ in range(10)]
    im2write1 = im2write1.astype('uint8')
    im2write2 = im3d[1]
    ust = im2write2.max()
    alt = im2write2.min()
    olcek = 250/(ust-alt)
    im2write2 = im2write2 + abs(alt)
    im2write2 = im2write2 * olcek
    l = len(im2write2[0])
    im2write2[0:10] = [[i*250/l for i in range(l)] for _ in range(10)]
    im2write2 = im2write2.astype('uint8')
    im2write3 = im3d[2]
    ust = im2write3.max()
    alt = im2write3.min()
    olcek = 250/(ust-alt)
    im2write3 = im2write3 + abs(alt)
    im2write3 = im2write3 * olcek
    l = len(im2write3[0])
    im2write3[0:10] = [[i*250/l for i in range(l)] for _ in range(10)]
    im2write3 = im2write3.astype('uint8')
    import matplotlib.image
    import numpy as np
    im2write = np.stack((im2write1,im2write2,im2write3), axis=2)
    matplotlib.image.imsave(impath,im2write)