import sys
import os
import math
import numpy
import logging
import time
import galsim
import struct
import random
import json
import subprocess
import argparse
import pathlib

class PSF:
    def __init__(self, beta = 3, fwhm = 2.85, e1 = -0.019, e2 = -0.007):
        self.beta = beta
        self.fwhm = fwhm
        self.trunc = 2 * fwhm
        self.e1 = e1
        self.e2 = e2

class GAL:
    def __init__(self, signalToNoise = 200, resolution = 0.98, ellipRMS = 0.2, ellipMax = 0.6, shiftRadius = 1.0, g1 = 0.013, g2 = -0.008):
        self.signalToNoise = signalToNoise
        self.resolution = resolution
        self.ellipRMS = ellipRMS
        self.ellipMax = ellipMax
        self.shiftRadius = shiftRadius
        self.shiftRadiusSQ = shiftRadius ** 2
        self.g1 = g1
        self.g2 = g2

class GalWrapper:
    def __init__(self, width = 40, height = 40, pixelScale = 1.0, skyLevel = 1.e6, randomShear = False, addNoise = True):
        self.width = width
        self.height = height
        self.scale = pixelScale
        self.skyLevel = skyLevel
        self.psfData = PSF()
        self.galData = GAL()
        self.randomShear = randomShear
        self.addNoise = addNoise


    def setPSF(self, beta, fwhm, e1, e2):
        self.psfData = PSF(beta=beta, fwhm=fwhm, e1=e1, e2=e2)

    def setPSF_Beta(self, beta):
        self.psfData.beta = beta

    def setPSF_FWHM(self, fwhm):
        self.psfData.fwhm = fwhm

    def setPSF_E1(self, e1):
        self.psfData.e1 = e1

    def setPSF_E2(self, e2):
        self.psfData.e2 = e2

    def setGAL(self, signalToNoise, resolution, ellipRMS, ellipMax, shiftRadius, g1, g2):
        self.galData = GAL(signalToNoise=signalToNoise, resolution=resolution, ellipRMS=ellipRMS, ellipMax=ellipMax, shiftRadius=shiftRadius, g1=g1, g2=g2)
    
    def setGAL_SNR(self, signalToNoise):
        self.galData.signalToNoise = signalToNoise

    def setGAL_Resolution(self, resolution):
        self.galData.resolution = resolution

    def setGAL_EllipRMS(self, ellipRMS):
        self.galData.ellipRMS = ellipRMS

    def setGAL_EllipMax(self, ellipMax):
        self.galData.ellipMax = ellipMax
    
    def setGAL_ShiftRadius(self, shiftRadius):
        self.galData.shiftRadius = shiftRadius
        self.galData.shiftRadiusSQ = shiftRadius ** 2

    def setGAL_G1(self, g1):
        self.galData.g1 = g1

    def setGAL_G2(self, g2):
        self.galData.g2 = g2

    def randomFloat(self, min, max):
        rand = (random.random() * (max - min)) + min

        return rand

    def __generatePSF(self):
        # Define the PSF profile
        self.psf = galsim.Moffat(beta=self.psfData.beta, fwhm=self.psfData.fwhm, trunc=self.psfData.trunc)

        e1 = self.psfData.e1
        e2 = self.psfData.e2
        re = self.psf.half_light_radius

        if self.randomShear:
            e1 = self.randomFloat(-0.04, 0.04)
            e2 = self.randomFloat(-0.04, 0.04)

        self.psf = self.psf.shear(e1 = e1, e2 = e2)
        return (e1, e2, re)

    def __generateGAL(self, re):
        self.gal = galsim.Exponential(flux=1.0, half_light_radius=re * self.galData.resolution)

    def __generateDeviates(self):
        # Generate random deviates
        ud = galsim.UniformDeviate(struct.unpack('i', os.urandom(4))[0])
        gd = galsim.GaussianDeviate(ud, sigma=self.galData.ellipRMS)

        return (ud, gd)

    def __generateSubImages(self, row, column):
        # - 1 for border of 1 pixel between stamps
        b = galsim.BoundsI(column * self.width + 1, (column + 1) * self.width - 1,
                            row * self.height + 1, (row + 1) * self.height - 1)

        subGalImage = self.galImage[b]
        subPSFImage = self.psfImage[b]

        return (subGalImage, subPSFImage)

    def __generateEllipticGalaxy(self, ud, gd):
        e1, e2, re = self.__generatePSF()
        self.__generateGAL(re)

        # Use a random orientation:
        beta = ud() * 2. * math.pi * galsim.radians

        # Generate gravity shear
        g1 = self.galData.g1
        g2 = self.galData.g2 
        
        if self.randomShear:
            g1 = self.randomFloat(-0.5, 0.5)
            g2 = self.randomFloat(-0.5, 0.5)

        # Determine the ellipticity to use for this galaxy.
        ellip = 1
        while (ellip > self.galData.ellipMax):
            # Don't do `ellip = math.fabs(gd())`
            # Python basically implements this as a macro, so gd() is called twice!
            val = gd()
            ellip = math.fabs(val)

        # Make a new copy of the galaxy with an applied e1/e2-type distortion
        # by specifying the ellipticity and a real-space position angle
        ellip_gal = self.gal.shear(e=ellip, beta=beta)
        
        # Gravitational shear
        ellip_gal = ellip_gal.shear(g1=g1, g2=g2)

        return (ellip_gal, e1, e2, g1, g2)

    def __generateRandomShift(self, ud):
        rsq = 2 * self.galData.shiftRadiusSQ
        dx = 1
        dy = 1

        while (rsq > self.galData.shiftRadiusSQ):
            dx = (2 * ud() - 1) * self.galData.shiftRadius
            dy = (2 * ud() - 1) * self.galData.shiftRadius

            rsq = dx ** 2 + dy ** 2

        return (rsq, dx, dy)

    def __addNoise(self, subGalImage, ud):
        skyLevelPixel = self.skyLevel * self.scale ** 2
        noise = galsim.PoissonNoise(ud, sky_level=skyLevelPixel)
        subGalImage.addNoiseSNR(noise, self.galData.signalToNoise)

    def generateImage(self, amount, outputFolder, psfFilename, galFilename, jsonFilename):
        jsonList = []

        for i in range(amount): 
            self.galImage = galsim.ImageF(self.width, self.height, scale=self.scale)
            self.psfImage = galsim.ImageF(self.width, self.height, scale=self.scale)

            psfBase = f"{psfFilename}_{i}"
            galBase = f"{galFilename}_{i}"
            psfFitsPath = os.path.join(outputFolder, f"{psfBase}.fits")
            galFitsPath = os.path.join(outputFolder, f"{galBase}.fits")
            psfTifPath = os.path.join(outputFolder, f"{psfBase}.tif")
            galTifPath = os.path.join(outputFolder, f"{galBase}.tif")

            # Generate random deviates
            ud, gd = self.__generateDeviates()

            # Random elliptict galaxy warped by gravity
            tempGal, e1, e2, g1, g2 = self.__generateEllipticGalaxy(ud, gd)
            
            # Random shift
            _, dx, dy = self.__generateRandomShift(ud)
            tempGal = tempGal.shift(dx, dy)
            tempPSF = self.psf.shift(dx, dy)

            finalGal = galsim.Convolve([self.psf, tempGal])
            finalGal.drawImage(self.galImage)
            tempPSF.drawImage(self.psfImage)

            if self.addNoise:
                self.__addNoise(self.galImage, ud)

            jsonList.append({
                "galaxy": f"{galBase}.tif",
                "psf": f"{psfBase}.tif",
                "e1": e1,
                "e2": e2,
                "g1": g1,
                "g2": g2
            })

            self.psfImage.write(psfFitsPath)
            self.galImage.write(galFitsPath)

            subprocess.check_call(["stiff", psfFitsPath])
            os.rename("stiff.tif", psfTifPath)

            subprocess.check_call(["stiff", galFitsPath])
            os.rename("stiff.tif", galTifPath)

        jsonFile = open(os.path.join(outputFolder, jsonFilename + ".json"), "w")
        jsonFile.write(json.dumps(jsonList))
        jsonFile.close()           
        os.remove("stiff.xml")

def mkdir(directory):
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--number", required=True, help="Amount of images to generate.")
ap.add_argument("-o", "--output", required=True, help="Output folder.")
ap.add_argument("--noise", required=False, action="store_true", help="Add noise to the images.")
args = vars(ap.parse_args())         

if not os.path.isdir(args["output"]):
    mkdir(args["output"])

wrapper = GalWrapper(addNoise=args["noise"], randomShear=True)
wrapper.generateImage(int(args["number"]), args["output"], "psf", "gal", "data")