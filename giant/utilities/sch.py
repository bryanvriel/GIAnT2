
import numpy as np
import tsinsar.tsio as tsio
import sys
import os

WGS84_A = 6378137.0
WGS84_E2 = 0.0066943799901
WGS84_B = np.sqrt(WGS84_A**2 * (1.0 - WGS84_E2))

class GenericClass:
    pass

def llh2xyz(lat, lon, h):
    """
    Computes ECEF XYZ coordinates (1x3 NumPy array) for a given llh
    X = llh2xyz(lat, lon, h).

    Arguments:
    lat         array of latitude values in radians
    lon         array of longitude values in radians
    h           array of elevations in meters

    Output:
    X           ECEF position vector
    """

    # Radius of curvature in east-west direction
    Nh = WGS84_A / np.sqrt(1.0 - WGS84_E2*np.sin(lat)**2)
    
    # XYZ array
    X = np.array([(Nh+h)*np.cos(lat)*np.cos(lon),
                  (Nh+h)*np.cos(lat)*np.sin(lon),
                  (Nh+h-WGS84_E2*Nh)*np.sin(lat)], dtype=np.float64)

    return X

def utc2xyz(rhoT, lat, lon):
    """
    Transforms topocentric UTC coordinates to ECEF XYZ coordinates
    rho = utc2xyz(rhoT, lat, lon).

    Arguments:
    rhoT        topocentric UTC position vector
    lat         latitude of station in radians
    lon         longitude of station in radians

    Output:
    rho         geocentric ECEF position vector
    """ 

    # Compute trigonometric operations for later use
    clat = np.cos(lat)
    clon = np.cos(lon)
    slat = np.sin(lat)
    slon = np.sin(lon)

    u = rhoT[0]
    t = rhoT[1]
    c = rhoT[2]
    
    # ECEF XYZ
    rho = np.array([-slon*u - slat*clon*t + clat*clon*c,
                     clon*u - slat*slon*t + clat*slon*c,
                     clat*t + slat*c], dtype=np.float64)

    return rho

def xyz2llh(X):
    """
    Computes lat/lon/h for given ECEF XYZ coordinates (Nx3 NumPy array)

    Argument:
    X       geocentric ECEF position vector

    Output:
    lat     array of latitude values in radians
    lon     array of longitude values in radians
    h       array of elevation values in meters
    """

    x = X[...,0]
    y = X[...,1]
    z = X[...,2]
    
    # Compute longitude
    lon = np.arctan2(y,x)

    # Compute latitude
    p = np.sqrt(x*x + y*y)
    alpha = np.arctan(z*WGS84_A / (p*WGS84_B))
    numer = z + WGS84_E2 / (1.0 - WGS84_E2)*WGS84_B*np.sin(alpha)**3
    denom = p - WGS84_E2*WGS84_A*np.cos(alpha)**3
    lat = np.arctan(numer/denom)

    # Compute height above ellipsoid
    h = p/np.cos(lat) - WGS84_A / np.sqrt(1.0 - WGS84_E2*np.sin(lat)**2)

    return lat,lon,h

def peg_fill(peg):
    """
    Updates the peg parameters given peg lat/lon/hdg
    """

    # Compute peg parameters
    peg.pos = llh2xyz(peg.lat, peg.lon, 0.0)
    peg.re = WGS84_A / np.sqrt(1.0 - WGS84_E2*np.sin(peg.lat)**2)
    peg.rn = WGS84_A * (1.0 - WGS84_E2) / np.sqrt((1.0 - WGS84_E2*np.sin(peg.lat)**2)**3)
    peg.ra = peg.re*peg.rn / (peg.re*np.cos(peg.hdg)**2 + peg.rn*np.sin(peg.hdg)**2)

    # ECEF XYZ coordinates of tangent sphere
    temp_raU = np.array([0, 0, peg.ra], dtype=np.float64)
    peg.raU = utc2xyz(temp_raU, peg.lat, peg.lon)

def llh2sch(lat, lon, h, peg):
    """
    Converts geocentric latitude/longitude/height to SCH
    Default parameters are for WGS-84 ellipsoid

    SCH = llh2sch(lat, lon, h, peg)

    lat/lon/h may be 1xN NumPy arrays, where N = number of observations
        -> Will return an Nx3 array for SCH data
    """

    # Compute trigonometric operations for later use
    ceta = np.cos(peg.hdg)
    seta = np.sin(peg.hdg)
    clat = np.cos(lat)
    slat = np.sin(lat)
    clon = np.cos(lon)
    slon = np.sin(lon)
    clat_peg = np.cos(peg.lat)
    slat_peg = np.sin(peg.lat)
    clon_peg = np.cos(peg.lon)
    slon_peg = np.sin(peg.lon)

    # Ellipsoid parameters
    Nh = WGS84_A/np.sqrt(1.0-WGS84_E2*slat*slat)

    # Convert current llh point to WGS-84 XYZ
    if isinstance(lat, np.ndarray): #check if lat is array
        X = np.vstack(((Nh+h)*clat*clon, (Nh+h)*clat*slon, (Nh+h-WGS84_E2*Nh)*slat))
        X = np.transpose(X)
    else:
        X = np.array([(Nh+h)*clat*clon, (Nh+h)*clat*slon, (Nh+h-WGS84_E2*Nh)*slat], dtype=np.float64)

    # Temp position vector
    temp = X - peg.pos + peg.raU

    # Transformation matrix from ECEF to UTC
    T1 = np.array([clat_peg*clon_peg, clat_peg*slon_peg, slat_peg], dtype=np.float64)
    T2 = np.array([-seta*slon_peg-ceta*clon_peg*slat_peg, 
                    clon_peg*seta-ceta*slat_peg*slon_peg, ceta*clat_peg], dtype=np.float64)
    T3 = np.array([ceta*slon_peg-seta*clon_peg*slat_peg, 
                  -clon_peg*ceta-seta*slat_peg*slon_peg, clat_peg*seta], dtype=np.float64)

    # Compute UTC coordinates
    if isinstance(lat, np.ndarray):
        u = np.sum(T1*temp,1)
        t = np.sum(T2*temp,1)
        c = np.sum(T3*temp,1)
    else:
        u = np.sum(T1*temp)
        t = np.sum(T2*temp)
        c = np.sum(T3*temp)

    # Compute SCH coordinates
    clambda = np.arctan(c/np.sqrt(u**2 + t**2))
    stheta = np.arctan(t/u)
    temp = np.transpose(np.vstack((peg.ra*stheta, peg.ra*clambda, h)))
    if isinstance(lat, np.ndarray):
        SCH = temp
    else:
        SCH = temp[0]

    return SCH

def sch2llh(SCH, peg):
    """
    Converts SCH coordinates to geocentric latitude/longitude/height
    Default parameters are for WGS-84 ellipsoid

    lat,lon,h = sch2llh(SCH, peg)

    SCH may be a 2-D NumPy array -> will return lat/lon/h arrays
    """

    # Compute trigonometric operations for later use
    ceta = np.cos(peg.hdg)
    seta = np.sin(peg.hdg)
    clat_peg = np.cos(peg.lat)
    slat_peg = np.sin(peg.lat)
    clon_peg = np.cos(peg.lon)
    slon_peg = np.sin(peg.lon)

    # Construct UTC vector from SCH coordinates
    clambda = SCH[...,1] / peg.ra
    stheta = SCH[...,0] / peg.ra
    h = SCH[...,2]
    R = peg.ra + h
    utc = np.array([R*np.cos(clambda)*np.cos(stheta), 
                    R*np.cos(clambda)*np.sin(stheta), 
                    R*np.sin(clambda)], dtype=np.float64)
    utc = np.transpose(utc)

    # Transformation matrix from UTC to ECEF
    T1 = np.array([clat_peg*clon_peg, -slat_peg*ceta*clon_peg-seta*slon_peg, 
                  -seta*slat_peg*clon_peg+slon_peg*ceta], dtype=np.float64)
    T2 = np.array([slon_peg*clat_peg, -slat_peg*slon_peg*ceta+seta*clon_peg, 
                  -seta*slat_peg*slon_peg-ceta*clon_peg], dtype=np.float64)
    T3 = np.array([slat_peg, ceta*clat_peg, seta*clat_peg], dtype=np.float64)

    # Transform UTC coordinates to ECEF XYZ
    if (utc[...,0].size > 1):
        X = np.vstack((np.sum(T1*utc,1), np.sum(T2*utc,1), np.sum(T3*utc,1)))
        X = np.transpose(X) + peg.pos - peg.raU
    else:
        X = np.array([np.sum(T1*utc), np.sum(T2*utc), np.sum(T3*utc)], dtype=np.float64) + peg.pos - peg.raU

    # Compute llh from ECEF XYZ
    return xyz2llh(X)

def sch2ijh(SCH, peg):
    """
    Convert SCH coordinates to range-azimuth-height coordinates. Also returns
    the incidence angle. SCH will be a Nx3 array, where N = number of observations.
    """

    # Range coordinates
    thetac = SCH[...,1] / peg.ra
    x1 = peg.ra + peg.alt
    x2 = peg.ra
    range = np.sqrt(x1**2 + x2**2 - 2.0*x1*x2*np.cos(thetac))
    jj = np.asarray(((range - peg.r0) / peg.dr), dtype=int)

    # Azimuth coordinates
    dt = SCH[...,0] / peg.Vs
    dt = (peg.utc + dt) - peg.utc0
    ii = np.asarray(dt / peg.d_utc, dtype=int)

    # Incidence angle
    look = np.arccos((x1**2 + range**2 - x2**2) / (2*x1*range))
    inc = np.arcsin(np.sin(look) * x1/x2)

    return ii,jj,inc

def getrsc_par(rscfile):
    """
    Read rsc file for viewing geometry-related parameters.
    Returns latitude and longitude bounds of image and a structure containing
    geometry parameters.
    """

    rad = np.pi / 180.0

    # Read in example *.rsc file
    latlim = []
    lonlim = []
    par = GenericClass()
    rdict = tsio.read_rsc(rscfile)
    
    # Extract relevant parameters
    latlim.append(float(rdict['LAT_REF1']))
    latlim.append(float(rdict['LAT_REF2']))
    latlim.append(float(rdict['LAT_REF3']))
    latlim.append(float(rdict['LAT_REF4']))
    lonlim.append(float(rdict['LON_REF1']))
    lonlim.append(float(rdict['LON_REF2']))    
    lonlim.append(float(rdict['LON_REF3']))
    lonlim.append(float(rdict['LON_REF4']))

    par.d_utc = float(rdict['DELTA_LINE_UTC'])
    par.hdg = float(float(rdict['HEADING'])) * rad
    par.lat = float(float(rdict['LATITUDE'])) * rad
    par.lon = float(float(rdict['LONGITUDE'])) * rad
    par.dr = float(rdict['RANGE_PIXEL_SIZE'])
    par.r0 = float(rdict['STARTING_RANGE'])
    #par.r0 = float(rdict['RAW_DATA_RANGE'])
    par.utc0 = float(rdict['FIRST_LINE_UTC'])
    par.alt = float(rdict['HEIGHT'])
    par.Vs = float(rdict['VELOCITY_S'])
    par.utc = float(rdict['PEG_UTC'])

    latlim = [min(latlim), max(latlim)]
    lonlim = [min(lonlim), max(lonlim)]

    # Fill out other peg fields
    peg_fill(par)

    return latlim, lonlim, par

############################################################
# Program is part of GIAnT v1.0                            #
# Copyright 2012, by the California Institute of Technology#
# Contact: earthdef@gps.caltech.edu                        #
############################################################
