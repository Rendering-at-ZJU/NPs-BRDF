#Read BRDF
import numpy as np
import os.path as path

def readUTIABRDF(filename):
    """Reads a UTIA-type .bin file, containing a densely sampled BRDF
    
    Returns a 5-dimensional array (phi_v, theta_v, phi_i, theta_i, channel)"""

    print("Loading UTIA-BRDF: ",filename)
    try: 
        f = open(filename, "rb")
        vals = np.fromfile(f,np.float64,-1)
        f.close()
    except IOError:
        print("Cannot read file:", path.basename(filename))
        return
        
    BRDFVals = np.reshape(vals,(48, 6, 48, 6, 3),'F')
    BRDFVals[BRDFVals<0] = 0
    
    return BRDFVals

def readMERLBRDF(filename):
    """Reads a MERL-type .binary file, containing a densely sampled BRDF
    
    Returns a 4-dimensional array (phi_d, theta_h,theta_d channel)"""
    print("Loading MERL-BRDF: ",filename)
    try: 
        f = open(filename, "rb")
        dims = np.fromfile(f,np.int32,3)
        vals = np.fromfile(f,np.float64,-1)
        f.close()
    except IOError:
        print("Cannot read file:", path.basename(filename))
        return
        
    BRDFVals = np.swapaxes(np.reshape(vals,(dims[2], dims[1], dims[0], 3),'F'),1,2)
    BRDFVals *= (1.00/1500,1.15/1500,1.66/1500) #Colorscaling
    BRDFVals[BRDFVals<0] = 0
    
    return BRDFVals
    
def saveMERLBRDF(filename,BRDFVals,shape=(180,90,90),toneMap=True):
    "Saves a BRDF to a MERL-type .binary file"
    print("Saving MERL-BRDF: ",filename)
    BRDFVals = np.array(BRDFVals)   #Make a copy
    if(BRDFVals.shape != (np.prod(shape),3) and BRDFVals.shape != shape+(3,)):
        print("Shape of BRDFVals incorrect")
        return
        
    #Do MERL tonemapping if needed
    if(toneMap):
        BRDFVals /= (1.00/1500,1.15/1500,1.66/1500) #Colorscaling
    
    #Are the values not mapped in a cube?
    if(BRDFVals.shape[1] == 3):
        BRDFVals = np.reshape(BRDFVals,shape+(3,))
        
    #Vectorize:
    vec = np.reshape(np.swapaxes(BRDFVals,1,2),(-1),'F')
    shape = [shape[2],shape[1],shape[0]]
    
    try: 
        f = open(filename, "wb")
        np.array(shape).astype(np.int32).tofile(f)
        vec.astype(np.float64).tofile(f)
        f.close()
    except IOError:
        print("Cannot write to file:", path.basename(filename))
        return
        
    

def saveUTIABRDF(filename,BRDFVals,shape=(48, 6, 48, 6, 3)):
    "Saves a BRDF to an UTIA-type .binary file"
    print("Saving UTIA-BRDF: ",filename)
    BRDFVals = np.array(BRDFVals)   #Make a copy
    
    #Vectorize:
    vec = np.reshape(BRDFVals.reshape(shape),(-1),'F')
    
    try: 
        f = open(filename, "wb")
        vec.astype(np.float64).tofile(f)
        f.close()
    except IOError:
        print("Cannot write to file:", path.basename(filename))
        return
        