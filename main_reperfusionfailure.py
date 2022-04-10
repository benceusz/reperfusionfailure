#!/usr/bin/env python
# coding: utf-8

# In[101]:


### DECLARATIONS AND IMPORTS ###
import os
from nipype.interfaces import fsl
from nipype.interfaces.freesurfer import MRIConvert
import ants
import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import warnings
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')

PATH_MNI_BRAINMASK = ref_img_mni = os.path.join(os.path.dirname(os.path.abspath("__file__")) , "mni152_brainmask.nii.gz")

### HELPER FUNCTIONS ###
def get_name_of_folder(a_path):
    without_extra_slash = os.path.normpath(a_path)
    last_part = os.path.basename(without_extra_slash)
    return last_part

def get_name_of_patient(a_path):
    a_path = os.path.join(a_path, '..')
    without_extra_slash = os.path.normpath(a_path)
    last_part = os.path.basename(without_extra_slash)
    return last_part

def get_name_of_visit(a_path):
    without_extra_slash = os.path.normpath(a_path)
    last_part = os.path.basename(without_extra_slash)
    pattern = "_(.*?)\_"
    substring = re.search(pattern, last_part).group(1)
    return substring

# save to hemisphere masks to nifti
def save_nifti(np_array, filename, output_folder,  original_img ):
    """
    Saves nifti file
    np_array: input 3D numpy matrix without header
    filename: example: "cbf" then the output name will be cbf.nii.gz
    original img: helper image that passes the header and affine parameters
    output folder:
    """
    original_img = nb.load(PATH_MASK_MRI)

    filepath = OUTPUT_FOLDER + '/' + filename + '.nii.gz'
    
    if not os.path.isfile(filepath):
        # array_data = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
        # affine = np.diag([1, 2, 3, 1])
        nii_img = nb.nifti1.Nifti1Image(np_array, original_img.affine , header=original_img.header)
        nb.save(nii_img, filepath)
        print('Image has been saved to {}'.format(filepath))
    else:
        print("Image already exists!")
        
# run flirt with default 6 dof
def run_flirt(path_infile, 
              out_file,
              path_reference, 
              dof, 
              bins = 256, 
              out_matrix_file = 'transformation_matrix.mat', 
              cost = 'corratio',
              interp = 'trilinear'):
    
    #out_file_cbf = 'mni_'+os.path.basename(path_infile)[:-7] + '_to_' + os.path.basename(path_reference)[:-7]+'_dof'+ str(dof) + '.nii.gz'

    print("flirt coregistration for file: {}".format(out_file))

    flt = fsl.FLIRT(bins=bins, 
                    cost_func=cost, 
                    interp = interp, 
                    in_file = path_infile,
                    reference = path_reference,
                    out_file = out_file,
                    dof = dof,
                    out_matrix_file = os.path.join( os.path.dirname(os.path.realpath(out_file)), out_matrix_file),
                    output_type = "NIFTI_GZ")

    if not os.path.isfile(out_file):
        res = flt.run()
        print("Coregistration has saved to: {}".format(out_file))
    else:
        print("Flirt coregistration already exists!")
    
    return out_file

        
# function for BET brain extraction
def run_bet(input_file_path):
    """
    input:
    input_file_path 
    
    output:
    path_outfile_bet: path of the resulted brain extracted file
    path_betmask: full file path of the binary brain extraction mask
    """
    btr = fsl.BET()
    OUTPUT_FOLDER = os.path.dirname(input_file_path)

    _out_file_bet = 'bet_' + os.path.basename(input_file_path)[:-7] + '.nii.gz'
    path_outfile_bet = os.path.join(OUTPUT_FOLDER, _out_file_bet)
    
    if not os.path.isfile(path_outfile_bet):
        result = btr.run(in_file= input_file_path, out_file=path_outfile_bet, frac=0.7 , mask = True)
    else:
        print("Bet file already exists")
    
    # path of the binary mask file
    _filename_mask = 'bet_' + os.path.basename(input_file_path)[:-7] + '_mask.nii.gz'
    path_betmask = os.path.join(OUTPUT_FOLDER, _filename_mask)
    return path_outfile_bet, path_betmask

def save_nifti(np_array, filename, output_folder,  original_img ):
    """
    Saves nifti file
    np_array: input 3D numpy matrix without header
    filename: example: "cbf" then the output name will be cbf.nii.gz
    original img: helper image that passes the header and affine parameters
    output folder:
    """
    original_img = nb.load(original_img)

    filepath = output_folder + '/' + filename + '.nii.gz'
    
    if not os.path.isfile(filepath):
        # array_data = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
        # affine = np.diag([1, 2, 3, 1])
        nii_img = nb.nifti1.Nifti1Image(np_array, original_img.affine , header=original_img.header)
        nb.save(nii_img, filepath)
        print('Image has been saved to {}'.format(filepath))
    else:
        print("Image already exists!")

def nativ_in_folder(path_dir, filename = "nativ.nii.gz"):    
    match = 0
    for root, dir, files in os.walk(path_dir):
        for i_file in files:
            if i_file == "nativ.nii.gz":
                print(i_file)

                match = 1
                break
            else:
                match = 0
    return match    


def subdirs(path):
    """Yield directory names not starting with '.' under given path."""
    for entry in os.scandir(path):
        if not entry.name.startswith('.') and entry.is_dir():
            yield entry.name
            
def run_ct_coreg(DATA_DIR):
    print("Running ct coregistration to mri in study: %s" % DATA_DIR)
    
    """"""
        # constants
    os.chdir(DATA_DIR)
    bins = 256
    PATH_MNI_BRAINMASK = ref_img_mni = '/home/nraresearch/research/perfusion_integral/data/crpps_perfusionintegral/ALLCRPP_T1_CBF_MOCO_T1MASKS/mni152_brainmask.nii.gz'
    #PATH_T1_BRAINMASK = os.path.join(DATA_DIR, "t1_masked_with_aseg.nii.gz")
    # coregistration to original t1 instead of brain extracted t1
    t1filename = [filename for filename in os.listdir(DATA_DIR) if filename.startswith("t1")]
    PATH_T1_BRAINMASK = os.path.join(DATA_DIR,t1filename[0])
    PATH_MASK_PENUMBRA = os.path.join(DATA_DIR, "coregt1", "coregt1_mask_penumbra_bl.nii.gz")

    # specify all the possible sequence names that the algorihtm should iterate on
    # more names can be listed. Only exact matches will be processed
    PERFUSION_FILES = [ "CBFD","CBVD","MIP", "MTTD", "TMAXD","TTDD", "TTPM"]
    MASK_FILES = ["mask_core", "mask_penumbra", "mask_penumbra_bl", "mask_penumbra_v1", "mask_penumbra_bl_4-6"]
    DWI_FILES = []
    NIFTI_FILES = []
    BOLD_FILES = []
    SELECTED_TO_ANALYSIS = ["CBFD","CBVD","TMAXD"]
    """
    PERFUSION_FILES = [ "MTT", "rBF", "rBV", "TMAX" , "TTP", "tMIP", "tMIPS"]
    DWI_FILES = ["ADC","IVIM_ADC", "IVIM_TRACEW_1","IVIM_TRACEW_2","IVIM_TRACEW_3","IVIM_TRACEW_4","IVIM_TRACEW_5","IVIM_TRACEW_B5", "B1000","B1000_NATIVE","Bxxxx" ]
    NIFTI_FILES = ["FLAIR","T2", "BOLD" ]
    BOLD_FILES = ["CVRmap", "CVRmap", "CVRMAP"]
    """

    # coregister the not perfusion nifti files NIFTI_FILES = ["ADC","IVIM_ADC, IVIM_TRACEW_B5"] 
    os.chdir(DATA_DIR)
    # path = "./" + dir_resliced_name
    path = "./original"

    try:
        os.chdir(path)
        print("Current working directory: {0}".format(os.getcwd()))
    except FileNotFoundError:
        print("Directory: {0} does not exist".format(path))
    except NotADirectoryError:
        print("{0} is not a directory".format(path))
    except PermissionError:
        print("You do not have permissions to change to {0}".format(path))

    # create coregt1 folder to store the t1 space coregistered nifti files
    dir_coreg_name = "coregt1"
    path_dir_coregt= os.path.join(os.getcwd(),"../"+dir_coreg_name)
    if not os.path.isdir(path_dir_coregt):
        os.mkdir(path_dir_coregt)


    # coregistrate rBV and store transformation matrix
    tag_base_file = "MIP"

    #PATH_BASE_PERF = os.path.abspath("./r_" +tag_base_file +".nii.gz")
    PATH_BASE_PERF = os.path.abspath("./" +tag_base_file +".nii.gz")

    if os.path.isfile(PATH_BASE_PERF):

        path_matrix_perf = os.path.join( os.getcwd(), "../" + dir_coreg_name  + "/" + dir_coreg_name +"_" + tag_base_file +"_2_t1matrix.mat")

        print("Baseline perfusion file is {}".format(PATH_BASE_PERF))
        path_out_file = os.path.join(os.getcwd(), "../" + dir_coreg_name  + "/" + dir_coreg_name +"_" + tag_base_file  + ".nii.gz")
        """
        PATH_CBF_2_BET_T1= run_flirt(path_infile = PATH_BASE_PERF,
                                     out_file=path_out_file,
                                     path_reference = PATH_T1_BRAINMASK,
                                     out_matrix_file= path_matrix_perf,
                                     dof = 6)
        """
        if not os.path.isfile(path_out_file):
            mr_t1 = ants.image_read(PATH_T1_BRAINMASK)
            perf_mip =  ants.image_read(PATH_BASE_PERF)
            registration = ants.registration(fixed = mr_t1 , moving = perf_mip, type_of_transform = 'Rigid' )
            path_out_file = os.path.join(os.getcwd(), "../" + dir_coreg_name  + "/" + dir_coreg_name +"_" + tag_base_file  + ".nii.gz")
            ants.image_write(registration['warpedmovout'], path_out_file)

        """
        txfile = ants.affine_initializer( fi, mi )
        tmatrix = ants.read_transform(txfile, dimension=2)
        """
        # iterate through perfusion maps
        for i_img in PERFUSION_FILES + MASK_FILES : 
            path_out_file = os.path.join(os.getcwd(), "../" + dir_coreg_name  + "/" + dir_coreg_name +"_" + i_img + ".nii.gz")

            if not os.path.isfile(path_out_file):
                try:
                    print("processing file %s" % i_img)

                    path_out_file = os.path.join(os.getcwd(), "../" + dir_coreg_name  + "/" + dir_coreg_name +"_" + i_img + ".nii.gz")
                    moving = ants.image_read(os.path.join(os.getcwd(),i_img + '.nii.gz'))

                    mywarpedimage = ants.apply_transforms( fixed=mr_t1, moving=moving, interpolator = 'nearestNeighbor', transformlist=registration['fwdtransforms'] )
                    ants.image_write(mywarpedimage, path_out_file)

                    """
                    path_infile = os.path.join(os.getcwd(),"r_" + i_img + '.nii.gz')
                    print("Processing {}".format(i_img))
                    applyxfm = fsl.preprocess.ApplyXFM()
                    applyxfm.inputs.in_file = path_infile
                    applyxfm.inputs.in_matrix_file = path_matrix_perf
                    applyxfm.inputs.out_file = path_out_file
                    applyxfm.inputs.reference = PATH_T1_BRAINMASK
                    applyxfm.inputs.apply_xfm = True
                    result = applyxfm.run()
                    """
                except:
                    # print("Failure at processing file %s" % i_img)
                    pass
            else:
                print("File %s already exist " % path_out_file)


    else:
        print("MIP baseline file does not exist to calculate the transformation matrix for perfusion files. Choose a new baseline file or place the B1000 file in the original folder")

    # coregistrate bold, and other nifti files in different spaces (T2 , FLAIR etc)
    """
    for i_img in NIFTI_FILES:
        try:
            path_infile = os.path.join(os.getcwd(),"r_" + i_img + '.nii.gz')
            path_matrix_file = os.path.join( os.getcwd(), "../" + dir_coreg_name  + "/" + dir_coreg_name +"_" + i_img +"_2_t1matrix.mat")

            print("Coregistration of {} to T1".format(i_img))
            path_out_file = os.path.join(os.getcwd(), "../" + dir_coreg_name  + "/" + dir_coreg_name +"_" + i_img  + ".nii.gz")
            PATH_CBF_2_BET_T1= run_flirt(path_infile = path_infile,
                                         out_file=path_out_file,
                                         path_reference = PATH_T1_BRAINMASK,
                                         out_matrix_file= path_matrix_file,
                                         dof = 6)
        except:
            print("File %s could not be processed" % i_img)
    """

    # coregister all the NIFTI_FILES to MNI wiht transformation matrix between t1 and MNI152
    os.chdir(DATA_DIR)
    path = "./" + dir_coreg_name
    try:
        os.chdir(path)
        print("Current working directory: {0}".format(os.getcwd()))
    except FileNotFoundError:
        print("Directory: {0} does not exist".format(path))
    except NotADirectoryError:
        print("{0} is not a directory".format(path))
    except PermissionError:
        print("You do not have permissions to change to {0}".format(path))

    dir_mni_name = "mni"
    path_dir_mni= os.path.join(os.getcwd(),"../"+dir_mni_name)
    if not os.path.isdir(path_dir_mni):
        os.mkdir(path_dir_mni)


    # coregistrate T1 to MNI and store transofrmation matrix
    tag_base_file = "t1_masked_with_aseg"
    path_matrix_mni_file = os.path.join( os.getcwd(), "../" + dir_mni_name  + "/" + tag_base_file +"_2_mnimatrix.mat")

    print("Path of T1 file to be coregistered to MNI is {}".format(PATH_T1_BRAINMASK))
    path_out_file = os.path.join(os.getcwd(), "../" + dir_mni_name  + "/" + dir_mni_name +"_" + tag_base_file  + ".nii.gz")



    PATH_CBF_2_BET_T1= run_flirt(path_infile = PATH_T1_BRAINMASK,
                                 out_file=path_out_file,
                                 path_reference = PATH_MNI_BRAINMASK,
                                 out_matrix_file= path_matrix_mni_file,
                                 dof = 12)

    # iterate through all files and apply the transformation matrix calculated above between the patient's t1  and mni space
    for i_img in PERFUSION_FILES+ MASK_FILES: 
        path_out_file = os.path.join(os.getcwd(), "../" + dir_mni_name  + "/" + dir_mni_name +"_" + i_img + ".nii.gz")
        if not os.path.isfile(path_out_file):
            try:
                path_infile = os.path.join(os.getcwd(),"coregt1_" + i_img + '.nii.gz')
                print("Processing {}".format(i_img))
                applyxfm = fsl.preprocess.ApplyXFM()
                applyxfm.inputs.in_file = path_infile
                applyxfm.inputs.in_matrix_file = path_matrix_mni_file
                applyxfm.inputs.out_file = path_out_file
                #applyxfm.inputs.reference = PATH_T1_BRAINMASK
                applyxfm.inputs.reference = PATH_MNI_BRAINMASK
                applyxfm.inputs.apply_xfm = True
                result = applyxfm.run()            
            except:
                # print("Failure at processing file %s" % path_out_file)
                pass
        else:
            print("File %s already exist " % path_out_file)

#  read the coregistered CBF file
    PATH_TMAX = os.path.join(os.path.dirname(PATH_MASK_PENUMBRA),'coregt1_TMAXD.nii.gz' )
    #if os.path.exists(PATH_TMAX):
    print("Tmax exist")
    vol_tmax = nb.load(PATH_TMAX)
    np_vol_tmax = vol_tmax.get_fdata()

    PATH_RBV = os.path.join(os.path.dirname(PATH_MASK_PENUMBRA),'coregt1_CBVD.nii.gz' )
    if os.path.exists(PATH_RBV):
        vol_rbv = nb.load(PATH_RBV)
        np_vol_rbv = vol_rbv.get_fdata()
    
    PATH_RBF = os.path.join(os.path.dirname(PATH_MASK_PENUMBRA),'coregt1_CBFD.nii.gz' )
    if os.path.exists(PATH_RBF):
        vol_rbf = nb.load(PATH_RBF)
        np_vol_rbf = vol_rbf.get_fdata()

    PATH_MASK_PENUMBRA = os.path.join(DATA_DIR, "coregt1", "coregt1_mask_penumbra_bl.nii.gz")
    # load the penumbra masks
    if os.path.exists(PATH_MASK_PENUMBRA):
        vol_mask = nb.load(PATH_MASK_PENUMBRA)
        vol_mask.shape
        np_vol_mask = vol_mask.get_fdata()
        np_vol_mask[np_vol_tmax < 6] = 0
        np_vol_mask[np_vol_rbv < 0] = 0
        np_vol_mask[np_vol_rbf < 0]  = 0
        
    PATH_MASK_PENUMBRA46 = os.path.join(DATA_DIR, "coregt1", "coregt1_mask_penumbra_bl_4-6.nii.gz")
    # load the penumbra masks
    if os.path.exists(PATH_MASK_PENUMBRA46):
        vol_mask_penumbra46 = nb.load(PATH_MASK_PENUMBRA46)
        vol_mask.shape
        np_vol_mask_p46 = vol_mask_penumbra46.get_fdata()
        np_vol_mask_p46[np_vol_tmax < 6] = 0
        np_vol_mask_p46[np_vol_rbv < 0] = 0
        np_vol_mask_p46[np_vol_rbf < 0]  = 0    

    PATH_MASK_CORE = os.path.join(DATA_DIR, "coregt1", "coregt1_mask_core_v1.nii.gz")
    if os.path.exists(PATH_MASK_CORE):
        vol_mask_core = nb.load(PATH_MASK_CORE)
        vol_mask.shape
        np_vol_mask_core = vol_mask_core.get_fdata()
        np_vol_mask_core[np_vol_tmax < 6] = 0
        np_vol_mask_core[np_vol_rbv < 0] = 0
        np_vol_mask_core[np_vol_rbf < 0]  = 0

    columns = ['patient','visit']
    values = [get_name_of_patient(DATA_DIR), get_name_of_visit(DATA_DIR) ]
    values_p46 = [get_name_of_patient(DATA_DIR), get_name_of_visit(DATA_DIR) ]
    values_core = [get_name_of_patient(DATA_DIR), get_name_of_visit(DATA_DIR) ]

    # index = [get_name_of_folder(DATA_DIR) + str(os.path.basename(PATH_CBF_PERF)[4:-7])] 
    index = [get_name_of_folder(DATA_DIR)] 

    for i_seq in SELECTED_TO_ANALYSIS:    
        list_calculated = [i_seq+'_mean',
                        i_seq+'_std',
                        i_seq+'_min',
                        i_seq+'_max',
                        i_seq+'_median',
                        i_seq+'_q1',
                        i_seq+'_q3']
        columns = columns + list_calculated
        
        path_infile = os.path.join(DATA_DIR,"coregt1", "coregt1_" + i_seq + '.nii.gz')

        if os.path.isfile(path_infile):
            vol = nb.load(path_infile)
            np_vol = vol.get_fdata()
            
            if os.path.exists(PATH_MASK_PENUMBRA):
                # apply left hemisphere mask on flirt_cbf_to_bett1
                np_vol_masked = np.zeros(np_vol_mask.shape)
                np.putmask(np_vol_masked, np_vol_mask, np_vol)
                roi = np_vol_masked[np_vol_mask>0.5]       
                seq_values = [np.mean(roi), np.std(roi), np.min(roi), np.max(roi),np.median(roi),np.percentile(roi,25),np.percentile(roi,75) ]
                print(seq_values)
                
            if os.path.exists(PATH_MASK_PENUMBRA46):
                # apply left hemisphere mask on flirt_cbf_to_bett1
                np_vol_masked_p46  = np.zeros(np_vol_mask_p46 .shape)
                np.putmask(np_vol_masked_p46 , np_vol_mask_p46 , np_vol)
                roi_p46  = np_vol_masked_p46 [np_vol_mask_p46 >0.5]
                seq_values_p46 = [np.mean(roi_p46), np.std(roi_p46), np.min(roi_p46), np.max(roi_p46),np.median(roi_p46),np.percentile(roi_p46,25),np.percentile(roi_p46,75) ]
                print(seq_values_p46)
                
            if os.path.exists(PATH_MASK_CORE):
                # apply left hemisphere mask on flirt_cbf_to_bett1
                np_vol_masked_core = np.zeros(np_vol_mask_core.shape)
                np.putmask(np_vol_masked_core , np_vol_mask_core , np_vol)
                roi_core  = np_vol_masked_core [np_vol_mask_core >0.5]      
                seq_values_core = [np.mean(roi_core ), np.std(roi_core ), np.min(roi_core ), np.max(roi_core ),np.median(roi_core ),np.percentile(roi_core ,25),np.percentile(roi_core ,75) ]
                print(seq_values_core )
                
        else:
            # seq_values = np.full([1, len(list_calculated)],1)
            # seq_values = np.full([1, len(list_calculated)], 0)
            if os.path.exists(PATH_MASK_PENUMBRA):
                seq_values = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan ]
            if os.path.exists(PATH_MASK_PENUMBRA46):
                seq_values_p46 = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan ]
            if os.path.exists(PATH_MASK_CORE):
                seq_values_core = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan ]

        # values to the sequence
        if os.path.exists(PATH_MASK_PENUMBRA):
            values = values + seq_values
            df_penumbra= pd.DataFrame(index=index, columns=columns)
            df_penumbra.loc[index[0]] = values
            PATH_CSV_LOCAL = os.path.join(DATA_DIR, get_name_of_folder(DATA_DIR)+'_penumbra.csv')
            df_penumbra.to_csv(PATH_CSV_LOCAL, index=True, mode='w+', sep = ',')
            #df_penumbra.to_csv(PATH_GLOBAL_CSV_POSTOP, mode='a', header=False)
            # df_penumbra.to_csv(PATH_GLOBAL_CSV_CT_MRI, mode='a', header=False)

        if os.path.exists(PATH_MASK_PENUMBRA46):
            values_p46 = values_p46 + seq_values_p46
            df_p46= pd.DataFrame(index=index, columns=columns)
            df_p46.loc[index[0]] = values_p46
            PATH_CSV_LOCAL_p46 = os.path.join(DATA_DIR, get_name_of_folder(DATA_DIR)+'_p46.csv')
            df_p46.to_csv(PATH_CSV_LOCAL_p46, index=True, mode='w+', sep = ',')

        if os.path.exists(PATH_MASK_CORE):
            values_core = values_core + seq_values_core
            df_core= pd.DataFrame(index=index, columns=columns)
            df_core.loc[index[0]] = values_core
            PATH_CSV_LOCAL_core = os.path.join(DATA_DIR, get_name_of_folder(DATA_DIR)+'_core.csv')
            df_core.to_csv(PATH_CSV_LOCAL_core, index=True, mode='w+', sep = ',')
            

    
def run_mr_coreg(DATA_DIR):
    print("Running mr coregistration to mri in study: %s" % DATA_DIR)
    """
    PATH_GLOBAL_CSV_POSTOP = os.path.join(DATA_DIR, '..','..','values_all_postop.csv')
    PATH_GLOBAL_CSV_CT_MRI = os.path.join(DATA_DIR, '..','..','values_all.csv')
    """
    # CONSTANTS
    os.chdir(DATA_DIR)
    bins = 256
    PATH_T1_BRAINMASK = os.path.join(DATA_DIR, "t1_masked_with_aseg.nii.gz")
    PATH_MASK_PENUMBRA = os.path.join(DATA_DIR, "coregt1", "coregt1_mask_penumbra_bl.nii.gz")

    # specify all the possible sequence names that the algorihtm should iterate on
    # more names can be listed. Only exact matches will be processed
    PERFUSION_FILES = [ "MTT", "rBF", "rBV", "TMAX" , "TTP", "tMIP", "tMIPS"]
    MASK_FILES = ["mask_core_v1","mask_penumbra","mask_penumbra_bl", "mask_penumbra_v1", "mask_penumbra_bl_4-6"]
    DWI_FILES = ["ADC","IVIM_ADC", "IVIM_TRACEW_1","IVIM_TRACEW_2","IVIM_TRACEW_3","IVIM_TRACEW_4","IVIM_TRACEW_5","IVIM_TRACEW_B5", "B1000","B1000_NATIVE","Bxxxx" ]
    NIFTI_FILES = ["FLAIR"]
    BOLD_FILES = ["CVRmap"]
    SELECTED_TO_ANALYSIS = ["rBF", "rBV", "TMAX" ,"B1000","ADC","IVIM_TRACEW_5","IVIM_ADC","CVRmap","FLAIR"]
    
    # mri convert all the files from "original" folder and save to "resliced" folder
    os.chdir(DATA_DIR)
    mc = MRIConvert()
    path = "./original" 
    try:
        os.chdir(path)
        print("Current working directory: {0}".format(os.getcwd()))
    except FileNotFoundError:
        print("Directory: {0} does not exist".format(path))
    except NotADirectoryError:
        print("{0} is not a directory".format(path))
    except PermissionError:
        print("You do not have permissions to change to {0}".format(path))

    dir_resliced_name = "resliced"
    path_dir_resliced = os.path.join(os.getcwd(),"../"+ dir_resliced_name)
    if not os.path.isdir(path_dir_resliced):
        os.mkdir(path_dir_resliced)

    for i_img in PERFUSION_FILES + MASK_FILES + DWI_FILES + NIFTI_FILES + BOLD_FILES: 
        path_out_file = os.path.join(os.getcwd(), "../resliced/r_" + i_img + ".nii.gz")
        if not os.path.isfile(path_out_file):
            try:
                mc.inputs.in_file = os.path.join(os.getcwd(), i_img + '.nii.gz')
                mc.inputs.out_file = path_out_file
                mc.inputs.out_type = 'niigz'
                mc.inputs.reslice_like = PATH_T1_BRAINMASK
                res_r = mc.run()

            except:
                # print("Failure at processing file %s", path_out_file)
                pass
        else:
            print("File %s already exist " % path_out_file)
            
    
    # coregister the not perfusion nifti files NIFTI_FILES = ["ADC","IVIM_ADC, IVIM_TRACEW_B5"] 
    os.chdir(DATA_DIR)
    path = os.path.join(DATA_DIR, dir_resliced_name)
    try:
        os.chdir(path)
        print("Current working directory: {0}".format(os.getcwd()))
    except FileNotFoundError:
        print("Directory: {0} does not exist".format(path))
    except NotADirectoryError:
        print("{0} is not a directory".format(path))
    except PermissionError:
        print("You do not have permissions to change to {0}".format(path))

    # create coregt1 folder to store the t1 space coregistered nifti files
    dir_coreg_name = "coregt1"
    path_dir_coregt= os.path.join(os.getcwd(),"../"+dir_coreg_name)
    if not os.path.isdir(path_dir_coregt):
        os.mkdir(path_dir_coregt)

    # coregistrate B1000 and store transformation matrix
    tag_base_file = "B1000"
    # tag_base_file = "IVIM_TRACEW_5"

    PATH_BASE_DWI = os.path.abspath("./r_" +tag_base_file +".nii.gz")

    if os.path.isfile(PATH_BASE_DWI):
        path_matrix_file = os.path.join( os.getcwd(), "../" + dir_coreg_name  + "/" + dir_coreg_name +"_" + tag_base_file +"_2_t1matrix.mat")

        print("Baseline DWI file is {}".format(PATH_BASE_DWI))
        path_out_file = os.path.join(os.getcwd(), "../" + dir_coreg_name  + "/" + dir_coreg_name +"_" + tag_base_file  + ".nii.gz")
        PATH_CBF_2_BET_T1= run_flirt(path_infile = PATH_BASE_DWI,
                                     out_file=path_out_file,
                                     path_reference = PATH_T1_BRAINMASK,
                                     out_matrix_file= path_matrix_file,
                                     dof = 6)

        # iterate through perfusion maps
        for i_img in DWI_FILES : 
            path_out_file = os.path.join(os.getcwd(), "../" + dir_coreg_name  + "/" + dir_coreg_name +"_" + i_img + ".nii.gz")
            if not os.path.isfile(path_out_file):
                try:
                    path_infile = os.path.join(os.getcwd(),"r_" + i_img + '.nii.gz')
                    print("Processing {}".format(i_img))
                    applyxfm = fsl.preprocess.ApplyXFM()
                    applyxfm.inputs.in_file = path_infile
                    applyxfm.inputs.in_matrix_file = path_matrix_file
                    applyxfm.inputs.out_file = path_out_file
                    applyxfm.inputs.reference = PATH_T1_BRAINMASK
                    applyxfm.inputs.apply_xfm = True
                    result = applyxfm.run()            
                except:
                    # print("Failure at processing file %s" % path_out_file)
                    pass
            else:
                print("File %s already exist " % path_out_file)
    else:
        print("B1000 baseline file does not exist to calculate the transformation matrix for perfusion files. Choose a new baseline file or place the B1000 file in the original folder")


    # coregistrate rBV and store transformation matrix
    tag_base_file = "rBV"
    PATH_BASE_PERF = os.path.abspath("./r_" +tag_base_file +".nii.gz")

    if os.path.isfile(PATH_BASE_PERF):

        path_matrix_perf = os.path.join( os.getcwd(), "../" + dir_coreg_name  + "/" + dir_coreg_name +"_" + tag_base_file +"_2_t1matrix.mat")

        print("Baseline perfusion file is {}".format(PATH_BASE_PERF))
        path_out_file = os.path.join(os.getcwd(), "../" + dir_coreg_name  + "/" + dir_coreg_name +"_" + tag_base_file  + ".nii.gz")
        PATH_CBF_2_BET_T1= run_flirt(path_infile = PATH_BASE_PERF,
                                     out_file=path_out_file,
                                     path_reference = PATH_T1_BRAINMASK,
                                     out_matrix_file= path_matrix_perf,
                                     dof = 6)

        # iterate through perfusion maps
        for i_img in PERFUSION_FILES + MASK_FILES : 
            path_out_file = os.path.join(os.getcwd(), "../" + dir_coreg_name  + "/" + dir_coreg_name +"_" + i_img + ".nii.gz")
            if not os.path.isfile(path_out_file):
                try:
                    path_infile = os.path.join(os.getcwd(),"r_" + i_img + '.nii.gz')
                    print("Processing {}".format(i_img))
                    applyxfm = fsl.preprocess.ApplyXFM()
                    applyxfm.inputs.in_file = path_infile
                    applyxfm.inputs.in_matrix_file = path_matrix_perf
                    applyxfm.inputs.out_file = path_out_file
                    applyxfm.inputs.reference = PATH_T1_BRAINMASK
                    applyxfm.inputs.apply_xfm = True
                    result = applyxfm.run()            
                except:
                    # print("Failure at processing file %s" % i_img)
                    pass
            else:
                print("File %s already exist " % path_out_file)
    else:
        print("rBV baseline file does not exist to calculate the transformation matrix for perfusion files. Choose a new baseline file or place the B1000 file in the original folder")

    # coregistrate bold, and other nifti files in different spaces (T2 , FLAIR etc)

    for i_img in NIFTI_FILES:
        try:
            path_infile = os.path.join(os.getcwd(),"r_" + i_img + '.nii.gz')
            # path_matrix_file = os.path.join( os.getcwd(), "../" + dir_coreg_name  + "/" + dir_coreg_name +"_" + i_img +"_2_t1matrix.mat")
            path_matrix_file = os.path.join( os.getcwd(), "../" + dir_coreg_name  + "/" + dir_coreg_name +"_" + i_img +"_2_t1matrix.mat")

            "inv_coreg_T1_BOLD"
            print("Coregistration of {} to T1".format(i_img))
            path_out_file = os.path.join(os.getcwd(), "../" + dir_coreg_name  + "/" + dir_coreg_name +"_" + i_img  + ".nii.gz")
            PATH_CBF_2_BET_T1= run_flirt(path_infile = path_infile,
                                         out_file=path_out_file,
                                         path_reference = PATH_T1_BRAINMASK,
                                         out_matrix_file= path_matrix_file,
                                         dof = 6)
        except:
            print("File %s could not be processed" % i_img)
            
    # registrate bold file with inverse transformation matrix
    for i_img in BOLD_FILES:
        #try:
        path_infile = os.path.join(os.getcwd(),"r_" + i_img + '.nii.gz')
        path_matrix_file_bold = os.path.join( os.getcwd(), "inv_coreg_T1_BOLD.txt")

        print("Coregistration of {} to T1".format(i_img))
        path_out_file = os.path.join(os.getcwd(), "../" + dir_coreg_name  + "/" + dir_coreg_name +"_" + i_img  + ".nii.gz")
        try:
            print("Processing {}".format(i_img))
            applyxfm = fsl.preprocess.ApplyXFM()
            applyxfm.inputs.in_file = path_infile
            applyxfm.inputs.in_matrix_file = path_matrix_file_bold
            applyxfm.inputs.out_file = path_out_file
            applyxfm.inputs.reference = PATH_T1_BRAINMASK
            applyxfm.inputs.apply_xfm = True
            result = applyxfm.run()   


        except:
            print("File %s could not be processed" % i_img)


    os.chdir("..")
    
       
    # coregister all the NIFTI_FILES to MNI wiht transformation matrix between t1 and MNI152
    os.chdir(DATA_DIR)
    path = "./" + dir_coreg_name
    try:
        os.chdir(path)
        print("Current working directory: {0}".format(os.getcwd()))
    except FileNotFoundError:
        print("Directory: {0} does not exist".format(path))
    except NotADirectoryError:
        print("{0} is not a directory".format(path))
    except PermissionError:
        print("You do not have permissions to change to {0}".format(path))

    dir_mni_name = "mni"
    path_dir_mni= os.path.join(os.getcwd(),"../"+dir_mni_name)
    if not os.path.isdir(path_dir_mni):
        os.mkdir(path_dir_mni)


    # coregistrate T1 to MNI and store transofrmation matrix
    tag_base_file = "t1_masked_with_aseg"
    path_matrix_mni_file = os.path.join( os.getcwd(), "../" + dir_mni_name  + "/" + tag_base_file +"_2_mnimatrix.mat")

    print("Path of T1 file to be coregistered to MNI is {}".format(PATH_T1_BRAINMASK))
    path_out_file = os.path.join(os.getcwd(), "../" + dir_mni_name  + "/" + dir_mni_name +"_" + tag_base_file  + ".nii.gz")
    PATH_CBF_2_BET_T1= run_flirt(path_infile = PATH_T1_BRAINMASK,
                                 out_file=path_out_file,
                                 path_reference = PATH_MNI_BRAINMASK,
                                 out_matrix_file= path_matrix_mni_file,
                                 dof = 12)

    # iterate through all files and apply the transformation matrix calculated above between the patient's t1  and mni space
    for i_img in PERFUSION_FILES+ MASK_FILES + DWI_FILES + NIFTI_FILES + BOLD_FILES : 
        path_out_file = os.path.join(os.getcwd(), "../" + dir_mni_name  + "/" + dir_mni_name +"_" + i_img + ".nii.gz")
        if not os.path.isfile(path_out_file):
            try:
                path_infile = os.path.join(os.getcwd(),"coregt1_" + i_img + '.nii.gz')
                print("Processing {}".format(i_img))
                applyxfm = fsl.preprocess.ApplyXFM()
                applyxfm.inputs.in_file = path_infile
                applyxfm.inputs.in_matrix_file = path_matrix_mni_file
                applyxfm.inputs.out_file = path_out_file
                #applyxfm.inputs.reference = PATH_T1_BRAINMASK
                applyxfm.inputs.reference = PATH_MNI_BRAINMASK
                applyxfm.inputs.apply_xfm = True
                result = applyxfm.run()            
            except:
                # print("Failure at processing file %s" % path_out_file)
                pass
        else:
            print("File %s already exist " % path_out_file)

        ### calculate the values
    #  read the coregistered CBF file
    PATH_TMAX = os.path.join(os.path.dirname(PATH_MASK_PENUMBRA),'coregt1_TMAX.nii.gz' )
    #if os.path.exists(PATH_TMAX):
    print("Tmax exist")
    vol_tmax = nb.load(PATH_TMAX)
    np_vol_tmax = vol_tmax.get_fdata()

    PATH_RBV = os.path.join(os.path.dirname(PATH_MASK_PENUMBRA),'coregt1_rBV.nii.gz' )
    if os.path.exists(PATH_RBV):
        vol_rbv = nb.load(PATH_RBV)
        np_vol_rbv = vol_rbv.get_fdata()
    
    PATH_RBF = os.path.join(os.path.dirname(PATH_MASK_PENUMBRA),'coregt1_rBF.nii.gz' )
    if os.path.exists(PATH_RBF):
        vol_rbf = nb.load(PATH_RBF)
        np_vol_rbf = vol_rbf.get_fdata()

    PATH_MASK_PENUMBRA = os.path.join(DATA_DIR, "coregt1", "coregt1_mask_penumbra_bl.nii.gz")
    # load the penumbra masks
    if os.path.exists(PATH_MASK_PENUMBRA):
        vol_mask = nb.load(PATH_MASK_PENUMBRA)
        vol_mask.shape
        np_vol_mask = vol_mask.get_fdata()
        np_vol_mask[np_vol_tmax < 6] = 0
        np_vol_mask[np_vol_rbv < 0] = 0
        np_vol_mask[np_vol_rbf < 0]  = 0
        
    PATH_MASK_PENUMBRA46 = os.path.join(DATA_DIR, "coregt1", "coregt1_mask_penumbra_bl_4-6.nii.gz")
    # load the penumbra masks
    if os.path.exists(PATH_MASK_PENUMBRA46):
        vol_mask_penumbra46 = nb.load(PATH_MASK_PENUMBRA46)
        vol_mask.shape
        np_vol_mask_p46 = vol_mask_penumbra46.get_fdata()
        np_vol_mask_p46[np_vol_tmax < 6] = 0
        np_vol_mask_p46[np_vol_rbv < 0] = 0
        np_vol_mask_p46[np_vol_rbf < 0]  = 0    

    PATH_MASK_CORE = os.path.join(DATA_DIR, "coregt1", "coregt1_mask_core_v1.nii.gz")
    if os.path.exists(PATH_MASK_CORE):
        vol_mask_core = nb.load(PATH_MASK_CORE)
        vol_mask.shape
        np_vol_mask_core = vol_mask_core.get_fdata()
        np_vol_mask_core[np_vol_tmax < 6] = 0
        np_vol_mask_core[np_vol_rbv < 0] = 0
        np_vol_mask_core[np_vol_rbf < 0]  = 0

    columns = ['patient','visit']
    values = [get_name_of_patient(DATA_DIR), get_name_of_visit(DATA_DIR) ]
    values_p46 = [get_name_of_patient(DATA_DIR), get_name_of_visit(DATA_DIR) ]
    values_core = [get_name_of_patient(DATA_DIR), get_name_of_visit(DATA_DIR) ]

    # index = [get_name_of_folder(DATA_DIR) + str(os.path.basename(PATH_CBF_PERF)[4:-7])] 
    index = [get_name_of_folder(DATA_DIR)] 

    for i_seq in SELECTED_TO_ANALYSIS:    
        list_calculated = [i_seq+'_mean',
                        i_seq+'_std',
                        i_seq+'_min',
                        i_seq+'_max',
                        i_seq+'_median',
                        i_seq+'_q1',
                        i_seq+'_q3']
        columns = columns + list_calculated
        
        path_infile = os.path.join(DATA_DIR,"coregt1", "coregt1_" + i_seq + '.nii.gz')

        if os.path.isfile(path_infile):
            vol = nb.load(path_infile)
            np_vol = vol.get_fdata()
            
            if os.path.exists(PATH_MASK_PENUMBRA):
                # apply left hemisphere mask on flirt_cbf_to_bett1
                np_vol_masked = np.zeros(np_vol_mask.shape)
                np.putmask(np_vol_masked, np_vol_mask, np_vol)
                roi = np_vol_masked[np_vol_mask>0.5]       
                seq_values = [np.mean(roi), np.std(roi), np.min(roi), np.max(roi),np.median(roi),np.percentile(roi,25),np.percentile(roi,75) ]
                print(seq_values)
                
            if os.path.exists(PATH_MASK_PENUMBRA46):
                # apply left hemisphere mask on flirt_cbf_to_bett1
                np_vol_masked_p46  = np.zeros(np_vol_mask_p46 .shape)
                np.putmask(np_vol_masked_p46 , np_vol_mask_p46 , np_vol)
                roi_p46  = np_vol_masked_p46 [np_vol_mask_p46 >0.5]
                seq_values_p46 = [np.mean(roi_p46), np.std(roi_p46), np.min(roi_p46), np.max(roi_p46),np.median(roi_p46),np.percentile(roi_p46,25),np.percentile(roi_p46,75) ]
                print(seq_values_p46)
                
            if os.path.exists(PATH_MASK_CORE):
                # apply left hemisphere mask on flirt_cbf_to_bett1
                np_vol_masked_core = np.zeros(np_vol_mask_core.shape)
                np.putmask(np_vol_masked_core , np_vol_mask_core , np_vol)
                roi_core  = np_vol_masked_core [np_vol_mask_core >0.5]      
                seq_values_core = [np.mean(roi_core ), np.std(roi_core ), np.min(roi_core ), np.max(roi_core ),np.median(roi_core ),np.percentile(roi_core ,25),np.percentile(roi_core ,75) ]
                print(seq_values_core )
                
        else:
            # seq_values = np.full([1, len(list_calculated)],1)
            # seq_values = np.full([1, len(list_calculated)], 0)
            if os.path.exists(PATH_MASK_PENUMBRA):
                seq_values = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan ]
            if os.path.exists(PATH_MASK_PENUMBRA46):
                seq_values_p46 = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan ]
            if os.path.exists(PATH_MASK_CORE):
                seq_values_core = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan ]

        # values to the sequence
        if os.path.exists(PATH_MASK_PENUMBRA):
            values = values + seq_values
            df_penumbra= pd.DataFrame(index=index, columns=columns)
            df_penumbra.loc[index[0]] = values
            PATH_CSV_LOCAL = os.path.join(DATA_DIR, get_name_of_folder(DATA_DIR)+'_penumbra.csv')
            df_penumbra.to_csv(PATH_CSV_LOCAL, index=True, mode='w+', sep = ',')
            #df_penumbra.to_csv(PATH_GLOBAL_CSV_POSTOP, mode='a', header=False)
            # df_penumbra.to_csv(PATH_GLOBAL_CSV_CT_MRI, mode='a', header=False)

        if os.path.exists(PATH_MASK_PENUMBRA46):
            values_p46 = values_p46 + seq_values_p46
            df_p46= pd.DataFrame(index=index, columns=columns)
            df_p46.loc[index[0]] = values_p46
            PATH_CSV_LOCAL_p46 = os.path.join(DATA_DIR, get_name_of_folder(DATA_DIR)+'._p46.csv')
            df_p46.to_csv(PATH_CSV_LOCAL_p46, index=True, mode='w+', sep = ',')

        if os.path.exists(PATH_MASK_CORE):
            values_core = values_core + seq_values_core
            df_core= pd.DataFrame(index=index, columns=columns)
            df_core.loc[index[0]] = values_core
            PATH_CSV_LOCAL_core = os.path.join(DATA_DIR, get_name_of_folder(DATA_DIR)+'_core.csv')
            df_core.to_csv(PATH_CSV_LOCAL_core, index=True, mode='w+', sep = ',')
            

            
if __name__ == '__main__':

    # change before run: Select the original folder where T1_masked_with_aseg.nii.gz is present and the images are in "original" folder
    PROJECT_DIR = os.getcwd()
    PATH_GLOBAL_CSV_CT_MRI = PROJECT_DIR
    #DATA_DIR = "/media/nraresearch/ben_usz/crpp_reperfusion_failure"
    #DATA_DIR = os.getcwd()
    "/media/nraresearch/ben_usz/crpp_reperfusion_failure/CRPP1_Test/crpp1_baseline_10102019"
    """
    PATH_GLOBAL_CSV_CT = os.path.join(DATA_DIR, '..','..','values_all_baseline.csv')
    PATH_GLOBAL_CSV_CT_MRI = os.path.join(DATA_DIR, '..','..','values_all.csv')
    """


    for i_patient in [ name for name in os.listdir(PROJECT_DIR) if os.path.isdir(os.path.join(PROJECT_DIR, name)) and name.startswith('CRPP') ]:
        print("processing patient: %s" % i_patient)
        patient_dir_path = os.path.join(PROJECT_DIR, i_patient)

        list_studies = [ name for name in os.listdir(patient_dir_path) if os.path.isdir(os.path.join(patient_dir_path, name)) and name.startswith('crpp') ]
        for i_dir in list_studies:
            print("processing study: %s" % i_dir)
            i_dirpath = os.path.join(PROJECT_DIR, i_patient, i_dir)
            match = nativ_in_folder(i_dirpath)
            if match==1:
                # try:
                run_ct_coreg(i_dirpath)
                #except:
                #print("Crush during processing %s" % i_patient)
                
            else:
                #pass
                #try:
                run_mr_coreg(i_dirpath)
                #except:
                #print("Crush during processing")

