"""
class mapping dr.choi
"""

def config_choi_setting():
    return {
    'Atelectasis':{
        'Atelectasis',
        'Atelectas',
        },

    'LungOpacity':{
        'Pneumonia',
        'Consolidation',
        'Infiltration',
        'Pulmonary Edema',
        },

    'Effusion':{
        'Pleural Effusion',
        },

    'NoduleMass':{
        'SPN (Solitary Pulmonary Nodule)',
        'Mass',
        'MPN (Multiple Pulmonary Nodule)',
        'Ill-defined Nodules',
        'Cavitary Leison',
        'Pleural Mass',
        'Nipple-like Pulmonary Nodule',  
        'Bone Island-like Pulmonary Nodule',
        'Granuloma or Benigh Nodule',
        'Miliary Nodules',
        'Patchy and ill defined nodules',
        'Cavitary Nodule',
        },

    'Hilar':{
        'Hilar Enlargement',
        'Hilar Density Increase',
        },

    'Fibrosis':{
        'Fibrosis',
        'Fibrocalcified',
        'Fibronodular',
        },

    'Cardiomegaly':{
        'Cardiomegaly',
        },
        
    'Pneumothorax':{
        'Pneumothorax',
        },
   
    'Pneumonia':{
        'Pneumonia',
        },
        
    'Tuberculosis':{
        'Tuberculosis',
        },
        
    'Normal':{
        # 'Normal, No Remarkable Finding',
        'Normal'
        },
             
    'Others':{
        'The Others',
        'DILD',
        'Collapse/Consolidation',
        'Pleural Thickening',
        'Destructive Pattern',
        'Mediastinal Mass',
        'Congestive Heart Failure',
        'Bulla',
        'Bronchiectasis/Atelectasis',
        'Pleural Calcification',
        'Rib Fracture',
        'The Others Pneumoconiosis and/or Tbc',
        'Tree-in-bud Sign',
        'Scoliosis',
        'The Others Subcutaneous emphysema',
        'Bronchiectasis',
        },
        
    'Discard':{

        'Infiltration Suspicious',
        'SPN Suspicious',        
        'No Decision Follow Up Needed',
        'Aortic Aneurysm',
        'Pneumothorax Suspicious',
        'Cardiomegally Suspicious',
        'Hiatal Hernia',
        'Aortic Enlargement',
        'Pneumonia Viral',
        'Emphysema Suspicious',
        'Pneumonia Suspicious',
        'Pulmonary Hemorrhage',
        'The Others Pneumoconiosis R/O MPN',
        'Others apical pleural thickening and fibrotic change',
        'COPD',
        'Emphysema',        
        'Lymphadenopathy',    
        'Spine Fracture',
        },
        
    'TB':{
        "ActiveTB",
        "UndeterminedTB",
        # "InActiveTB"
        }
    }



"""
class mapping 3dr
"""



"""
finding
"""

def finding_config_setting():
    return {
        1: 'Atelectasis',
        2: 'Lung Opacity',
        3: 'Pleural Effusion',
        4: 'Nodule/Mass without Cavitation',
        5: 'Hilar Abnormality',
        6: 'Fibrosis of ILD',
        7: 'Cardiomegaly',
        8: 'Pneumothorax',
        9: 'Bronchiectasis',
        10: 'Subcutaneous Emphysema',
        11: 'Emphysema',
        12: 'Bulla',
        13: 'Destroyed Lung',
        14: 'Pleural Thickening',
        15: 'Pleural Calcification',
        16: 'Pleural Mass',
        17: 'Mediastinal Abnormality',
        18: 'Calcification',
        19: 'Nipple Shadow',
        20: 'Rib Fracture Remote',
        21: 'Bone Island',
        22: 'Spinal Compression Fracture',
        23: 'Scoliosis',
        24: 'Medical Device',
        25: 'Pneumoperitoneum',
        26: 'Miliary Nodule',
        27: 'Cavitary Nodule',
        28: 'Fibrosis of Infection Sequelae',
        29: 'Rib Fracture Recent'
    }

def config_8class():
    """
    -1 indicates "not our target"
    0 : atelectasis
    1 : lung opacity
    2 : effusion
    3 : nodule mass
    4 : hilar
    5 : fibrosis
    6 : cardiomegaly
    7 : pneumothorax
    """
    return {
        1: 0,
        2: 1,
        3: 2,
        4: 3,
        5: 4,
        6: -1,
        7: 6,
        8: 7,
        9: -1,
        10: -1,
        11: -1,
        12: -1,
        13: -1,
        14: -1,
        15: -1,
        16: -1,
        17: -1,
        18: -1,
        19: -1,
        20: -1,
        21: -1,
        22: -1,
        23: -1,
        24: -1,
        25: -1,
        26: 3,
        27: 3,
        28: 5,
        29: -1
    }


"""
disease
"""

def disease_config_setting():
    return {
        1 : 'Completely Normal',
        2 : 'Normal with Inactive Lesions',
        3 : 'Pneumonia',
        4 : 'Active Tuberculosis',
        5 : 'Others',
        6 : 'Poor Image Quality',
        7 : 'Lung Cancer'
    }




# def config_disease():
#     """
#     -1 indicates "not our target"
#     0 : tuberculosis
#     1 : pneumonia
#     2 : lung cancer
#     3 : normal
#     """
#     return {
#         1: 3,
#         2: 3,
#         3: 1,
#         4: 0,
#         5: -1,
#         6: -1,
#         7 : 2
#     }


### now we are not separating disease : only TB and Normal added to finding
def config_disease():
    """
    -1 indicates "not our target"
    0 : tuberculosis
    1 : pneumonia
    2 : lung cancer
    3 : normal
    """
    return {
        1: 9,
        2: -1,
        3: -1,
        4: 8,
        5: -1,
        6: -1,
        7 : -1
    }
    
    