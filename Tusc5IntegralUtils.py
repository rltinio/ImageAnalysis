import datajoint as dj
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import seaborn as sns
import statistics as st



# DataJoint Startup
dj.conn()
sln_results = dj.VirtualModule('sln_results.py', 'sln_results')
sln_symphony = dj.VirtualModule('sln_symphony.py', 'sln_symphony')
sln_cell = dj.VirtualModule('sln_cell.py', 'sln_cell')
sln_animal = dj.VirtualModule('sln_animal.py', 'sln_animal')

def AssignCurrentType():
    
    current = sln_cell.Cell * sln_cell.CellEvent * sln_cell.AssignType & \
    sln_cell.Cell.aggr(sln_cell.AssignType * sln_cell.CellEvent, entry_time = "max(entry_time)") * dj.U('entry_time')

    current = current.proj('cell_type','cell_class','file_name','source_id')

    return current

def AssignCurrentType_Count(): # Adds 'animal_id' which causes problems for analysis queries
    
    current = sln_cell.Cell * sln_cell.CellEvent * sln_cell.AssignType & \
    sln_cell.Cell.aggr(sln_cell.AssignType * sln_cell.CellEvent, entry_time = "max(entry_time)") * dj.U('entry_time')

    current = current.proj('cell_type','cell_class','file_name','source_id','animal_id')

    return current

# Makes Lists nested in cells into strings, allowing it to be exported for use in MatLab

def Lists_To_Strings(dataframe, list_of_columns):
    
    for column in list_of_columns:

        dataframe[column] = dataframe[column].apply(lambda x: ', '.join(map(str, x)))

    return dataframe

# Instantly make analysis table including user-defined params

def JoinData(AnalysisName):
    analysis_data = AnalysisName * sln_cell.Cell * AssignCurrentType()
    
    userdefined_params = sln_symphony.UserParamAnimalAnimalType * sln_symphony.UserParamCellLabeled\
    * sln_symphony.UserParamCellBadRecording * sln_symphony.UserParamDatasetDrugPresent\
    * sln_symphony.UserParamCellBadSpikeCounting * sln_symphony.UserParamCellRt5Ignore\
    * sln_symphony.UserParamDatasetRt5Keep * sln_symphony.UserParamDatasetBadParamSet
    
    datasets = analysis_data * userdefined_params & 'cell_class = "RGC"'
    datasets = datasets.fetch(format = 'frame').reset_index().set_index('cell_unid')

    return datasets

# Performs np.ravel() on desired list of columns
def unravel(dataframe, list_of_columns):
    
    # Unravels each column and puts in back into the df
    for col in list_of_columns:
        dataframe[col] = dataframe[col].apply(lambda x: np.ravel(x))
    
    return dataframe


def CleanData(dataframe):

    dataframe_edit = dataframe.copy()
    badparam_indicies = []

    for parameter in ['rt_5_keep','rt_5_ignore', 'bad_recording', 'drug_present', 'bad_spike_counting', 'bad_param_set']:
        
        if parameter == 'rt_5_keep':
            
            indices = dataframe_edit.loc[dataframe_edit[parameter] == 'F']
            badparam_indicies.append(indices.index.tolist())
        
        else:
            
            indicies = dataframe_edit.loc[dataframe_edit[parameter] == 'T']
            badparam_indicies.append(indicies.index.tolist())

    badparam_indicies = [item for sublist in badparam_indicies for item in sublist] # Flattening List

    # Keeping rows with indicies that are not in the bad param indicies list
    dataframe_edit = dataframe_edit.loc[~dataframe_edit.index.isin(badparam_indicies)]


    # Removing Unknown and Unclassified Data
    cell_types_edit = dataframe_edit.cell_type.unique()
    cell_types_final = [type for type in cell_types_edit if type not in ['unknown', 'unclassified']]

    dataframe_edit = dataframe_edit.loc[dataframe_edit.cell_type.isin(cell_types_final)]


    # Removing weird names
    dataframe_edit = dataframe_edit.loc[dataframe_edit.dataset_name.str.contains('VC|hold|TTX|(?i)synBlock|-|stim|axon|Loss') == False]

    dataframe_final = dataframe_edit.copy().reset_index() # Added reset_index() 9/10/23
    return dataframe_final



## Needs dataframe, list of columns
## Obviously the rows of the columns need to be a list of lists

#######################################################################################################################################

def Labeled_Finder(dataframe):

    # Identifying rows with the desired animal types | Excluding 'Rosa', None, ''
    desired_animaltypes = ['WT','Tusc5 homo','Tusc5 het']
    dataframe_edit = dataframe.loc[dataframe.animal_type.isin(desired_animaltypes),:] # Retrieves only Homo/Het/WT animal

    # Identifying rows with labeled subtypes
    labeled_types = dataframe_edit.loc[dataframe_edit.labeled == 'T',:].cell_type.unique() # Retrieves array of labeled subtypes
    labeled_types = np.delete(labeled_types, np.where(labeled_types == 'unknown')) # Removes 'unknown' from the list
    T5_Cells = dataframe_edit.loc[dataframe_edit.cell_type.isin(labeled_types),:] # Retrieves table with only labeled subtypes

    return T5_Cells

## Needs dataframe
## Extracts Tusc5 Cells from dataframe of analysis data


def NonLabeled_Finder(dataframe):

    # Identifying rows with the desired animal types | Excluding 'Rosa', None, '' 
    desired_animaltypes = ['WT','Tusc5 homo','Tusc5 het']
    dataframe_edit = dataframe.loc[dataframe.animal_type.isin(desired_animaltypes),:] # Retrieves only Homo/Het/WT animal

    # Identifying rows with nonlabeled subtypes
    labeled_types = dataframe.loc[dataframe.labeled == 'T',:].cell_type.unique()
    #nonRGC_types = NonLabeled_Cells.loc[NonLabeled_Cells.cell_class != 'RGC',:].cell_type.unique() # Done earlier
    #undesired_subtypes = np.concatenate([labeled_types,nonRGC_types]) # List of nonlabeled RGCs and amacrine cells

    all_types = dataframe.cell_type.unique() # Retrieves all subtypes (within the NonLabeled_Cells dataframe)

    # Identifying rows that contain types in the nonlableed list
    nonlabeled_types = [i for i in all_types if i not in labeled_types] # Subtracts undesired labeledcell_types from all subtypes
    nonlabeled_types = np.delete(nonlabeled_types, np.where(nonlabeled_types == ['unknown','unclassified'])) # Deletes unknown/unclassified from non-labeled list
    
    NonLabeled_Cells = dataframe_edit.loc[dataframe_edit.cell_type.isin(nonlabeled_types),:] # Retrieves table with nonlabeledcell types

    return NonLabeled_Cells

#######################################################################################################################################

def Label_Separator(dataframe):

    # Identifying rows with the desired animal types | Excluding 'Rosa', None, '' 
    desired_animaltypes = ['WT','Tusc5 homo','Tusc5 het']
    dataframe_edit = dataframe.loc[dataframe.animal_type.isin(desired_animaltypes),:] # Retrieves only Homo/Het/WT animal

    # Identifying rows with nonlabeled subtypes
    all_types = dataframe_edit.cell_type.unique() # Retrieves all subtypes (within the NonLabeled_Cells dataframe)
    all_types = [type for type in all_types if type not in ['unknown', 'unclassified']] # Removing unknown and unclassified from the list

    # Identifying rows that contain types in the labeled list
    labeled_types = dataframe_edit.loc[dataframe_edit.labeled == 'T',:].cell_type.unique()
    # Identifying rows that contain types in the nonlableed list
    nonlabeled_types = [i for i in all_types if i not in labeled_types] # Subtracts labeledcell_types from all subtypes

    NonLabeled_Cells = dataframe_edit.loc[dataframe_edit.cell_type.isin(nonlabeled_types),:] # Retrieves table with nonlabeledcell types
    Labeled_Cells = dataframe_edit.loc[dataframe_edit.cell_type.isin(labeled_types),:] # Retrieves table with only labeled subtypes

    return Labeled_Cells, NonLabeled_Cells


# Calculating Current Injections at Block
def CalculateBlock(dataframe):
    
    blocked_injected_currents = []
    blocked_firing_rates = []

    for idx in dataframe.index:
        firing_rates, injected_currents = dataframe.loc[idx, ['fr_per_current_mean', 'inj_current']]

        # Plus 1 because we want the index of the fr/injection one past the max
        potential_block_index = firing_rates.argmax() + 1

        # If the cell doesn't block, append NA
        if potential_block_index > len(firing_rates) - 1:
            blocked_injected_currents.append(np.nan)
            blocked_firing_rates.append(np.nan)

        # If it doesn't block, then...
        else:
            injected_current_at_blocked = injected_currents[potential_block_index]
            blocked_injected_currents.append(injected_current_at_blocked)
            
            firing_rate_at_blocked = firing_rates[potential_block_index]
            blocked_firing_rates.append(firing_rate_at_blocked)

    dataframe['Blocked_Injection'] = blocked_injected_currents
    dataframe['Blocked_Firing_Rate'] = blocked_firing_rates

    return dataframe