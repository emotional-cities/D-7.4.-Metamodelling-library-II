import os
import numpy as np
import pandas as pd
import subprocess
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def fn_simulation_config(params):
    """
    Update the simulation configuration using given parameters.

    Args:
        params (pd.DataFrame or list of dict): Input parameters to update the simulation configuration.

    Returns:
        list: List of statuses indicating the success or failure of each parameter update.
    """

    print("fn_simulation_config")

    # Initialize a list to store the status of each parameter update
    status = []

    # Convert the input params DataFrame into a list of dictionaries
    params = params.to_dict("records")

    # Iterate over each parameter and update the model configuration using a helper function
    for p in params:
        # The function fn_param_update is presumed to apply the parameter to the model 
        # and return a status indicating success or failure
        status.append(fn_param_update(p['model'], p['param_name'], p['declaration'], p['value']))

    # Return the list of status codes
    return status


def fn_param_update(model, param_name, declaration, value):
    """
    Update the specified parameter's value in the given lua model file.

    Args:
        model (str): The name of the lua model file (without the .lua extension).
        param_name (str): The name of the parameter to be updated.
        declaration (str): The declaration associated with the parameter (e.g., local).
        param_name_left_hand (str): Left-hand side representation of the parameter (likely deprecated or unused here).
        value (any): The new value to be assigned to the parameter.

    Note:
        This function assumes that the parameter declaration in the lua file is in the format:
        <declaration> <param_name> = <some_value>
        And it will replace <some_value> with the provided value.
    """
    
    # Define the directory path where the lua files are located
    lua_folder_path = "/home/ltola/simmobility/CPE/scripts/lua/mid/VC_Lorena"
    
    # Construct the full path to the desired lua file based on the given model name
    lua_file_path = os.path.join(lua_folder_path, f'{model}.lua')

    # Open the lua file and read its contents
    with open(lua_file_path, 'r') as file:
        filedata = file.read()

    # Search for the line that contains the target parameter name
    param_with_value = next((line for line in filedata.split('\n') if param_name in line), None)
    
    # If the parameter is found in the file, replace its value with the new one
    if param_with_value:
        filedata = filedata.replace(param_with_value, f'{declaration} {param_name} = {value}')

    # Write the modified content back to the lua file
    with open(lua_file_path, 'w') as file:
        file.write(filedata)


def fn_simulation_call():
    """
    Execute a simulation process in a specific directory and check for errors.

    Args:
        setting_obj (dict): A dictionary containing various settings. This argument is currently unused in the function.

    Returns:
        int: A return code indicating success (0) or failure (non-zero).
        
    Note:
        The function currently assumes a fixed simulation path and command.
        In case of an error, the function will raise an exception with the simulation's error messages.
    """
    
    print("fn_simulation_call")

    # Define the path to the simulation and the command to run it
    simulation_path = "/home/ltola/simmobility/CPE/"
    command = "./SimMobility_Medium ./simulation_maas_base_case.xml ./simrun_MidTerm_maas_base_case.xml"

    # Execute the simulation command in the specified directory
    with subprocess.Popen(command, cwd=simulation_path, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        try:
            # Wait for the process to complete and capture any output (stdout and stderr)
            stdout, stderr = process.communicate()
            
            # Uncomment the below lines if you want to print the simulation output (useful for debugging)
            # print(f"Simulation Stdout:\n{stdout.decode('utf-8')}")
            # print(f"Simulation Stderr:\n{stderr.decode('utf-8')}")
            
            # If the simulation returns a non-zero exit code, it means an error has occurred
            if process.returncode != 0:
                raise RuntimeError("Simulation terminated with an error!")
        except Exception as e:
            # If there's any issue in the above process, capture the error and terminate the simulation process
            print(f"Original Error: {e}")
            process.kill()
            print("Simulation Error")
            # Raise the captured error to the calling function or user
            raise

    # Return 0 to indicate a successful function execution
    return 0

def fn_process_activities():
    """
    Process activity schedules and return calculated output statistics.

    The function reads an activity schedule CSV file and computes output statistics
    based on the activity data. It utilizes the `fn_output_od_mode_balance` function
    to compute the desired outcomes.

    Returns:
        dict: A dictionary containing calculated output statistics.
        
    Note:
        This function assumes a global variable `shared_env` that contains 
        the settings, including the path to the activity schedule CSV file.
    """
    
    print("fn_process_activities")
    ### I change this to account for the accesibility
    # Define the expected column names for logsum CSV file
    # Define the expected column names for the activity schedule CSV file
    columns = ["person_id", "tour_no", "tour_type", "stop_no", "stop_type", "stop_location", "stop_zone", "stop_mode", "primary_stop", "arrival_time", "departure_time", "prev_stop_location", "prev_stop_zone", "prev_stop_departure_time", "pid"]

    # Read the specified CSV file into a pandas DataFrame using the defined column names
    activity = pd.read_csv("/home/ltola/simmobility/CPE/activity_schedule", names=columns, usecols=range(len(columns)))
    activity['person_id'] = activity['person_id'].str[:-2]
    activity['person_id'] = activity['person_id'].astype(int)
    # Calculate the output statistics
    #logsums["sum"] = logsums['work'] + logsums['education'] + logsums['shop'] + logsums['other'] + logsums['dp_tour'] # + logsums['dp_stop']
    #count_other = logsums[['person_id', 'sum']]

    # Specify the file path
    file_path = 'Data/VC_population.csv'

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path,delimiter = ";")
    df = df[['id', 'gender_id']]
    merged_df = pd.merge(activity, df, left_on='person_id', right_on='id', how='left')

    # Calculate the number of shopping trip that women and men do
    filtered_df_2 = merged_df[(merged_df['gender_id'] == 2) & (merged_df['stop_type'] == 'Shop')]
    filtered_df_1 = merged_df[(merged_df['gender_id'] ==1) & (merged_df['stop_type'] == 'Shop')]


    count_filtered_1 = len(filtered_df_1)
    count_filtered_2 = len(filtered_df_2)
    count_gender_id_1 = len(df[df['gender_id'] == 1])
    count_gender_id_2 = len(df[df['gender_id'] == 2])


    # Calculate the number of shopping trips that men and women do on percentage of the total number of trips
    percentage_men = (count_filtered_1 / count_gender_id_1)
    percentage_women = (count_filtered_2 / count_gender_id_2)

    difference_shop = percentage_women - percentage_men

    print(difference_shop)

    # Calcute the total work time for each gender
    filtered_df_2_work = merged_df[(merged_df['gender_id'] == 2) & (merged_df['stop_type'] == 'Work')]
    work_time_2 = (filtered_df_2_work['departure_time'] - filtered_df_2_work['arrival_time'])*30
    work_time_2 = sum(work_time_2)

    filtered_df_1_work = merged_df[(merged_df['gender_id'] == 1) & (merged_df['stop_type'] == 'Work')]
    work_time_1 = (filtered_df_1_work['departure_time'] - filtered_df_1_work['arrival_time'])*30
    work_time_1 = sum(work_time_1)

    # Calculate the number of women and men
    count_gender_id_2 = len(df[df['gender_id'] == 2])
    count_gender_id_1 = len(df[df['gender_id'] == 1])

    # Calculate the average work time
    average_work_time_women = work_time_2/count_gender_id_2
    average_work_time_men = work_time_1/count_gender_id_1

    difference_work = average_work_time_women - average_work_time_men

    # Return the calculated output statistics
    return [difference_shop, difference_work]

def fn_simulation(params): #first function to call
    """
    Simulate a model or system using given parameters.

    Args:
        params (pd.DataFrame or dict): Input parameters for the simulation.

    Returns:
        dict: A dictionary containing configuration output, inadequacy value
    """

    print("fn_simulation")

    # Update the simulation configuration files with the given parameters
    param_config_output = fn_simulation_config(params)

    # Execute the actual simulation process, using a predefined global object "setting_obj"
    fn_simulation_call()

    print("The model has been run successfully")
    # Process the results of the simulation to get the activity outcomes
    outcome = fn_process_activities()

    # Prepare the inadequacy dictionary based on the outcome of activities
    #inadequacy = {"value": outcome}

    # Return a dictionary that includes configuration output, inadequacy values, and residuals
    return outcome

def rf_fit(X, y):

    # Define the hyperparameters to tune
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None,10,20]
    }

    # Initialize a Random Forest Regressor
    rf_clf = RandomForestRegressor(random_state=42)

    # Initialize GridSearchCV for cross validation
    grid_search = GridSearchCV(rf_clf, param_grid, cv=3)

    # Fit the model to the data
    grid_search.fit(X, y)

    # Get the results as a DataFrame
    results_df = pd.DataFrame(grid_search.cv_results_)

    # Print the mean squared errors (MSE) for different models
    print("Mean squared errors for different models:")
    print(results_df[['params', 'mean_test_score', 'std_test_score']])


    # Get the best model
    best_rf_model = grid_search.best_estimator_

    # Access the best parameters
    best_params = grid_search.best_params_
    print("Best parameters:", best_params)


    return best_rf_model

def make_predictions_rf(best_rf_model, X_test):

    # Make predictions with the best model
    predictions = best_rf_model.predict(X_test)

    return predictions


###################################################### 

###### Execution code
import pandas as pd
with open('Data/X10_4.pickle', 'rb') as file:
    X = pickle.load(file)
with open('Data/Xtest20_4.pickle', 'rb') as file:
    X_test = pickle.load(file)

outcome = []

# Run the simulation for each set of parameters
for i in range(X.shape[0]):
    model_data = ['nto', 'nts', 'dpb', 'dpb', 'dps', 'dpt', 'nte', 'nto', 'nts', 'ntw', 'uw']
    param_name_data = ['beta_female_other2_uncertanty', 'beta_female_shop2_uncertanty', 'beta_female_travel_uncertanty', 'uncertainty_income', 'uncertainty_income','uncertainty_income','uncertainty_income','uncertainty_income','uncertainty_income','uncertainty_income','uncertainty_income']
    declaration_data = ['local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local']
    value_data = [X[i,0],X[i,1], X[i,2], X[i,3], X[i,3], X[i,3], X[i,3], X[i,3], X[i,3], X[i,3], X[i,3]]
    data = {
        'model': model_data,
        'param_name': param_name_data,
        'declaration': declaration_data,
        'value': value_data
    }

    # Save the outputs of the model
    params = pd.DataFrame(data)
    outcome.append(fn_simulation(params))

X = np.array(X)
y1 = np.array([outcomelist[0] for outcomelist in outcome])
y2 = np.array([outcomelist[1] for outcomelist in outcome])
print(y1.shape)

#Save the results
with open('Y_train_1.pickle', 'wb') as handle:
    pickle.dump(y1, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('Y_train_2.pickle', 'wb') as handle:
    pickle.dump(y2, handle, protocol=pickle.HIGHEST_PROTOCOL)

best_rf_model = rf_fit(X, np.column_stack((y1, y2)))

predictions = make_predictions_rf(best_rf_model, X_test)

#Save the predicted results of the metamodel
with open('Prediction_test_2_outputs.pickle', 'wb') as handle:
    pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)