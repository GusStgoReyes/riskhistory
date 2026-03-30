import os
import pyddm
import numpy as np
import pandas as pd

class Model:
    def __init__(self):
        self.model = None
        self.data = None
        self.parameters_names = None
        self.parameter_fits = None
        self.num_parameters = None
        self.sub = None
        self.name = None
        self.dataset_name = None

    def set_data(self):
        raise NotImplementedError("This method should be implemented in subclasses.")
    def fit_model(self):
        raise NotImplementedError("This method should be implemented in subclasses.")
    def compute_loglikelihood_and_accuracy(self):
        raise NotImplementedError("This method should be implemented in subclasses.")
    def save_model_results(self):
        raise NotImplementedError("This method should be implemented in subclasses.")
    def save_model_results(self, 
                           file_path, 
                           simulated_params = None):
        """
            Save the fitted model parameters and performance metrics to a CSV file.
                - file_path: path to the CSV file where results will be saved
                - simulated_params: dictionary of true parameter values if data is simulated
        """
        if self.model is None:
            raise ValueError("Model not set. Please fit the model using fit_model() method before saving parameters.")
        
        data_LL, accuracy = self.compute_loglikelihood_and_accuracy() if self.data is not None else (None, None)
        fitted_parameters = dict()
        fitted_parameters["param_name"] = self.parameters_names
        fitted_parameters["param_value"] = [self.parameter_fits[name] for name in self.parameters_names]
        if simulated_params is not None:
            fitted_parameters["param_value_real"] = [simulated_params[name] for name in self.parameters_names]
        n_params = len(fitted_parameters["param_name"])
        fitted_parameters["sub"] = [self.sub] * n_params
        fitted_parameters["model"] = [self.name] * n_params
        fitted_parameters["dataset"] = [self.dataset_name] * n_params
        fitted_parameters["data_LL"] = [data_LL] * n_params
        fitted_parameters["accuracy"] = [accuracy] * n_params
        fitted_parameters["num_params"] = self.num_parameters
        fitted_parameters["num_datapoints"] = [len(self.data)] * n_params
        df = pd.DataFrame(fitted_parameters)
        df.to_csv(file_path, mode="a", index=False)
    
class PTDDM(Model):
    def __init__(self, 
                 name, 
                 drift, 
                 boundary, 
                 starting_position, 
                 nondecision, 
                 noise, 
                 parameters, 
                 dx=0.002, 
                 dt=0.002, 
                 T_dur=4):
        
        # Parameters for the DDM model
        self.dx = dx
        self.dt = dt
        self.T_dur = T_dur

        # Initialize model attributes
        self.drift = drift
        self.boundary = boundary
        self.starting_position = starting_position
        self.nondecision = nondecision
        self.noise = noise

        # Initialize parameters
        self.num_parameters = parameters.num_parameters
        self.parameters_names = parameters.parameters_names
        self.conditions = parameters.conditions
        self.parameters_bounds = parameters.parameters_bounds
        self.fixed_parameters = parameters.parameters_constant

        # Placeholder for data and fitting status
        self.name = name
        self.data = None
        self.model_parameters_set = False
        self.sub = None
        self.dataset_name = None

    def set_model(self):
        """
            Set up the pyddm model with the specified components and parameters.
        """
        self.model = pyddm.gddm(
            starting_position=self.starting_position,
            drift=self.drift,
            nondecision=self.nondecision,
            bound= self.boundary,
            noise = self.noise,
            parameters=self.parameters_bounds,
            conditions=self.conditions,
            T_dur=self.T_dur,
            dx=self.dx,
            dt=self.dt,
        )

        self.parameters_names = self.model.get_model_parameter_names()
    
    def set_data(self, 
                 data, 
                 sub, 
                 dataset_name,
                 rt_column_name="rt", 
                 choice_column_name="accept"):
        """
        Set the data for model fitting.
            - data: pandas DataFrame containing the data
            - sub: subject identifier
            - dataset_name: name of the dataset
                - rt_column_name: name of the column containing reaction times
                - choice_column_name: name of the column containing choices
        """
        
        self.data = data
        self.sub = sub
        self.dataset_name = dataset_name
        self.rt_column_name = rt_column_name
        self.choice_column_name = choice_column_name

        # If exlude column exists, filter out excluded trials
        if "exclude" in self.data.columns:
            self.data = self.data.query(f"exclude == False").reset_index(drop = True)

        # Set fixed parameters as conditions in the data
        for param in self.fixed_parameters:
            self.data[param] = self.fixed_parameters[param]
        
    def fit_model(self, 
                  fitting_method="bads"):
        """
            Fit the model to the data by minimizing the negative log-likelihood.
             - fitting_method: method to use for fitting ("bads" supported)
        """
        
        if self.data is None:
            raise ValueError("Data not set. Please set data using set_data() method before fitting the model.")
        if self.model_parameters_set:
            raise Exception("Model parameters already set. Please create a new instance of the model to fit with different parameters.")
        
        self.set_model()

        samples = pyddm.Sample.from_pandas_dataframe(
            self.data,
            rt_column_name=self.rt_column_name,
            choice_column_name=self.choice_column_name
        )
        
        # Fit the model
        self.model.fit(
            sample=samples,
            fitting_method=fitting_method,
        )

        self.model_parameters_set = True
        self.parameter_fits = {param : val.default() for param, val in zip(self.model.get_model_parameter_names(), self.model.get_model_parameters())}
        self.parameters_names = self.model.get_model_parameter_names()
        print(self.parameter_fits, flush = True)
        print(self.parameters_names, flush = True)
    def set_parameters(self, 
                       parameters):
        """
            Set the model parameters directly.
            - parameters: dictionary of parameter names and their values
        """
        if self.model_parameters_set:
            raise Exception("Model parameters already set. Please create a new instance of the model to fit with different parameters.")
        
        self.set_model()
        
        if len(parameters) != len(self.model.get_model_parameters()):
            raise ValueError("Number of parameters does not match the model's parameters.")
        
        parameters_values = [parameters[name] for name in self.model.get_model_parameter_names()] 
        self.model.set_model_parameters(parameters_values)
        self.model_parameters_set = True
    
    def generate_data(self, 
                      conditions, 
                      samples_per_condition=1,
                      rt_column_name="rt",
                      choice_column_name="accept"):
        """
            Generate artificial data based on a set of model parameters. Only allows for mixed gamble task. 
                - conditions: pandas DataFrame containing the conditions (gain, loss)
        """
        if not self.model_parameters_set:
            raise ValueError("Model parameters not set. Please fit the model using fit_model() method before generating data.")
        
        self.rt_column_name = rt_column_name
        self.choice_column_name = choice_column_name

        artificial_data = {"gain" : [], 
                           "loss" : [], 
                           "mean_rt" : [], 
                           "mean_accept" : [],
                           "pred_repeat" : []}
        
        # Generate data for each gain, loss condition
        for _, row in conditions.iterrows():
            sample = self.model.solve(
                conditions={"gain": row["gain"], "loss": row["loss"], 
                            "choice_prev1" : row["choice_prev1"], **self.fixed_parameters}
            ).sample(samples_per_condition)

            accept_rts = sample.choice_upper
            reject_rts = sample.choice_lower
            mean_accept = len(accept_rts) / (len(accept_rts) + len(reject_rts))
            mean_rt = (np.sum(accept_rts) + np.sum(reject_rts)) / (len(accept_rts) + len(reject_rts))
            if mean_accept > 0.5 and row["choice_prev1"] == 1:
                pred_repeat = mean_accept
            elif mean_accept <= 0.5 and row["choice_prev1"] == -1:
                pred_repeat = mean_accept
            else:
                pred_repeat = 1 - mean_accept

            artificial_data["gain"].extend([row["gain"]])
            artificial_data["loss"].extend([row["loss"]])
            artificial_data["mean_rt"].extend([mean_rt])
            artificial_data["mean_accept"].extend([mean_accept])
            artificial_data["pred_repeat"].extend([pred_repeat])

                
        artificial_data = pd.DataFrame(artificial_data)

        return artificial_data

    def compute_loglikelihood_and_accuracy(self):
        """
            Compute the log-likelihood of the data given the fitted model parameters.
            And compute the accuracy of the model's predictions.
        """
        data_loglikelihoods, accuracy = [], []
        for _, row in self.data.iterrows():
            solved_model = self.model.solve(conditions={cond: row[cond] for cond in self.conditions})
            log_pdf = np.log(solved_model.evaluate(row[self.rt_column_name], choice=row[self.choice_column_name]))
            correct = 1 if solved_model.prob(row[self.choice_column_name]) >= 0.5 else 0
            accuracy.append(correct)
            data_loglikelihoods.append(log_pdf)

        return np.sum(data_loglikelihoods), np.mean(accuracy)

    def mean_RT(self, solved_model, choice):
        """
            Compute the mean RT for a given choice from the solved model.
            Not sure why I have this here though. 
        """
        if choice == 1:
            return np.sum(solved_model.choice_upper*solved_model.t_domain) / solved_model.prob('correct')
        else:
            return np.sum(solved_model.choice_lower*solved_model.t_domain) / solved_model.prob('error')