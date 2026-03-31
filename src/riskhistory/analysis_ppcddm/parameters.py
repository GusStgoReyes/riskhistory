import numpy as np

class DDMParameters:
    def __init__(self):
        pass
    
    def set_parameters_to_fit(self, parameters_list):
        """
        Set which parameters to fit by removing them from being constant if you want to fit them
        or adding them to conditions if you dont want to fit them. 
            - parameters_list : list
                List of parameter names to fit.
        """
        self.num_parameters = len(parameters_list)
        for param in self.parameters_names:
            if param in parameters_list:
                del self.parameters_constant[param]
            else:
                del self.parameters_bounds[param]
                self.conditions.append(param)

    def set_conditions_constant(self, conditions_list):
        """
        Set how to treat the conditions (e.g., the prev_choices)
            - parameters_list : list
                List of parameter names to be treated as constant conditions.
        """
        for param in conditions_list:
            self.parameters_constant[param] = 0
    
    def remove_parameters(self, parameters_list):
        """
        Remove parameters from the model so that they are not considered in the fitting or as conditions.
        Maybe I don't need this anymore. 
        """
        for param in parameters_list:
            if param not in self.parameters_names:
                raise ValueError(f"Parameter '{param}' is not a valid parameter name.")
            self.parameters_names.remove(param)
            del self.parameters_bounds[param]
            del self.parameters_constant[param]

class FullDDMParameters(DDMParameters):
    def __init__(self):
        self.parameters_names = ["v_scale",     # Drift rate scaling factor
                                 "v_gain",      # Weight on gain
                                 "v_loss",      # Weight on loss
                                 "alpha",       # Curvature of value function
                                 "v_Intercept", # Bias intercept term
                                 "z",           # Starting point
                                 "a",           # Boundary separation
                                 "theta",       # Boundary collapse angle
                                 "ter",           # Non-decision time
                                 "noise",       # Noise standard deviation
                                ]
        self.parameters_names += [f"v_choice_prev{i}" for i in range(1,7)]    # Lag parameters for utility
        self.parameters_names += [f"z_choice_prev{i}" for i in range(1,7)]    # Lag parameters for starting point
        self.parameters_names += [f"v_loss_choice_prev{i}" for i in range(1,7)] # Lag parameters for loss effects
        self.parameters_names += [f"v_gain_choice_prev{i}" for i in range(1,7)] # Lag parameters for gain effects
        
        self.num_parameters = len(self.parameters_names)

        self.parameters_bounds = {"v_scale" : (0.1, 5),
                                  "v_gain" : (0, 4),
                                  "v_loss" : (-8, 0),
                                  "alpha" : (0, 1.2), 
                                  "v_Intercept" : (-4, 4), 
                                  "z" : (-2, 2), 
                                  "a" : (0.5, 5), 
                                  "theta" : (0, np.pi/2), 
                                  "ter" : (0.2, 1.3),
                                  "noise" : (1, 1)}
        self.parameters_bounds.update({f"v_choice_prev{i}" : (-4, 4) for i in range(1,7)})
        self.parameters_bounds.update({f"z_choice_prev{i}" : (-0.5, 0.5) for i in range(1,7)})
        self.parameters_bounds.update({f"v_loss_choice_prev{i}" : (-8, 8) for i in range(1,7)})
        self.parameters_bounds.update({f"v_gain_choice_prev{i}" : (-4, 4) for i in range(1,7)})

        self.parameters_constant = {"v_scale" : 1,
                                    "v_gain" : 1,
                                    "v_loss" : -1,
                                    "alpha" : 1, 
                                    "v_Intercept" : 0, 
                                    "z" : 0,
                                    "a" : 1,
                                    "theta" : 0,
                                    "ter" : 0.6,
                                    "noise" : 1,
                                    }
        self.parameters_constant.update({f"v_choice_prev{i}" : 0 for i in range(1,7)})
        self.parameters_constant.update({f"z_choice_prev{i}" : 0 for i in range(1,7)})
        self.parameters_constant.update({f"v_loss_choice_prev{i}" : 0 for i in range(1,7)})
        self.parameters_constant.update({f"v_gain_choice_prev{i}" : 0 for i in range(1,7)})

        self.conditions = ["gain", 
                           "loss", 
                           "choice_prev1", 
                           "choice_prev2", 
                           "choice_prev3", 
                           "choice_prev4", 
                           "choice_prev5", 
                           "choice_prev6"]