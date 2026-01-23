import hddm
import pandas as pd
import numpy as np

def z_link_func(x):
    return 1 / (1 + np.exp(-x))

def aic(model):
    k = len(model.get_stochastics())
    logp = sum([x.logp for x in model.get_observeds()['node']])  
    return 2 * k - 2 * logp

def bic(model):
    k = len(model.get_stochastics())
    n = len(model.data)
    logp = sum([x.logp for x in model.get_observeds()['node']])
    return -2 * logp + k * np.log(n)

def load_data(data_pth,
              dataset_name, 
              subj_IDs = None,
              ):
    
    data = hddm.load_csv(
        data_pth
    ).query(f"dataset == '{dataset_name}'")

    if subj_IDs is not None:
        data = data.query(f"sub in {subj_IDs}")
    
    data = data.rename(columns={"accept": "response", "sub": "subj_idx"})
    data = data.query("exclude == False")
    data = hddm.utils.flip_errors(data)

    return data

def get_reg_descr(model_id):
    if model_id == "HDDM_Collapse":
        v_reg = {"model": f"v ~ 1 + gain + loss", "link_func": lambda x: x}
        reg_descr = [v_reg]
    elif model_id == "HDDM_Collapse_Lag1-c0":
        v_reg = {"model": f"v ~ 1 + gain + loss + choice_prev1", "link_func": lambda x: x}
        reg_descr = [v_reg]
    elif model_id == "HDDM_Collapse_Lag1-x0":
        v_reg = {"model": f"v ~ 1 + gain + loss", "link_func": lambda x: x}
        z_reg = {"model": f"z ~ 1 + choice_prev1", "link_func": z_link_func}
        reg_descr = [v_reg, z_reg]
    elif model_id == "HDDM_Collapse_Lag1-x0_c0":
        v_reg = {"model": f"v ~ 1 + gain + loss + choice_prev1", "link_func": lambda x: x}
        z_reg = {"model": f"z ~ 1 + choice_prev1", "link_func": z_link_func}
        reg_descr = [v_reg, z_reg]
    elif model_id == "HDDM_Collapse_Lag3-x0":
        v_reg = {"model": f"v ~ 1 + gain + loss", "link_func": lambda x: x}
        z_reg = {"model": f"z ~ 1 + choice_prev1 + choice_prev2 + choice_prev3", "link_func": z_link_func}
        reg_descr = [v_reg, z_reg]
    elif model_id == "HDDM_Collapse_Lag1-ndt":
        v_reg = {"model": f"v ~ 1 + gain + loss", "link_func": lambda x: x}
        t_reg = {"model": f"t ~ 1 + repeat", "link_func": lambda x: x}
        reg_descr = [v_reg, t_reg]
    else:
        reg_descr = None
        raise ValueError(f"Model ID '{model_id}' not recognized.")
    return reg_descr

def save_model_statistics(model, 
                          stats, 
                          model_id, 
                          dataset_name, 
                          results_pth, 
                          idx_save=""):
    
    df_model = stats[~stats['param_name'].str.contains('subj')]
    df_model = df_model[['param_name', 'mean', 'std']]
    df_model = df_model.rename(columns={'mean': 'param_value', 'std': 'param_std'})
    df_model["model"] = model_id
    df_model["dataset"] = dataset_name
    df_model["dic"] = model.dic
    df_model["aic"] = aic(model)
    df_model["bic"] = bic(model)

    df_model.to_csv(f"{results_pth}/hddm_stats/{model_id}_{dataset_name}{idx_save}_model_stats.csv")

def save_subj_statistics(stats, 
                         model_id, 
                         dataset_name, 
                         results_pth, 
                         idx_save=""):
    
    # Extract subject parameters (from previous step)
    df_subj = stats[stats['param_name'].str.contains('subj')]
    df_subj = df_subj[['param_name', 'mean', 'std']]
    df_subj = df_subj.rename(columns={'mean': 'param_value', 'std': 'param_std'})
    # Extract the subject ID (the number after 'subj.')
    df_subj['sub'] = df_subj['param_name'].str.extract(r'subj\.(\d+)').astype(int)
    # Remove the "_subj.X" portion from the param name
    df_subj['param_name'] = df_subj['param_name'].str.replace(r'_subj\.\d+', '', regex=True)
    df_subj["model"] = model_id
    df_subj["dataset"] = dataset_name

    df_subj.to_csv(f"{results_pth}/hddm_params/{model_id}_{dataset_name}{idx_save}_subj_params.csv")

def fit_model(model_id, 
              dataset_name,
              data_pth, 
              results_pth,
              subj_IDs = None,
              idx_save = "",
              burnin=2000,
              sampling_iters=10000,):
    
    data = load_data(data_pth = data_pth,
                     dataset_name = dataset_name, 
                     subj_IDs = subj_IDs, 
                     )
    
    reg_descr = get_reg_descr(model_id)

    m_reg = hddm.HDDMnnRegressor(
            data,
            reg_descr,
            model="angle",
            informative = False, # Set to False given this is not a perceptual task
            is_group_model = True,
            group_only_regressors = False,
            group_only_nodes = ["v_choice_prev1", "z_choice_prev1"],
            include=["v", "a", "t", "z", "theta"],
        )

    m_reg.sample(
        sampling_iters,
        burn=burnin,
        loglike = True,
        dbname=f"{results_pth}/db/{model_id}_{dataset_name}{idx_save}.db",
        db="pickle",
    )

    stats = m_reg.gen_stats()
    stats = stats.reset_index().rename(columns={'index': 'param_name'})
    print(stats, flush=True)
    # Save results
    save_model_statistics(m_reg, stats, model_id, dataset_name, results_pth, idx_save)
    save_subj_statistics(stats, model_id, dataset_name, results_pth, idx_save)
    m_reg.save(f"{results_pth}/hddm/{model_id}_{dataset_name}{idx_save}.hddm")

if __name__ == "__main__":
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser(description="Fit a DDM model to a subject's data")
    parser.add_argument(
        "--model_id", type=str, default = "HDDM_Collapse", help="The subject ID to fit the model to"
    )
    parser.add_argument(
        "--dataset_name", type=str, default = "equalIndifference", help="The name of the dataset to fit"
    )
    parser.add_argument(
        "--data_pth", default = "/scratch/users/gustxsr/cptpyddm/data/behavioral_data.csv", type=str, help="The path to the data"
    )
    parser.add_argument(
        "--results_pth", default = "/scratch/users/gustxsr/cptpyddm/results", type=str, help="The path to save results"
    )
    parser.add_argument(
        "--test", action="store_true", help="Whether to run a test fit with fewer iterations"
    )
    args = parser.parse_args()

    # For testing purposes
    if args.test:
        subj_IDs = [1, 3, 5, 9]
        idx_save = "_test2"
        burnin=2
        sampling_iters=10
    else:
        subj_IDs = None
        idx_save = "_histeffectsummary"
        burnin=2000
        sampling_iters=10000

    print(f"Fitting model {args.model_id} to dataset {args.dataset_name}...", flush=True)

    # Run the fitting
    fit_model(
        model_id = args.model_id,
        data_pth = args.data_pth,
        results_pth = args.results_pth,
        dataset_name = args.dataset_name,
        subj_IDs = subj_IDs,
        idx_save = idx_save,
        burnin=burnin,
        sampling_iters=sampling_iters,
    )
