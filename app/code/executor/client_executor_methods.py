import copy
import math
import os
import pandas as pd
import numpy as np
import numpy.linalg as la

from nvflare.apis.shareable import Shareable

from utils.types import CombatType, ConfigDTO
from utils.html_templates import get_results_page_template
from .client_input_preprocessor import validate_and_get_inputs
from .local_ancillary import interpolate_missing_data, identify_categorical_covariates, \
    encode_covariates, add_site_covariates

np.seterr(divide='ignore')

def parse_clientId(inp_str):
    num = ""
    for c in inp_str:
        if c.isdigit():
            num = num + c
    return int(num)

def convert_zeroes(x):
    x[x==0] = 1
    return x

def aprior(delta_hat):
    m = np.mean(delta_hat)
    s2 = np.var(delta_hat,ddof=1)
    return (2 * s2 +m**2) / float(s2)

def bprior(delta_hat):
    m = delta_hat.mean()
    s2 = np.var(delta_hat,ddof=1)
    return (m*s2+m**3)/s2

def postmean(g_hat, g_bar, n, d_star, t2):
    return (t2*n*g_hat+d_star * g_bar) / (t2*n+d_star)

def postvar(sum2, n, a, b):
    return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)

def convert_zeroes(x):
    x[x==0] = 1
    return x

def int_eprior(sdat, g_hat, d_hat):
    r = sdat.shape[0]
    gamma_star, delta_star = [], []
    for i in range(0,r,1):
        g = np.delete(g_hat,i)
        d = np.delete(d_hat,i)
        x = sdat[i,:]
        n = x.shape[0]
        j = np.repeat(1,n)
        A = np.repeat(x, g.shape[0])
        A = A.reshape(n,g.shape[0])
        A = np.transpose(A)
        B = np.repeat(g, n)
        B = B.reshape(g.shape[0],n)
        resid2 = np.square(A-B)
        sum2 = resid2.dot(j)
        LH = 1/(2*math.pi*d)**(n/2)*np.exp(-sum2/(2*d))
        LH = np.nan_to_num(LH)
        gamma_star.append(sum(g*LH)/sum(LH))
        delta_star.append(sum(d*LH)/sum(LH))
    adjust = (gamma_star, delta_star)
    return adjust


def find_non_parametric_adjustments(s_data, LS):
    gamma_star, delta_star = [], []
    temp = int_eprior(s_data, LS['gamma_hat'], LS['delta_hat'])
    gamma_star.append(temp[0])
    delta_star.append(temp[1])
    gamma_star = np.array(gamma_star)
    delta_star = np.array(delta_star)
    # raise Exception(gamma_star.shape, delta_star.shape)
    return gamma_star, delta_star


def adjust_data_final(s_data, batch_design, gamma_star, delta_star, stand_mean, mod_mean, var_pooled, dat, local_n_sample):
    bayesdata = s_data
    gamma_star = np.array(gamma_star)
    delta_star = np.array(delta_star)
    dsq = np.sqrt(delta_star)
    denom = np.dot(dsq.T, np.ones((1, local_n_sample)))
    numer = np.array(bayesdata  - np.dot(batch_design, gamma_star).T)
    bayesdata = numer / denom
    vpsq = np.sqrt(var_pooled).reshape((len(var_pooled), 1))

    bayesdata = bayesdata * np.dot(vpsq, np.ones((1, local_n_sample))) + stand_mean + mod_mean
    return bayesdata

def list_recursive(d, key):
    for k, v in d.items():
        if isinstance(v, dict):
            for found in list_recursive(v, key):
                yield found
        if k == key:
            yield v

def saveBin(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb+") as fh:
        header = "%s" % str(arr.dtype)
        for index in arr.shape:
            header += " %d" % index
        header += "\n"
        fh.write(header.encode())
        fh.write(arr.data.tobytes())
        os.fsync(fh)

def loadBin(path):
    with open(path, "rb") as fh:
        header = fh.readline().decode().split()
        dtype = header.pop(0)
        arrayDimensions = []
        for dimension in header:
            arrayDimensions.append(int(dimension))
        arrayDimensions = tuple(arrayDimensions)
        return np.frombuffer(fh.read(), dtype=dtype).reshape(arrayDimensions)


def perform_task_step1(config: ConfigDTO):
    input_directory = config.data_path
    
    computation_parameters = config.computation_params
    covariate_file_name = computation_parameters.get('covariate_file')
    data_file_name = computation_parameters.get('data_file')
    
    covariates_path = os.path.join(input_directory, covariate_file_name)
    data_path = os.path.join(input_directory,data_file_name)
    combat_algo = computation_parameters.get('combat_algo')

    config.logger.info('Checking file paths: ', covariates_path, data_path)
    config.cache_dict.update({
        'covar_urls': covariates_path,
        'data_urls': data_path,
        'lambda_value': 0,
        'combat_alg_type': combat_algo
    })
    
    result = {
        'cache': config.cache_dict,
        'output': {}
    }
    
    return result
    

def perform_task_step2(sharebale: Shareable, config: ConfigDTO):
    agg_results = sharebale.get('result')
    cache_dict = config.cache_dict
    
    covar_url = cache_dict.get('covar_urls')
    data_url = cache_dict.get('data_urls')
    lambda_value = cache_dict.get('lambda_value')
    combat_alg_type = cache_dict.get('combat_alg_type')
    
    config.logger.info('site files: ', covar_url, data_url, lambda_value, combat_alg_type)

    mat_X, mat_Y = validate_and_get_inputs(covar_url, data_url, combat_alg_type, config.computation_params, config.logger)
    config.logger.info('Completed Site Data Validation')
    
    X_cat = identify_categorical_covariates(mat_X)
    

    if str in X_cat:
        mat_X = encode_covariates(mat_X, X_cat)

    site_name = config.site_name
    site_index = agg_results['site_indexes'][site_name]
    
    X = mat_X
    Y = mat_Y.values
    Y_columns = mat_Y.columns 
    sample_count = len(Y)

    # Interpolation Missing Data
    if combat_alg_type == CombatType.COMBAT_MEGA_DC.value:
        Y = interpolate_missing_data(Y.T,X.to_numpy()).T # convert nan values to NULL/None .replace({np.nan: None}
    
    # Save Data File
    saveBin(os.path.join(config.cache_path, "Y.npy"), Y)

    # Add Site Data to Covariates to Prep for COMBAT
    augmented_X = add_site_covariates(agg_results, site_name, X, sample_count)
    biased_X = augmented_X.values 

    # Save Covariate File
    saveBin(os.path.join(config.cache_path, "X.npy"), np.array(list(augmented_X.to_numpy()),dtype=float))

    XtransposeX_local = np.matmul(np.matrix.transpose(biased_X), biased_X)
    Xtransposey_local = np.matmul(np.matrix.transpose(biased_X), Y)

    output_dict = {
        "local_sample_count": sample_count,
        "XtransposeX_local": XtransposeX_local.tolist(),
        "Xtransposey_local": Xtransposey_local.tolist(),
        "lambda_value": lambda_value,
        "site_index": site_index,
    }
    config.cache_dict.update({
        "local_sample_count": sample_count,
        "data_names":pd.DataFrame(Y_columns).to_json(orient='split'),
        "site_index": site_index,
        "combat_alg_type": combat_alg_type
    })

    result = {"output": output_dict, "cache": config.cache_dict}

    return result


def perform_task_step3(sharebale: Shareable, config: ConfigDTO):
    agg_results = sharebale.get('result')
    cache_dict = config.cache_dict

    covar = loadBin(os.path.join(config.cache_path, "X.npy"))
    mat_Y = loadBin(os.path.join(config.cache_path, "Y.npy"))
    
    data = np.array(np.transpose(mat_Y))
    data_columns = pd.read_json(cache_dict.get("data_names"), orient='split').values
    
    config.logger.info('data_columns: ', data_columns)
    
    design = covar
    B_hat = np.array(agg_results["B_hat"])
    n_sample = agg_results["n_sample"]
    n_batch = agg_results["n_batch"]
    site_array = agg_results["site_array"]
    local_n_sample = cache_dict.get("local_sample_count")
    
    local_var_pooled = np.dot(((data - np.dot(design, B_hat).T)**2), np.ones((local_n_sample, 1)) / float(n_sample))
    stand_mean = np.array(agg_results["stand_mean"])
    mod_mean = 0
    
    if design is not None:
        tmp = copy.deepcopy(design)
        tmp[:,range(-n_batch,0)] = 0
        mod_mean = np.transpose(np.dot(tmp, B_hat))
    
    config.cache_dict.update({
        "mod_mean": mod_mean.tolist(),
        "stand_mean": stand_mean.tolist(),
        "site_array": site_array,
        "n_batch": n_batch,
    })
    
    output_dict = {
       "local_var_pooled": local_var_pooled.tolist(),
       "computation_phase": "local_2",
       "data_names":pd.DataFrame(data_columns).to_json(orient='split')
    }
    
    result = {
        'output': output_dict,
        'cache': config.cache_dict
    }
    
    return result
    

def perform_task_step4(sharebale: Shareable, config: ConfigDTO):
    agg_results = sharebale.get('result')
    cache_dict = config.cache_dict
    
    var_pooled = np.array(agg_results['global_var_pooled'])
    data = np.array(np.transpose(loadBin(os.path.join(config.cache_path, 'Y.npy'))))
    stand_mean = np.array(cache_dict.get('stand_mean')).T
    mod_mean = np.array(cache_dict.get('mod_mean'))
    local_n_sample = cache_dict.get('local_sample_count')
    
    site_index = cache_dict.get('site_index')
    site_array = cache_dict.get('site_array')
    
    config.logger.info('site4: stand_mean: ', {
            'stand_mean': stand_mean, 
            'mod_mean': mod_mean, 
            'local_n_sample': local_n_sample, 
            'site_index': site_index
        }
    )
    
    indices = [index for index,element in enumerate(site_array) if element== int(site_index)]
    filtered_mean = stand_mean[indices]
    local_stand_mean = filtered_mean.T
    s_data = ((data - local_stand_mean - mod_mean) / np.dot(np.sqrt(var_pooled), np.ones((1, local_n_sample))))
    
    batch_design = np.array([[1]*local_n_sample]).T
    gamma_hat = np.dot(np.dot(la.inv(np.dot(batch_design.T, batch_design)), batch_design.T), s_data.T)
    delta_hat = []
    delta_hat.append(np.var(s_data, axis=1, ddof=1))
    delta_hat = list(map(convert_zeroes, delta_hat))
    gamma_bar = np.mean(gamma_hat, axis=1)
    t2 = np.var(gamma_hat, axis=1, ddof=1)
    
    a_prior = list(map(aprior, delta_hat))
    b_prior = list(map(bprior, delta_hat))
    LS_dict = {}
    LS_dict['gamma_hat'] = gamma_hat
    LS_dict['delta_hat'] = delta_hat
    LS_dict['gamma_bar'] = gamma_bar
    LS_dict['t2'] = t2
    LS_dict['a_prior'] = a_prior
    LS_dict['b_prior'] = b_prior
    gamma_star, delta_star = find_non_parametric_adjustments(s_data, LS_dict)
    bayesdata = adjust_data_final(s_data, batch_design,gamma_star, delta_star, local_stand_mean, mod_mean, var_pooled, data, local_n_sample)
    harmonized_data = np.transpose(bayesdata)
    
    mat_Y_labels = pd.read_json(cache_dict.get('data_names'), orient='split').values
    
    df = pd.DataFrame(harmonized_data, columns=mat_Y_labels)
    output_name = 'harmonized_site_'+str(site_index)+'_data.csv'
    config.logger.info('results path: ',config.output_path, output_name)
    df.to_csv(os.path.join(config.output_path, output_name), index=False)

    html_content = get_results_page_template(output_name)
    with open(os.path.join(config.output_path, "index.html"), "w") as html_file:
        html_file.write(html_content)
