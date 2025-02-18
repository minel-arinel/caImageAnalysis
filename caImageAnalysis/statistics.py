import numpy as np
from scipy.stats import boxcox, fligner, ks_2samp, normaltest, pearsonr, shapiro, spearmanr
from sklearn.preprocessing import PowerTransformer


global alpha # Significance level for hypothesis tests
alpha = 0.05


def fligner_test(a, b):
    """
    Perform the Fligner-Killeen test to compare the variances of two samples.
    Parameters:
        a (array-like): First sample data.
        b (array-like): Second sample data.
    Returns:
        None
            Displays the Fligner-Killeen statistic and p-value, and interprets the result.
    """
    # Run the Fligner-Killeen test
    statistic, p_value = fligner(a, b)

    # Display the results
    print(f"Fligner-Killeen Statistic: {statistic}")
    print(f"P-value: {p_value}")

    # Interpret the result
    if p_value < alpha:
        print("The variances are statistically significantly different (reject null hypothesis).")
        print("Run Kolmogorov-Smirnov test.")
    elif p_value < 0.1:
        print("The variances may be different (weak evidence against the null hypothesis).")
        print("Run Kolmogorov-Smirnov test.")
    else:
        print("The variances are not statistically significantly different (fail to reject null hypothesis).")
        print("Run Mann-Whitney U test.")


def check_normality(*arrays, verbose=True):
    """
    Check the normality and lognormality of multiple arrays using Shapiro-Wilk or D'Agostino & Pearson tests.
    Parameters:
        *arrays : array-like
            One or more arrays to be tested for normality.
        verbose : bool
            If True, prints the test results and interpretation for each array.
    Returns:
        list
            A list of booleans indicating whether each array is normal (True) or not (False).
    """
    normality_results = []

    # Replace None with np.nan
    arrays = [np.array(arr) for arr in arrays]
    arrays = [np.array(arr, dtype=float) for arr in arrays]
    arrays = [np.where(arr == None, np.nan, arr) for arr in arrays]

    # Flatten each array and remove NaN values
    flattened_arrays = [arr.flatten() for arr in arrays]
    flattened_arrays = [arr[~np.isnan(arr)] for arr in flattened_arrays]
    
    # Compare the lengths of these arrays
    for i, values in enumerate(flattened_arrays):
        # Run the appropriate normality test based on the longest length
        if len(values) < 50:
            stat, p = shapiro(values)
            test_name = "Shapiro-Wilk"
        else:
            stat, p = normaltest(values)
            test_name = "D'Agostino & Pearson"
        
        if verbose:
            print(f'{test_name} test for array {i+1} - Statistics={stat:.5f}, p={p:.5f}')
        
        # Interpret the result
        is_normal = p > alpha
        normality_results.append(is_normal)
        
        if verbose:
            if is_normal:
                print(f'Sample for array {i+1} looks Gaussian (normal)\n')
            else:
                print(f'Sample for array {i+1} does not look Gaussian (not normal)\n')

        # Check for lognormality
        if np.any(values <= 0):
            if verbose:
                print(f'Skipping lognormality test for array {i+1} due to non-positive values\n')
            continue
        else:
            log_values = np.log(values)  # Log-transform the values
            if len(log_values) < 50:
                stat, p = shapiro(log_values)
                test_name = "Shapiro-Wilk"
            else:
                stat, p = normaltest(log_values)
                test_name = "D'Agostino & Pearson"
            
            if verbose:
                print(f'{test_name} test for log-transformed array {i+1} - Statistics={stat:.5f}, p={p:.5f}')
            
            # Interpret the result
            if verbose:
                if p > alpha:
                    print(f'Sample for log-transformed array {i+1} looks Gaussian (lognormal)\n')
                else:
                    print(f'Sample for log-transformed array {i+1} does not look Gaussian (not lognormal)\n')

    return normality_results


def spearman_correlation(a, b, verbose=True):
    """
    Calculate the Spearman correlation coefficient between two arrays.
    Parameters:
        a (array-like): First array.
        b (array-like): Second array.
    Returns:
        tuple
            Spearman correlation coefficient and p-value.
    """
    correlation, p_value = spearmanr(a, b, nan_policy='omit')

    if verbose:
        print(f"Spearman correlation: {correlation:.5f}")
        print(f"P-value: {p_value:.5f}")
    
        if p_value < alpha:
            print("The correlation is statistically significant.")
        else:
            print("The correlation is not statistically significant.")
    
    return correlation, p_value


def spearman_correlation_repeated_measures(df):
    """
    Calculate the Spearman correlation coefficient for repeated measures data.
    Parameters:
        df (DataFrame): DataFrame where each column represents a different subject and each row represents a repeated measure.
    Returns:
        None
            Prints the mean Spearman correlation coefficient and mean p-value across all measures.
    """
    correlations = list()
    p_values = list()

    for col in df.columns:
        if df[col].notnull().any():
            correlation, p_value = spearman_correlation(list(df[col].index.values), list(df[col]), verbose=False)
            correlations.append(correlation)
            p_values.append(p_value)

    # Remove NaNs before calculating median
    correlations = [corr for corr in correlations if not np.isnan(corr)]
    p_values = [p for p in p_values if not np.isnan(p)]

    print(f"Median correlation: {np.median(correlations):.5f}")
    print(f"Median p-value: {np.median(p_values):.5f}")

    if np.mean(p_values) < alpha:
        print("The median correlation is statistically significant.")
    else:
        print("The median correlation is not statistically significant.")

    return correlations, p_values


def pearson_correlation(a, b, verbose=True):
    """
    Calculate the Pearson correlation coefficient between two arrays.
    Parameters:
        a (array-like): First array.
        b (array-like): Second array.
    Returns:
        tuple
            Pearson correlation coefficient and p-value.
    """
    correlation, p_value = pearsonr(a, b)

    if verbose:
        print(f"Pearson correlation: {correlation:.5f}")
        print(f"P-value: {p_value:.5f}")
    
        if p_value < alpha:
            print("The correlation is statistically significant.")
        else:
            print("The correlation is not statistically significant.")
    
    return correlation, p_value


def pearson_correlation_repeated_measures(df):
    """
    Calculate the Pearson correlation coefficient for repeated measures data.
    Parameters:
        df (DataFrame): DataFrame where each column represents a different subject and each row represents a repeated measure.
    Returns:
        None
            Prints the mean Pearson correlation coefficient and mean p-value across all measures.
    """
    correlations = list()
    p_values = list()

    for col in df.columns:
        if df[col].notnull().any():
            correlation, p_value = pearson_correlation(list(df[col].index.values), list(df[col]), verbose=False)
            correlations.append(correlation)
            p_values.append(p_value)

    # Remove NaNs before calculating median
    correlations = [corr for corr in correlations if not np.isnan(corr)]
    p_values = [p for p in p_values if not np.isnan(p)]

    print(f"Median correlation: {np.median(correlations):.5f}")
    print(f"Median p-value: {np.median(p_values):.5f}")

    if np.mean(p_values) < alpha:
        print("The median correlation is statistically significant.")
    else:
        print("The median correlation is not statistically significant.")

    return correlations, p_values


def check_monotonicity_repeated_measures(df, return_results=False):
    """
    Check the monotonicity of a repeated measures DataFrame.
    Parameters:
        df (DataFrame): DataFrame where each column represents a different subject and each row represents a repeated measure.
        return_results (bool): If True, returns the correlations and p-values.
    Returns:
        tuple (optional)
            If return_results is True, returns a tuple containing the correlations and p-values.
    """
    normality = check_normality(df, verbose=False)
    
    # Interpret the result and run the appropriate correlation test
    if normality[0]:
        print('Data looks Gaussian (normal). Running Pearson correlation.')
        correlations, p_values = pearson_correlation_repeated_measures(df)
    else:
        print('Data does not look Gaussian (not normal). Running Spearman correlation.')
        correlations, p_values = spearman_correlation_repeated_measures(df)

    if return_results:
        return correlations, p_values


def boxcox_transformation(data):
    """
    Apply the Box-Cox transformation to the data.
    Parameters:
        data (array-like): Data to be transformed.
    Returns:
        array-like
            Transformed data.
    """
    return boxcox(data)[0]


def yeo_johnson_transformation(data):
    """
    Apply the Yeo-Johnson transformation to the data.
    Parameters:
        data (array-like): Data to be transformed.
    Returns:
        array-like
            Transformed data.
    """
    transformer = PowerTransformer(method='yeo-johnson')
    return transformer.fit_transform(data.reshape(-1, 1))


def transform_to_parametric(*data, force_boxcox=False, shift=0.0001):
    """
    Apply the Box-Cox or Yeo-Johnson transformation to the data to make it more Gaussian-like.
    Parameters:
        *data : array-like
            One or more arrays to be transformed.
        force_boxcox : bool, optional
            If True, forces the use of Box-Cox transformation even if data contains non-positive values.
        shift : float, optional
            The value to shift the data by to make all values positive for Box-Cox transformation.
    Returns:
        list
            A list of transformed arrays.
    """
    transformed_data = []

    for arr in data:   
        arr = np.array(arr)
        if np.any(arr <= 0):
            if force_boxcox:
                print("Running Box-Cox transformation with shift")
                arr = arr + shift
                transformed_data.append(boxcox_transformation(arr))
            else:
                print("Running Yeo-Johnson transformation")
                transformed_data.append(yeo_johnson_transformation(arr))
        else:
            print("Running Box-Cox transformation")
            transformed_data.append(boxcox_transformation(arr))
        
    return transformed_data


def kolmogorov_smirnov_test(a, b, verbose=True):
    """
    Perform the Kolmogorov-Smirnov test to compare two samples.
    Parameters:
        a (array-like): First sample data.
        b (array-like): Second sample data.
    Returns:
        tuple
            Kolmogorov-Smirnov statistic and p-value.
    """
    statistic, p_value = ks_2samp(a, b)

    if verbose:
        print(f"Kolmogorov-Smirnov Statistic: {statistic:.5f}")
        print(f"P-value: {p_value:.5f}")
    
        if p_value < alpha:
            print("The distributions are statistically significantly different.")
        else:
            print("The distributions are not statistically significantly different.")
    
    return statistic, p_value