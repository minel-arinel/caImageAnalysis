from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import sem

from caImageAnalysis.temporal_old import sort_by_peak_with_indices


def get_classified_neurons(sorted_components, last_component_idxs_combined, last_stimulus_idxs):
	'''Returns the classified neuron weights and their indices for each component'''
	classified_neurons = dict()
	classified_neuron_idxs = dict()
	for a in range(sorted_components.shape[0]):  # for each neuron component
		starts = list()
		ends = list()
		try:
			stops = last_component_idxs_combined[:, a]

			if a == 0:  # for the first component
				for s, stop in enumerate(stops):
					if s == 0:  # for egg water
						start = 0
					else:  # other stimuli
						start = last_stimulus_idxs[s-1] + 1
						stop = start + stop
					starts.append(start)
					ends.append(stop)

			else:  # other neuron components
				for s, stop in enumerate(stops):
					if s == 0:  # for egg water
						# find the stop point of the previous component range and add 1
						start = last_component_idxs_combined[s, a-1] + 1
					else:
						prev_comp_len = last_component_idxs_combined[s, a-1]
						prev_start = last_stimulus_idxs[s-1] + 1
						prev_stop = prev_start + prev_comp_len
						start = prev_stop + 1
						stop = prev_start + stop
					starts.append(start)
					ends.append(stop)

		except IndexError:  # for the final component
			for s, stop in enumerate(last_stimulus_idxs):
				if s == 0:  # for egg water
					# find the stop point of the previous component range and add 1
					start = last_component_idxs_combined[s, a-1] + 1
				else:
					prev_comp_len = last_component_idxs_combined[s, a-1]
					prev_start = last_stimulus_idxs[s-1] + 1
					prev_stop = prev_start + prev_comp_len
					start = prev_stop + 1
				starts.append(start)
				ends.append(stop)

			# spans for the final stimulus
			s = 3
			prev_comp_len = last_component_idxs_combined[s, a-1]
			prev_start = last_stimulus_idxs[s-1] + 1
			prev_stop = prev_start + prev_comp_len
			start = prev_stop + 1
			stop = sorted_components.shape[1]-1
			starts.append(start)
			ends.append(stop)

		cl_neurons = list()
		cl_neuron_idxs = list()
		for start, end in zip(starts, ends):
			cl_neurons.extend(sorted_components[a][start:end+1])
			cl_neuron_idxs.extend(np.arange(start, end+1))
		classified_neurons[a] = np.array(cl_neurons)
		classified_neuron_idxs[a] = np.array(cl_neuron_idxs)

	return classified_neurons, classified_neuron_idxs


def plot_top_n_neurons(sorted_components, sorted_data, classified_neurons, classified_neuron_idxs, final_sorting_idx, N_neurons=1, sorted_reconst=None):
	'''Plots the average of top N neurons for each component.
	Only the classified neurons are used for the weight sorting.
	sorted_reconst: if an array is given, will overlay the reconstructed trace average on top of the actual data'''
	for a in range(sorted_components.shape[0]):  # for each neuron component
		highest_weight_idxs = np.argpartition(np.abs(classified_neurons[a]), -N_neurons)[-N_neurons:]
		highest_weight_neurons = sorted_data[:, classified_neuron_idxs[a][highest_weight_idxs], :]

		# if there are any top neurons with negative weights, flip those traces
		if len(np.where(classified_neurons[a][highest_weight_idxs] < 0)[0]) > 0:
			for i in np.where(classified_neurons[a][highest_weight_idxs] < 0)[0]:
				highest_weight_neurons[:, i, :] *= -1

		# highest_weight_neurons = np.array([d / d.max() for d in np.array([d - d.min() for d in highest_weight_neurons])])

		avg_trace = np.mean(highest_weight_neurons, axis=1)
		sems = sem(highest_weight_neurons, axis=1)

		if sorted_reconst is not None:
			highest_weight_reconst_neurons = sorted_reconst[:, classified_neuron_idxs[a][highest_weight_idxs], :]

			# if there are any top neurons with negative weights, flip those traces
			if len(np.where(classified_neurons[a][highest_weight_idxs] < 0)[0]) > 0:
				for i in np.where(classified_neurons[a][highest_weight_idxs] < 0)[0]:
					highest_weight_reconst_neurons[:, i, :] *= -1

			# highest_weight_reconst_neurons = np.array([d / d.max() for d in np.array([d - d.min() for d in highest_weight_reconst_neurons])])
			avg_trace_reconst = np.mean(highest_weight_reconst_neurons, axis=1)
			sems_reconst = sem(highest_weight_reconst_neurons, axis=1)

		fig, axs = plt.subplots(1, sorted_data.shape[0], sharex=True, sharey=True, figsize=(10, 2))
		for p, pulse in enumerate(avg_trace):
			axs[p].plot(np.arange(-3, 14), pulse)
			axs[p].fill_between(np.arange(-3, 14), pulse-sems[p], pulse+sems[p], alpha=0.2)
			axs[p].axhline(0, ls="dashed", color="black")
			axs[p].axvspan(-1, 0, color="red", alpha=0.2)

			if sorted_reconst is not None:
				axs[p].plot(np.arange(-3, 14), avg_trace_reconst[p], label="Reconstruction", linestyle="dashed")
				axs[p].fill_between(np.arange(-3, 14), avg_trace_reconst[p]-sems_reconst[p], avg_trace_reconst[p]+sems_reconst[p], alpha=0.2)

		if sorted_reconst is not None:
			plt.legend()
		
		if N_neurons == 1:
			fig.suptitle(f"Component {a} - Neuron {final_sorting_idx[classified_neuron_idxs[a][highest_weight_idxs]][0]} - Weight index {classified_neuron_idxs[a][highest_weight_idxs][0]}")
		else:
			fig.suptitle(f"Component {a} - Average of {N_neurons} neurons")
		# print(highest_weight_idxs)


def plot_all_classified_neurons(sorted_components, sorted_data, classified_neurons, classified_neuron_idxs, sorted_reconst=None, savefig=False, save_folder=None):
	'''Plots the average of all neurons for each component.
	Only the classified neurons are used for the weight sorting.
	sorted_reconst: if an array is given, will overlay the reconstructed trace average on top of the actual data'''
	for a in range(sorted_components.shape[0]):  # for each neuron component
		highest_weight_neurons = sorted_data[:, classified_neuron_idxs[a], :]

		# if there are any top neurons with negative weights, flip those traces
		if len(np.where(classified_neurons[a] < 0)[0]) > 0:
			for i in np.where(classified_neurons[a] < 0)[0]:
				highest_weight_neurons[:, i, :] *= -1

		avg_trace = np.mean(highest_weight_neurons, axis=1)
		sems = sem(highest_weight_neurons, axis=1)

		if sorted_reconst is not None:
			highest_weight_reconst_neurons = sorted_reconst[:, classified_neuron_idxs[a], :]

			# if there are any top neurons with negative weights, flip those traces
			if len(np.where(classified_neurons[a] < 0)[0]) > 0:
				for i in np.where(classified_neurons[a] < 0)[0]:
					highest_weight_reconst_neurons[:, i, :] *= -1

			avg_trace_reconst = np.mean(highest_weight_reconst_neurons, axis=1)
			sems_reconst = sem(highest_weight_reconst_neurons, axis=1)

		fig, axs = plt.subplots(1, sorted_data.shape[0], sharex=True, sharey=True, figsize=(10, 2))
		for p, pulse in enumerate(avg_trace):
			axs[p].plot(np.arange(-3, 14), pulse)
			axs[p].fill_between(np.arange(-3, 14), pulse-sems[p], pulse+sems[p], alpha=0.2)
			axs[p].axhline(0, ls="dashed", color="black")
			axs[p].axvspan(-1, 0, color="red", alpha=0.2)

			if sorted_reconst is not None:
				axs[p].plot(np.arange(-3, 14), avg_trace_reconst[p], label="Reconstruction", linestyle="dashed")
				axs[p].fill_between(np.arange(-3, 14), avg_trace_reconst[p]-sems_reconst[p], avg_trace_reconst[p]+sems_reconst[p], alpha=0.2)

		if sorted_reconst is not None:
			plt.legend()
		
		fig.suptitle(f"Component {a} - Average of all neurons")
		# print(highest_weight_idxs)

		if savefig:
			plt.savefig(save_folder.joinpath(f"component{a}_avg_all_classified.pdf"), transparent=True)


def classify_neurons_by_weights(components):
	"""
	Classify and sort neurons according to the component they had the highest weight in.

	Input
	-------
	SliceTCA components

	Returns
	-------
	sorted_comps, components now with neurons sorted.
	sort_fac, sort indices to keep track of cell identity
	sort_max_fac, list of component indices per neuron, showing 

	"""
	# we take the absolute value in case the sliceTCA is initialized as "uniform"
	max_fac = np.argmax(np.abs(components), axis=0)
	sort_fac = np.argsort(max_fac)
	sort_max_fac = max_fac[sort_fac]

	# sort all neurons across all components based on the component they belong to
	# (i.e., neurons that belong to component 0, 1, ..., n are grouped together)
	sorted_comps = components[:, sort_fac]

	return sorted_comps, sort_fac, sort_max_fac


def sort_neurons_by_weight(components):
	"""
	Sort neurons by which component they contribute to the most.
	Within each component-responsive sort, neurons are further organized by their weights in descending order.

	Input
	-------
	SliceTCA components

	Returns
	-------
	sorted_comps, copy of components now with neurons sorted.
	full_sort, sort indices to keep track of cell identity

	"""
	components = deepcopy(components)

	first_sort, sort_fac, sort_max_fac = classify_neurons_by_weights(components)

	# descending sort within each group of sorted neurons
	second_sort = list()
	for i in np.unique(sort_max_fac):
		second_inds = (np.where(sort_max_fac == i)[0])
		second_sub_sort = np.argsort(first_sort[i, sort_max_fac == i])
		second_sort.extend(second_inds[second_sub_sort][::-1])

	# apply the second sort
	full_sort = sort_fac[second_sort]
	sorted_comps = components[:, full_sort]

	return sorted_comps, full_sort


def compute_covariance_matrix(data):
    """
    Computes the covariance matrix along the time axis for each neuron across trials.
    
    Parameters:
    data (numpy array): 3D array (trials x neurons x time points)
    
    Returns:
    covariance_matrix (numpy array): Covariance matrix (time points x time points)
    """
    # Reshape the data to combine trials and neurons into a single dimension (flattening across neurons and trials)
    reshaped_data = np.reshape(data, (-1, data.shape[2]))  # Reshape to (neurons*trials, time points)
    
    # Compute covariance matrix along the time axis
    covariance_matrix = np.cov(reshaped_data, rowvar=False)
    
    return covariance_matrix


def perform_eigenvalue_decomposition(covariance_matrix):
    """
    Performs eigenvalue decomposition on the covariance matrix.
    
    Parameters:
    covariance_matrix (numpy array): Covariance matrix (time points x time points)
    
    Returns:
    eigenvalues (numpy array): Eigenvalues in descending order
    eigenvectors (numpy array): Corresponding eigenvectors
    """
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Sort eigenvalues in descending order and corresponding eigenvectors
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return eigenvalues, eigenvectors


def plot_eigenvalue_decay(eigenvalues):
    """
    Plots the decay of the eigenvalues to visualize how quickly they drop off.
    
    Parameters:
    eigenvalues (numpy array): Eigenvalues in descending order
    """
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, 'o-', label='Eigenvalues')
    plt.xlabel('Component Number')
    plt.ylabel('Eigenvalue')
    plt.title('Eigenvalue Decay (Temporal Structure)')
    plt.grid(True)
    plt.show()
   
    
def calculate_variance_explained(eigenvalues, threshold=0.9):
    """
    Calculate how many eigenvalues explain the threshold percentage of the variance.
    
    Parameters:
    eigenvalues (numpy array): Eigenvalues in descending order
    threshold (float): The proportion of variance to explain (default is 90%)
    
    Returns:
    num_components (int): The number of components (time points) needed to explain the threshold variance
    """
    total_variance = np.sum(eigenvalues)
    cumulative_variance = np.cumsum(eigenvalues) / total_variance
    
    # Find the number of components needed to explain the threshold variance
    num_components = np.searchsorted(cumulative_variance, threshold) + 1
    
    return num_components


def rand_jitter(arr):
    return arr + np.random.normal(0, 0.1, len(arr))


def jitter(x, y, s=20, color='black', marker='o', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, hold=None, **kwargs):
    return plt.scatter(rand_jitter(x), y, s=s, color=color, marker=marker, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, alpha=alpha, linewidths=linewidths, **kwargs)


def get_all_component_weights(sorted_components, ordered_regions, ordered_stims):
	'''Returns a dictionary of neuron weights per component.
	Only the classified neurons are used for the component weights.'''
	all_weights = dict()
	
	for a in range(sorted_components.shape[0]):  # for each neuron component
		weights_per_comp = dict()
		
		for r, s, c in zip(ordered_regions, ordered_stims, sorted_components[a]):
			if r == "vagal_L": r = "_L"
			elif r == "vagal_R": r = "_R"
			else: r = ""
			
			if s == "eggwater": s = "Water"
			elif s == "glucose": s = "Glu"
			elif s == "glycine": s = "Gly"
			elif s == "zAITC": s = "AITC"

			col_name = s + r
			if col_name not in weights_per_comp:
				weights_per_comp[col_name] = list()

			weights_per_comp[col_name].append(c)

		weights_per_comp = dict([(k,pd.Series(v)) for k,v in weights_per_comp.items()])
		all_weights[a] = pd.DataFrame(weights_per_comp)

	return all_weights


def get_clustered_component_weights(sorted_components, classified_neurons, classified_neuron_idxs, ordered_regions, ordered_stims):
	'''Returns a dictionary of neuron weights per component.
	Only the classified neurons are used for the component weights.'''
	all_weights = dict()
	
	for a in range(sorted_components.shape[0]):  # for each neuron component
		weights_per_comp = dict()
		
		for i, idx in enumerate(classified_neuron_idxs[a]):
			r = ordered_regions[idx]
			s = ordered_stims[idx]

			if r == "vagal_L": r = "_L"
			elif r == "vagal_R": r = "_R"
			else: r = ""
			
			if s == "eggwater": s = "Water"
			elif s == "glucose": s = "Glu"
			elif s == "glycine": s = "Gly"
			elif s == "zAITC": s = "AITC"

			col_name = s + r
			if col_name not in weights_per_comp:
				weights_per_comp[col_name] = list()

			weights_per_comp[col_name].append(classified_neurons[a][i])

		weights_per_comp = dict([(k,pd.Series(v)) for k,v in weights_per_comp.items()])
		all_weights[a] = pd.DataFrame(weights_per_comp)

	return all_weights


def get_clustered_component_percentages(sorted_df, classified_neuron_idxs, ordered_regions, ordered_stims):
	'''Returns a dictionary of percentages of clustered neurons per component.
	Only the classified neurons are used for the component weights.'''
	all_percentages = dict()
	for c in classified_neuron_idxs.keys():
		percentages_per_comp = dict()		
		subdf = sorted_df.loc[classified_neuron_idxs[c], :]

		for s in np.unique(ordered_stims):
			for r in np.unique(ordered_regions):
				if r == "vagal_L": _r = "_L"
				elif r == "vagal_R": _r = "_R"
				else: _r = ""
				
				if s == "eggwater": _s = "Water"
				elif s == "glucose": _s = "Glu"
				elif s == "glycine": _s = "Gly"
				elif s == "zAITC": 
					_s = "AITC"
					s = "AITC"
				elif s == "AITC": _s = "AITC"

				col_name = _s + _r

				_df = subdf[(subdf.stimulus == s) & (subdf.region == r)]

				if col_name not in percentages_per_comp:
					percentages_per_comp[col_name] = list()

				for _id in sorted_df[(sorted_df.stimulus == s) & (sorted_df.region == r)].fish_id.unique():
					n = len(_df[_df.fish_id == _id])
					N = len(sorted_df[sorted_df.fish_id == _id])
					percentages_per_comp[col_name].append(n/N * 100)
		
		percentages_per_comp = dict([(k,pd.Series(v)) for k,v in percentages_per_comp.items()])
		df = pd.DataFrame(percentages_per_comp)

		if "vagal_L" in ordered_regions or "vagal_R" in ordered_regions:
			df = df[["Water_L", "Water_R", "Glu_L", "Glu_R", "Gly_L", "Gly_R", "AITC_L", "AITC_R"]]
		else:
			df = df[["Water", "Glu", "Gly", "AITC"]]
			
		all_percentages[c] = df

	return all_percentages


def plot_random_neuron(sorted_data, n=1, sorted_reconst=None):
	'''Plots traces of n random neurons.
	n: number of random neurons to plot
	sorted_reconst: if an array is given, will overlay the reconstructed trace on top of the actual data'''
	idxs = np.random.choice(range(sorted_data.shape[1]), n, replace=False)

	for neuron in idxs:
		avg_trace = sorted_data[:, neuron, :]
		# highest_weight_neurons = np.array([d / d.max() for d in np.array([d - d.min() for d in highest_weight_neurons])])

		if sorted_reconst is not None:
			avg_trace_reconst = sorted_reconst[:, neuron, :]
			# highest_weight_reconst_neurons = np.array([d / d.max() for d in np.array([d - d.min() for d in highest_weight_reconst_neurons])])

		fig, axs = plt.subplots(1, sorted_data.shape[0], sharex=True, sharey=True, figsize=(10, 2))
		for p, pulse in enumerate(avg_trace):
			axs[p].plot(np.arange(-3, 14), pulse)
			axs[p].axhline(0, ls="dashed", color="black")
			axs[p].axvspan(-1, 0, color="red", alpha=0.2)

			if sorted_reconst is not None:
				axs[p].plot(np.arange(-3, 14), avg_trace_reconst[p], label="Reconstruction", linestyle="dashed")

		if sorted_reconst is not None:
			plt.legend()
		
		fig.suptitle(f"Neuron {neuron}")


def plot_neurons(sorted_data, sorted_reconst=None):
	'''Plots individual traces of all neurons in sorted_data.
	sorted_reconst: if an array is given, will overlay the reconstructed trace on top of the actual data'''
	for neuron in range(sorted_data.shape[1]):
		avg_trace = sorted_data[:, neuron, :]
		# highest_weight_neurons = np.array([d / d.max() for d in np.array([d - d.min() for d in highest_weight_neurons])])

		if sorted_reconst is not None:
			avg_trace_reconst = sorted_reconst[:, neuron, :]
			# highest_weight_reconst_neurons = np.array([d / d.max() for d in np.array([d - d.min() for d in highest_weight_reconst_neurons])])

		fig, axs = plt.subplots(1, sorted_data.shape[0], sharex=True, sharey=True, figsize=(10, 2))
		for p, pulse in enumerate(avg_trace):
			axs[p].plot(np.arange(-3, 14), pulse)
			axs[p].axhline(0, ls="dashed", color="black")
			axs[p].axvspan(-1, 0, color="red", alpha=0.2)

			if sorted_reconst is not None:
				axs[p].plot(np.arange(-3, 14), avg_trace_reconst[p], label="Reconstruction", linestyle="dashed")

		if sorted_reconst is not None:
			plt.legend()


def get_df_with_classified_neurons(a, sorted_components, sorted_df, last_component_idxs_combined, last_stimulus_idxs):
	'''Returns the sorted dataframe with the classified neurons.
	The raw_norm_temporal column of the dataframe is flipped if the weight for that neuron are negative
	a: component index'''
	starts = list()
	ends = list()
	try:
		stops = last_component_idxs_combined[:, a]

		if a == 0:  # for the first component
			for s, stop in enumerate(stops):
				if s == 0:  # for egg water
					start = 0
				else:  # other stimuli
					start = last_stimulus_idxs[s-1] + 1
					stop = start + stop
				starts.append(start)
				ends.append(stop)

		else:  # other neuron components
			for s, stop in enumerate(stops):
				if s == 0:  # for egg water
					# find the stop point of the previous component range and add 1
					start = last_component_idxs_combined[s, a-1] + 1
				else:
					prev_comp_len = last_component_idxs_combined[s, a-1]
					prev_start = last_stimulus_idxs[s-1] + 1
					prev_stop = prev_start + prev_comp_len
					start = prev_stop + 1
					stop = prev_start + stop
				starts.append(start)
				ends.append(stop)

	except IndexError:  # for the final component
		for s, stop in enumerate(last_stimulus_idxs):
			if s == 0:  # for egg water
				# find the stop point of the previous component range and add 1
				start = last_component_idxs_combined[s, a-1] + 1
			else:
				prev_comp_len = last_component_idxs_combined[s, a-1]
				prev_start = last_stimulus_idxs[s-1] + 1
				prev_stop = prev_start + prev_comp_len
				start = prev_stop + 1
			starts.append(start)
			ends.append(stop)

		# spans for the final stimulus
		s = 3
		prev_comp_len = last_component_idxs_combined[s, a-1]
		prev_start = last_stimulus_idxs[s-1] + 1
		prev_stop = prev_start + prev_comp_len
		start = prev_stop + 1
		stop = sorted_components.shape[1]-1
		starts.append(start)
		ends.append(stop)

	classified_neurons = list()
	classified_neuron_idxs = list()
	for start, end in zip(starts, ends):
		classified_neurons.extend(sorted_components[a][start:end+1])
		classified_neuron_idxs.extend(np.arange(start, end+1))
	classified_neurons = np.array(classified_neurons)
	classified_neuron_idxs = np.array(classified_neuron_idxs)

	sorted_df_neurons = sorted_df.loc[sorted_df.index.isin(classified_neuron_idxs)].reset_index().drop(columns=["index"])

	# if there are any neurons with negative weights, flip those traces
	if len(np.where(classified_neurons < 0)[0]) > 0:
		for i in np.where(classified_neurons < 0)[0]:
			sorted_df_neurons.at[i, "raw_norm_temporal"] = sorted_df_neurons.loc[i, "raw_norm_temporal"] * -1

	return sorted_df_neurons


def plot_all_neurons(sorted_data, sorted_reconst=None):
	'''Plots the average of all neurons in the data.
	sorted_reconst: if an array is given, will overlay the reconstructed trace average on top of the actual data'''
	avg_trace = np.mean(sorted_data, axis=1)
	sems = sem(sorted_data, axis=1)

	if sorted_reconst is not None:
		avg_trace_reconst = np.mean(sorted_reconst, axis=1)
		sems_reconst = sem(sorted_reconst, axis=1)

	fig, axs = plt.subplots(1, sorted_data.shape[0], sharex=True, sharey=True, figsize=(10, 2))
	for p, pulse in enumerate(avg_trace):
		axs[p].plot(np.arange(-3, 14), pulse)
		axs[p].fill_between(np.arange(-3, 14), pulse-sems[p], pulse+sems[p], alpha=0.2)
		axs[p].axhline(0, ls="dashed", color="black")
		axs[p].axvspan(-1, 0, color="red", alpha=0.2)

		if sorted_reconst is not None:
			axs[p].plot(np.arange(-3, 14), avg_trace_reconst[p], label="Reconstruction", linestyle="dashed")
			axs[p].fill_between(np.arange(-3, 14), avg_trace_reconst[p]-sems_reconst[p], avg_trace_reconst[p]+sems_reconst[p], alpha=0.2)

	if sorted_reconst is not None:
		plt.legend()
	
	fig.suptitle(f"Average of all neurons")


def sort_by_stimulus_region(stimuli, regions):
	_stimuli = np.array(["zAITC" if s == "AITC" else s for s in stimuli[0]])
	stimulus_sorting = np.argsort(_stimuli)

	last_indices = list()  # include the end frame
	count = 0
	stimulus_region_sorting = list()
	for s in np.unique(_stimuli[stimulus_sorting]):
		length = len(_stimuli[stimulus_sorting][np.where(_stimuli[stimulus_sorting] == s)])
		last_idx = count + length - 1
		last_indices.append(last_idx)

		sorted_region_idxs = np.argsort(regions[0][stimulus_sorting][count:last_idx+1])  # regions sorted in a single stim
		sorted_idxs = stimulus_sorting[count:last_idx+1][sorted_region_idxs]  # stimulus sorting indices are sorted by region as well
		stimulus_region_sorting.extend(sorted_idxs)

		count += length

	# Now order by when the peak response occurs
	stimulus_region_sorting = np.array(stimulus_region_sorting)  # orders both by stim and by region

	return stimulus_region_sorting


def order_by_trial_components(components, stimulus_region_sorting, sort_idxs):
	# we sort the neurons of the trial slices according to their peak activity in the first slice.
	final_sorting_idx = list()  # include the end frame
	ordered_components = components[0][1][0][stimulus_region_sorting]
	
	for i, idx in enumerate(sort_idxs):
		if i == 0:
			comp_idx = np.argsort(np.argmax(ordered_components[:idx+1], axis=1))
			final_sorting_idx.extend(stimulus_region_sorting[:idx+1][comp_idx])
		else:
			comp_idx = np.argsort(np.argmax(ordered_components[sort_idxs[i-1]+1:idx+1], axis=1))
			final_sorting_idx.extend(stimulus_region_sorting[sort_idxs[i-1]+1:idx+1][comp_idx])

	if sort_idxs[-1] != ordered_components.shape[1]:
		comp_idx = np.argsort(np.argmax(ordered_components[sort_idxs[-1]+1:], axis=1))
		final_sorting_idx.extend(stimulus_region_sorting[sort_idxs[-1]+1:][comp_idx])

	return np.array(final_sorting_idx)


def order_by_neuron_components(components, stimulus_region_sorting, sort_idxs):
	final_sorting_idx = list()  # include the end frame
	ordered_components = components[1][0][:, stimulus_region_sorting]

	last_component_idxs_combined = list()  # final indices of the sorted clusters
	neuron = components[1][0].shape[0]  # number of neuron components
	
	for i, idx in enumerate(sort_idxs):
		if i == 0:  # for egg water
			sorted_comps, sorted_idxs = sort_neurons_by_weight(ordered_components[:, :idx+1])
			final_sorting_idx.extend(stimulus_region_sorting[:idx+1][sorted_idxs])

			_, _, sort_max_fac = classify_neurons_by_weights(ordered_components[:, :idx+1])
			# last_component_idxs = np.where(sort_max_fac[:-1] != sort_max_fac[1:])[0]
			# if len(last_component_idxs) == 0:  # if there is only a single stimulus, take the final idx of the final element
			# 	last_component_idxs = np.array([sort_max_fac.shape[0]])
			last_component_idxs = list()
			for n in range(neuron):
				try: 
					last_component_idxs.append(np.where(sort_max_fac == n)[0][-1])
				except IndexError:
					last_component_idxs.append(np.nan)
			last_component_idxs_combined.append(last_component_idxs)
		else:
			sorted_comps, sorted_idxs = sort_neurons_by_weight(ordered_components[:, sort_idxs[i-1]+1:idx+1])
			final_sorting_idx.extend(stimulus_region_sorting[sort_idxs[i-1]+1:idx+1][sorted_idxs])

			_, _, sort_max_fac = classify_neurons_by_weights(ordered_components[:, sort_idxs[i-1]+1:idx+1])
			# last_component_idxs = np.where(sort_max_fac[:-1] != sort_max_fac[1:])[0]
			# if len(last_component_idxs) == 0:  # if there is only a single component class, take the final idx of the final element
			# 	last_component_idxs = np.array([sort_max_fac.shape[0]])
			last_component_idxs = list()
			for n in range(neuron):
				try: 
					last_component_idxs.append(np.where(sort_max_fac == n)[0][-1])
				except IndexError:
					last_component_idxs.append(np.nan)
			last_component_idxs_combined.append(last_component_idxs)

	if sort_idxs[-1] != ordered_components.shape[1]:  # for AITC
		sorted_comps, sorted_idxs = sort_neurons_by_weight(ordered_components[:, sort_idxs[-1]+1:])
		final_sorting_idx.extend(stimulus_region_sorting[sort_idxs[-1]+1:][sorted_idxs])

		_, _, sort_max_fac = classify_neurons_by_weights(ordered_components[:, sort_idxs[-1]+1:])
		# last_component_idxs = np.where(sort_max_fac[:-1] != sort_max_fac[1:])[0]
		# if len(last_component_idxs) == 0:  # if there is only a single stimulus, take the final idx of the final element
		# 	last_component_idxs = np.array([sort_max_fac.shape[0]])
		last_component_idxs = list()
		for n in range(neuron):
			try: 
				last_component_idxs.append(np.where(sort_max_fac == n)[0][-1])
			except IndexError:
				last_component_idxs.append(np.nan)
		last_component_idxs_combined.append(last_component_idxs)

	last_component_idxs_combined = np.array(last_component_idxs_combined)

	return np.array(final_sorting_idx), last_component_idxs_combined


def embed_image(image, default_size=1024):
    """
    puts images inside a black cube - standardizes size and such (helpful sometimes)
    :param image:
    :param default_size:
    :return:
    """
    if image.ndim == 3:
        print("please input 2D image")
        return

    while max(image.shape) > default_size:
        default_size *= 2
        print(f"increasing default size to {default_size}")

    new_image = np.zeros((default_size, default_size))
    midpt = default_size // 2

    image = np.clip(image, a_min=0, a_max=2**16)

    offset_y = 0
    ydim = image.shape[0]
    if ydim % 2 != 0:  # if its odd kick it one pixel
        offset_y += 1
    offset_x = 0
    xdim = image.shape[1]
    if xdim % 2 != 0:  # if its odd kick it one pixel
        offset_x += 1

    new_image[
        midpt - ydim // 2 : midpt + ydim // 2 + offset_y,
        midpt - xdim // 2 : midpt + xdim // 2 + offset_x,
    ] = image

    return new_image


def sort_by_neuron_component(data, classified_neuron_idxs, separate_array=None, sort_within_component=False):
    """
    Sorts data based on neuron components defined in `classified_neuron_idxs` and optionally by peak within each component.

    Parameters:
    - data: Numpy array to be sorted (e.g., neuron responses across time).
    - classified_neuron_idxs: Dictionary where keys are component numbers and values are lists of neuron indices belonging to each component.
    - separate_array: Optional Numpy array to be sorted in the same way as `data`.
    - sort_within_component: Boolean, if True, sorts neurons within each component by their peak.

    Returns:
    - sorted_data: The data sorted by component indices and optionally by peak within each component.
    - sorted_separate_array: The separate array sorted using the same indices (if provided).
    """
    sorting_indices = []

    # Iterate over components in ascending order
    for component in sorted(classified_neuron_idxs.keys()):
        component_indices = classified_neuron_idxs[component]
        
        if sort_within_component:
            # Extract the neurons for this component
            component_data = data[component_indices]
            if separate_array is not None:
                component_separate_array = separate_array[component_indices]
				
                # Sort within component by peak
                component_data, component_separate_array, component_sort_indices = sort_by_peak_with_indices(
                    component_data, separate_array=component_separate_array
                )
            else:
                component_separate_array = None
			
                # Sort within component by peak
                component_data, component_sort_indices = sort_by_peak_with_indices(
                    component_data, separate_array=component_separate_array
                )
            
            # Append the global indices sorted by peak within this component
            sorting_indices.extend([component_indices[i] for i in component_sort_indices])
        else:
            # Append the global indices directly for this component
            sorting_indices.extend(component_indices)

    # Reorder the data and separate array based on the final sorting indices
    sorted_data = data[sorting_indices]
    if separate_array is not None:
        sorted_separate_array = separate_array[sorting_indices]
        return sorted_data, sorted_separate_array

    return sorted_data
