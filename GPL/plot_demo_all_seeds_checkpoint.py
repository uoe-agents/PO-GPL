import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
sns.set(font_scale=1.5)

parser = argparse.ArgumentParser()
# Experiment logistics
parser.add_argument('--env-name', type=str, default="Wolfpack", help="Env name.")
parser.add_argument('--exp-name', type=str, default="exp_1", help="Env name.")
parser.add_argument('--logging-dir', type=str, default="logs1", help="Tensorboard logging directory")
parser.add_argument('--saving-dir', type=str, default="parameters1", help="Parameter saving directory.")
parser.add_argument('--test-type', type=str, default="50", help="Parameter saving directory.")
args = parser.parse_args()


def plot_me(algorithm_name_list, legend, title, ylabel, fig_name, buff_to_load, log_inversion = False):
	"""
	This figure is useful for plotting several type of data. Particularly is useful for plotting: 
	- Reconstruction RMSE error 
	- Reconstruction log error 
	- Correct particle weight
	"""
	fig, ax = plt.subplots()


	legend_count = 0
	num_checkpoints = 40
	num_experiments = 8
	pallette = sns.color_palette()
	print(pallette)
	color_i = -1
	for algorithm_name in algorithm_name_list:
		color_i += 1 
		print(algorithm_name)
		if algorithm_name=="NA":
			legend_count += 1
		else:
			buff_mean = np.zeros((num_experiments,num_checkpoints))
			buff_std = np.zeros((num_experiments,num_checkpoints))
			buff_ci = np.zeros((num_experiments,num_checkpoints))
			for exp_n in range(num_experiments):
				print("algorithm", algorithm_name, exp_n)

				for checkpoint in range(num_checkpoints):

					# This is of shape (n test, episode lengh)
					buff_rmse = np.load(args.saving_dir + "/" + str(algorithm_name)  +  '/exp_' + str(exp_n +1) + '/' + buff_to_load + str(checkpoint) + '.npy')

					buff_mean[exp_n,checkpoint] = buff_rmse.mean(-1).mean(-1)
					


					buff_std[exp_n,checkpoint] = buff_rmse.mean(-1).std(-1)
					# calculate confidence interval 
					buff_ci[exp_n,checkpoint] = 1.96 * buff_rmse.mean(-1).std(-1)/np.sqrt(len(buff_rmse.mean(-1)))
					# print(buff_ci[checkpoint], buff_std[checkpoint])


			algo_mean_per_checkpoint = buff_mean.mean(0)
			algo_ci_per_checkpoint = 1.96 * buff_mean.std(0)/np.sqrt(len(buff_rmse.mean(1)))


			s_dim = len(algo_mean_per_checkpoint)

			x = np.linspace(0, s_dim, s_dim)
			ax.plot(x, algo_mean_per_checkpoint, '-', label=legend[legend_count], color = pallette[color_i])
			
			legend_count += 1
			ax.fill_between(x, algo_mean_per_checkpoint - buff_mean.std(0), algo_mean_per_checkpoint + buff_mean.std(0), alpha=0.2, color = pallette[color_i])

			# plt.axis([0, end_n, 0.4, 0.8])

	# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)

	lgd = ax.legend(framealpha=1, frameon=True, loc='upper left', bbox_to_anchor=(1.01, 1.01));
	plt.title(title)
	plt.ylabel(ylabel)
	plt.xlabel("Total Steps (x160000)")
	# plt.tight_layout()
	plt.savefig(fig_name, dpi=300,  bbox_extra_artists=(lgd,), bbox_inches='tight')
	plt.show()




log_name = ["VAE-GPL", "AE-GPL",  "GPL-Q", "GLANCE-10", "GLANCE-5", "GLANCE-1" ] 
log_dir = ["liam_s", "liam", "gpl" , "glance_10", "glance_5", "glance_1" ] 


legend = ["VAE-GPL","AE-GPL",  "GPL-Q", "PF-GPL-20", "PF-GPL-10", "PF-GPL-5", "PF-GPL-1" ] 
algorithm_name = ["liam_s", "liam", "GPL", "glance_20", "glance_10", "glance_5", "glance_1" ] 


title = "Action Reconstruction in " + args.env_name
ylabel = "Log prob"
fig_name = args.env_name +'_action_log_prob.pdf'
buff_to_load = 'buff_action_reconstruction_log_prob_'
plot_me(algorithm_name, legend, title, ylabel, fig_name, buff_to_load, log_inversion = False)


legend = ["VAE-GPL","AE-GPL" , "", "PF-GPL-20", "PF-GPL-10", "PF-GPL-5", "PF-GPL-1", ] 
algorithm_name = ["liam_s", "liam" , "NA", "glance_20", "glance_10", "glance_5", "glance_1" ] 

title = "State Reconstruction in " + args.env_name
ylabel = "log error"
fig_name = args.env_name + '_state_log_prob.pdf'
buff_to_load = 'buff_state_reconstruction_log_prob_'
plot_me(algorithm_name, legend, title, ylabel, fig_name, buff_to_load, log_inversion = True)


title = "Existence Reconstruction in "+args.env_name
ylabel = "Squarred error"
fig_name = args.env_name +'_existence_avg_error.pdf'
buff_to_load = 'buff_agent_existence_squarred_'
plot_me(algorithm_name, legend, title, ylabel, fig_name, buff_to_load, log_inversion = False)




