import _pickle
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Plot and save data
    result_df = []
    for folder in os.listdir("data"):
        if os.path.isfile("data/" + folder):
            continue

        data_lst = list()
        for file in os.listdir("data/" + folder):
            file = "data/{}/{}".format(folder, file)
            try:
                data_lst.append(pd.read_pickle(file))
            except _pickle.UnpicklingError:
                continue

        # Average aver all runs
        data_mean = pd.DataFrame(index=data_lst[0].index, columns=data_lst[0].columns)
        for index in data_mean.index:
            for col in data_mean.columns:
                data_mean.at[index, col] = np.mean([df.at[index, col] for df in data_lst], axis=0)

        # data_std = pd.DataFrame(index=data_lst[0].index, columns=data_lst[0].columns)
        # for index in data_std.index:
        #     for col in data_std.columns:
        #         data_std.at[index, col] = np.std([df.at[index, col] for df in data_lst], axis=0)

        # with pd.option_context("display.max_columns", None, "display.width", None):
        #     print(data_mean)

        # ----- Plot fair efficient reward over episodes
        reward_by_episode_mean = np.array([episode.mean() for episode in data_mean["meta_rewards"].values])
        reward_by_episode_std = np.array([np.array([elem for elem in df["meta_rewards"]]).mean(axis=(1, 2))
                                          for df in data_lst]).std(axis=0)
        pd.DataFrame(reward_by_episode_mean).plot()
        plt.fill_between(range(len(reward_by_episode_mean)),
                         reward_by_episode_mean - reward_by_episode_std,
                         reward_by_episode_mean + reward_by_episode_std, alpha=0.1)
        plt.xlabel("Episodes")
        plt.ylabel("Mean fair-efficient reward")
        plt.legend([])
        plt.savefig("data/{}/{}".format(folder, "mean_fair_efficient_reward.png"))

        # ----- Plot subpolicy usage over time
        selecting_probability_data = []
        for df in data_lst:
            for i_episode, row_episode in df.iterrows():
                for i_agent in range(df.at[1, "meta_z"].shape[0]):
                    for i_step in range(df.at[1, "meta_z"].shape[1]):
                        selecting_probability_data.append([df.at[i_episode, "rat"][0],
                                                           df.at[i_episode, "meta_z"][i_agent, i_step, 0]])
        selecting_probability_data = pd.DataFrame(selecting_probability_data, columns=["rat", "subpol1"])
        selecting_probability_data.sort_values(by="rat", axis=0, inplace=True, ignore_index=True)

        averaged_selecting_probability_data = []
        for interval_min in np.linspace(-1, 3, 41):
            averaged_selecting_probability_data\
                .append([interval_min, np.array(selecting_probability_data["subpol1"]
                                                [(interval_min < selecting_probability_data["rat"])
                                                 & (selecting_probability_data["rat"] < interval_min+0.1)]).mean()])
        averaged_selecting_probability_data = pd.DataFrame(averaged_selecting_probability_data,
                                                           columns=["rat", "prob_pol1"])

        averaged_selecting_probability_data.plot(x="rat", y="prob_pol1")
        plt.xlabel("$(u_i - \\bar u) / \\bar u$")
        plt.ylabel("Subpolicy selecting probability")
        plt.legend([])
        plt.savefig("data/{}/{}".format(folder, "selecting_probability.png"))

        n_episode = len(data_mean.index)
        result_df.append([folder,
                          data_mean.at[n_episode, "utility"].sum(), np.array([df.at[n_episode, "utility"].sum()
                                                                              for df in data_lst]).std(),
                          data_mean.at[n_episode, "utility"].std(), np.array([df.at[n_episode, "utility"].std()
                                                                              for df in data_lst]).std(),
                          data_mean.at[n_episode, "utility"].min(), np.array([df.at[n_episode, "utility"].min()
                                                                              for df in data_lst]).std(),
                          data_mean.at[n_episode, "utility"].max(), np.array([df.at[n_episode, "utility"].max()
                                                                              for df in data_lst]).std()])

    result_df = pd.DataFrame(result_df, columns=["Params",
                                                 "Resource_utilization", "Resource_utilization_std",
                                                 "CV", "CV_std",
                                                 "min_uti", "min_uti_std",
                                                 "max_uti", "max_uti_std"])
    with pd.option_context("display.max_columns", None, "display.width", None):
        print(result_df)
    result_df.to_csv("data/result_df.csv")
