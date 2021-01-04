import _pickle
import os
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def moving_average(x, w=50):
    # https://stackoverflow.com/a/54628145/12097439
    return np.convolve(x, np.ones(w), 'valid') / w


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Plot and save data
    fig, ax = plt.subplots()
    result_df = []
    for folder in os.listdir("data"):
        if os.path.isfile("data/" + folder) \
                or not re.fullmatch("matthew_n_episode_1000_max_steps_1000_epsilon_(.*)"
                                    "_controler_layer_size_128_sub_policy_layer_size_256", folder):
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
        reward_by_episode_mean = moving_average(reward_by_episode_mean)
        reward_by_episode_std = np.array([np.array([elem for elem in df["meta_rewards"]]).mean(axis=(1, 2))
                                          for df in data_lst]).std(axis=0)
        reward_by_episode_std = moving_average(reward_by_episode_std)
        pd.DataFrame(reward_by_episode_mean).plot(ax=ax)
        # plt.fill_between(range(len(reward_by_episode_mean)),
        #                  reward_by_episode_mean - reward_by_episode_std,
        #                  reward_by_episode_mean + reward_by_episode_std, alpha=0.1)
        plt.xlabel("Episodes")
        plt.ylabel("Mean fair-efficient reward")
        # plt.savefig("data/{}/{}".format(folder, "mean_fair_efficient_reward.png"))

    plt.legend(["0.1", "0.2", "0.3"])
    plt.show()
