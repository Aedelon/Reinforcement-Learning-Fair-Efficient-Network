import threading

import job
import matthew
import plant

if __name__ == "__main__":
    thread_list = list()
    n_simu = 3
    n_episode = 1000
    max_steps = 1000

    for i_name in range(n_simu):
        for fun in (job.main_loop, matthew.main_loop, plant.main_loop):
            # Test epsilon of PPO
            for eps in (0.1, 0.2, 0.3):
                thread_list.append(threading.Thread(target=fun, args=("simu{}".format(i_name+1), ),
                                                    kwargs=dict(n_episode=n_episode, max_steps=max_steps, epsilon=eps)))
            # # test network density for controler
            # for layer_size in (64, 128, 256):
            #     controler_layer_size = layer_size
            #     sub_policy_layer_size = 2*layer_size
            #     thread_list.append(threading.Thread(target=fun, args=("simu{}".format(i_name+1), ),
            #                                         kwargs=dict(n_episode=n_episode, max_steps=max_steps,
            #                                                     controler_layer_size=controler_layer_size,
            #                                                     sub_policy_layer_size=sub_policy_layer_size)))

    for thread in thread_list:
        thread.start()
    for thread in thread_list:
        thread.join()
