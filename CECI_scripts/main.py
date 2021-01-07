import sys
import threading

import job
import matthew
import plant

if __name__ == "__main__":
    thread_list = list()

    args = ((float(arg) if float(arg) < 1 else int(arg)) for arg in sys.argv[1:])
    n_simu, n_episode, max_steps, eps, controler_layer_size, sub_policy_layer_size = args

    for fun in (job.main_loop, matthew.main_loop, plant.main_loop):
        thread_list.append(threading.Thread(target=fun, args=("simu{}".format(n_simu+1), ),
                                            kwargs=dict(n_episode=n_episode, max_steps=max_steps, epsilon=eps,
                                                        controler_layer_size=controler_layer_size,
                                                        sub_policy_layer_size=sub_policy_layer_size)))

    for thread in thread_list:
        thread.start()
    for thread in thread_list:
        thread.join()
