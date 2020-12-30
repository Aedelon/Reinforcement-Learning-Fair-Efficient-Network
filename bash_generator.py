if __name__ == "__main__":
    thread_list = list()
    n_simu = 3
    n_episode = 1000
    max_steps = 1000

    with open("template.sh") as template_file:
        template_text = template_file.read().replace("\r", "")

    # args = n_simu, n_episode, max_steps, eps, controler_layer_size, sub_policy_layer_size

    for i_name in range(n_simu):
        # Test epsilon of PPO
        for eps in (0.1, 0.2, 0.3):
            args = [i_name, n_episode, max_steps, eps, 128, 256]
            file_name = "fen_" + "_".join([str(arg) for arg in args]).replace(".", "")
            with open(file_name + ".sh", "w") as file:
                file.write(template_text.format(file_name, *args))
        # test network density for controler
        for controler_layer_size in (64, 128, 256):
            args = [i_name, n_episode, max_steps, 0.2, controler_layer_size, 256]
            file_name = "fen_" + "_".join([str(arg) for arg in args]).replace(".", "")
            with open(file_name + ".sh", "w") as file:
                file.write(template_text.format(file_name, *args))
        # test network density for sub policies
        for sub_policy_layer_size in (128, 256, 512):
            args = [i_name, n_episode, max_steps, 0.2, 128, sub_policy_layer_size]
            file_name = "fen_" + "_".join([str(arg) for arg in args]).replace(".", "")
            with open(file_name + ".sh", "w") as file:
                file.write(template_text.format(file_name, *args))
