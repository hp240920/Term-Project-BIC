import matplotlib.image as mpimg
import numpy as np
import os
from lif_model import lif
import matplotlib.pyplot as plot
from lif_model import count_spikes


def validate(weights):
    path_img = ['Validating_set/Red', 'Validating_set/Green', 'Validating_set/Yellow']
    color = 0
    correct = 0
    incorrect = 0
    for paths in path_img:
        for path, dirs, files in os.walk(paths):
            for f in files:
                filename = os.path.join(path, f)
                img1 = mpimg.imread(filename)
                R = img1[:, :, 0]
                G = img1[:, :, 1]

                # R-> 0 , G-> 1, Y-> 2
                pixel_arr = np.append(np.reshape(R, [1, 100]), np.reshape(G, [1, 100]))
                # pixel_arr = np.append(pixel_arr, np.reshape(B, [1, 100]))
                output = weights.dot(np.transpose(pixel_arr))
                current_output = np.argmax(output)
                if current_output == color:
                    correct += 1
                else:
                    incorrect += 1
                # print(count)
            color += 1
            # print(color)
        # print(correct, incorrect)

    return correct / (correct + incorrect)


if __name__ == '__main__':


    corr_constant = 0.1

    accuracies = np.zeros(351)

    weights = np.zeros([3, 200])
    neuron = lif()
    max_spike_rate = 25
    raster_data_input = np.zeros([200, 1004])
    raster_data_output = np.zeros([3, 1004])

    path_img = ['Mix_imgs']
    color = 0
    iteration = 0
    print(validate(weights))
    for paths in path_img:
        for path, dirs, files in os.walk(paths):
            for f in files:
                if iteration > 350:
                    break
                filename = os.path.join(path, f)
                print(filename)
                img1 = mpimg.imread(filename)
                R = img1[:, :, 0]
                G = img1[:, :, 1]
                B = img1[:, :, 2]

                # R-> 0 , G-> 1, Y-> 2
                pixel_arr = np.append(np.reshape(R, [1, 100]), np.reshape(G, [1, 100]))
                # pixel_arr = np.append(pixel_arr, np.reshape(B, [1, 100]))

                # print(count)
                pixel_count = 0

                color = int(f[-5])
                for pixel in pixel_arr:
                    pre_syn_current = pixel
                    neuron.threshold = 0.2
                    neuron.I = pre_syn_current
                    pre_rate = count_spikes(neuron) / max_spike_rate  # between 0 and 1
                    # if pre_rate > 0:
                    # raster_data_input[pixel_count][iteration] = iteration

                    for i in range(3):
                        current = weights[i].dot(np.transpose(pixel_arr))
                        # print(current)
                        neuron.I = current
                        neuron.threshold = 10
                        post_rate = count_spikes(neuron) / max_spike_rate  # between 0 and 1
                        # if post_rate > 0:
                        # raster_data_output[i][iteration] = iteration
                        if i == color:
                            post_rate = 1
                            weights[i][pixel_count] += pre_rate * post_rate * corr_constant
                        else:
                            weights[i][pixel_count] -= pre_rate * post_rate * corr_constant
                    pixel_count += 1
                accu = validate(weights)
                accuracies[iteration] = accu
                print(accu)
                iteration += 1
        color += 1
    print("Accuracy")
    print(validate(weights))
    # print(weights)
    np.save('accuracies_0.1', accuracies)
    # np.save('raster_input_0.7_time_{}'.format(corr_constant), raster_data_input)
    # np.save('raster_output_0.7_time_{}'.format(corr_constant), raster_data_output)
    # np.save('weights_0.7_raster_red_{}'.format(corr_constant),weights)
