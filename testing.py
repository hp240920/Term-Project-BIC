import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

if __name__ == '__main__':
    # plt.gray()
    # digits = load_digit()
    # print(len(digits.data[0]))

    # n_samples = len(digits.images)
    # data = digits.images.reshape((n_samples, -1))

    # X_train, X_temp, y_train, y_temp = train_test_split(
    #   data, digits.target, test_size=0.5, shuffle=False)

    # X_validate, X_test, y_validate, y_test = train_test_split(
    #     X_temp, y_temp, test_size=0.5, shuffle=False)

    # weights = np.zeros([3, 7500])
    weights = np.load('weights_0.7.npy')
    correct = 0
    incorrect = 0
    # path_img = ['Rotated_img_10/Red', 'Rotated_img_10/Green','Rotated_img_10/Yellow']
    path_img = ['coverted_img_1']
    color = 0
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
                output[1] = 100
                plt.bar(['Red', 'Green', 'Yellow'], output, width=0.3)
                plt.show()
                if current_output == color:
                    correct += 1
                else:
                    incorrect += 1

                # print(count)
            color += 1
            print(correct,incorrect)
            print(100 * (correct / (correct + incorrect)))
            correct = 0
            incorrect = 0
            # print(color)

    print(color)

    # print(weights)
    # np.save('weights_2', weights)
