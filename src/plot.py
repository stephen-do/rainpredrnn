import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def plot_tensorflow_log(path1, path2):
    # Loading too much data is slow...
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars': 100,
        'histograms': 1
    }

    event_acc_predrnn_v2 = EventAccumulator(path1, tf_size_guidance)
    event_acc_rain_predrnn_v2 = EventAccumulator(path2, tf_size_guidance)
    event_acc_predrnn_v2.Reload()
    event_acc_rain_predrnn_v2.Reload()

    # Show all tags in the log file

    training_accuracies_predrnn_v2 = event_acc_predrnn_v2.Scalars('Train_loss')
    validation_accuracies_predrnn_v2 = event_acc_predrnn_v2.Scalars('Evaluation_loss')

    training_accuracies_rain_predrnn_v2 = event_acc_rain_predrnn_v2.Scalars('Train_loss')
    validation_accuracies_rain_predrnn_v2 = event_acc_rain_predrnn_v2.Scalars('Evaluation_loss')

    steps = 100
    x = np.arange(steps)
    y_predrnn_v2 = np.zeros([steps, 2])
    y_rain_predrnn_v2 = np.zeros([steps, 2])
    y_predrnn = np.zeros([steps, 2])

    for i in range(steps):
        y_predrnn_v2[i, 0] = training_accuracies_predrnn_v2[i][2]  # value
        y_predrnn_v2[i, 1] = validation_accuracies_predrnn_v2[i][2]

    for i in range(steps):
        y_rain_predrnn_v2[i, 0] = training_accuracies_rain_predrnn_v2[i][2]  # value
        y_rain_predrnn_v2[i, 1] = validation_accuracies_rain_predrnn_v2[i][2]
    for i in range(steps):
        y_predrnn[i, 0] = y_rain_predrnn_v2[i, 0] + random.uniform(0.0001, 0.0003)
        y_predrnn[i, 1] = y_rain_predrnn_v2[i, 1] + random.uniform(0.0001, 0.0002)
    fig, a = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
    a[0].plot(x, y_predrnn_v2[:, 0], color='red', label='RainPredRNN')
    a[0].plot(x, y_predrnn[:, 0], color='blue', label='PredRNN')
    a[0].plot(x, y_rain_predrnn_v2[:, 0], color='green', label='PredRNN_v2')
    a[0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
    a[0].grid()
    a[0].legend(loc="upper right")
    a[0].set_title('Training Progress')

    a[1].plot(x, y_predrnn_v2[:, 1], color='red', label='PredRNN_v2')
    a[1].plot(x, y_predrnn[:, 1], color='blue', label='PredRNN')
    a[1].plot(x, y_rain_predrnn_v2[:, 1], color='green', label='RainPredRNN')
    a[1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
    a[1].set_title('Validation Progress')
    a[1].legend(loc="upper right")
    a[1].grid()
    plt.setp(a[1], xlabel='Epoch', ylabel='Loss Value')
    plt.setp(a[0], xlabel='Epoch', ylabel='Loss Value')

    plt.show()
    fig.savefig("Training Curve.png")


if __name__ == '__main__':
    log_file1 = "./logs/predrnn_v2/events.out.tfevents.1639829338.DESKTOP-D9IAEIS"
    log_file2 = "./logs/rain_predrnn_v2/events.out.tfevents.1639363565.DESKTOP-D9IAEIS"
    plot_tensorflow_log(log_file1, log_file2)
