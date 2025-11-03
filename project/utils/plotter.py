import matplotlib.pyplot as plt
import pandas as pd

def plot(result_path, test_results, output_path='results/plots/accuracy.png'):
    df = pd.read_csv(result_path)
    df_test = pd.read_csv(test_results)
    plt.title('Accuracy plot')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy [%]')
    plt.xlim(1, max(df['epoch']))
    plt.plot(df['epoch'], df['train_accuracy'], label="Train accuracy")
    plt.plot(df_test['epoch'], df_test['test_accuracy'], label="Test accuracy")
    plt.legend()
    plt.savefig(output_path)#, transparent=True)
    plt.clf()

def plot_loss(result_path, test_results, output_path='results/plots/losses.png'):
    df = pd.read_csv(result_path)
    df_test = pd.read_csv(test_results)
    plt.title('Loss plot')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xlim(1, max(df['epoch']))
    plt.plot(df['epoch'], df['train_avg_loss'], label="Train loss")
    plt.plot(df_test['epoch'], df_test['test_avg_loss'], label="Test loss")
    plt.legend()
    plt.savefig(output_path)#, transparent=True)
    plt.clf()


if __name__ == '__main__':

    plot(
        result_path='results/train_results.csv',
        test_results='results/test_results.csv',
        output_path='results/plots/accuracy.png'
        )
    
    plot_loss(
        result_path='results/train_results.csv',
        test_results='results/test_results.csv',
        output_path='results/plots/losses.png'
        )
    # plot(
    #     result_path='results/test_results.csv',
    #     output_path='results/plots/test_accuracy.png'
    #     )